import io
import re
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import streamlit as st

MODEL_CHOICES = {
    "SVM (word TF-IDF) — baseline": "data/pkls/svm.pkl",
    "SVM (character TF-IDF) — higher accuracy": "data/pkls/svm_char.pkl",
}

SENTIMENT_META = {
    "positive": {
        "label": "Positive",
        "color": "#16a34a",
        "bg": "rgba(22, 163, 74, 0.10)",
        "explanation": "Indicates strong financial performance",
    },
    "neutral": {
        "label": "Neutral",
        "color": "#f59e0b",
        "bg": "rgba(245, 158, 11, 0.12)",
        "explanation": "Informational or balanced tone",
    },
    "negative": {
        "label": "Negative",
        "color": "#ef4444",
        "bg": "rgba(239, 68, 68, 0.10)",
        "explanation": "Indicates financial risk or decline",
    },
}

SAMPLE_TEXT = """Tesla reported record quarterly revenue, beating analyst expectations.
Gross margins improved as production costs declined, and management raised guidance for the next quarter."""

@dataclass(frozen=True)
class PredictionResult:
    label: str
    raw_class: str
    confidence: Optional[float]
    explanation: str

def _split_candidate_lines(text: str) -> list[str]:
    """
    Build candidate units to explain predictions:
    - Prefer natural newlines if the document already has them.
    - Fall back to sentence chunks for single-paragraph text.
    """
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    if len(lines) >= 2:
        return lines

    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", text or "") if p.strip()]
    return parts or lines

def _top_k_for_input(text: str, available_items: int) -> int:
    """
    Return a dynamic number of evidence lines based on input size.
    """
    length = len(text or "")
    if length < 280:
        wanted = 1
    elif length < 900:
        wanted = 3
    elif length < 2500:
        wanted = 5
    else:
        wanted = 7
    return max(1, min(wanted, available_items))

def _score_text_for_class(model, text: str, target_class: str) -> Optional[float]:
    cleaned = preprocess_text(text)
    if not cleaned:
        return None

    classes = getattr(model, "classes_", None)
    if classes is None:
        return None

    class_labels = [str(c).lower() for c in classes]
    if target_class not in class_labels:
        return None
    target_idx = class_labels.index(target_class)

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba([cleaned])[0]
        return float(probs[target_idx])

    if hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function([cleaned]))
        if scores.ndim == 1:
            # Binary case: sklearn margin sign corresponds to classes_[1].
            margin = float(scores[0])
            signed_margin = margin if target_idx == 1 else -margin
            return float(1.0 / (1.0 + np.exp(-signed_margin)))

        margins = np.asarray(scores[0], dtype=np.float64)
        probs = _softmax(margins)
        return float(probs[target_idx])

    # Last-resort fallback when probability/margins are unavailable.
    pred = str(model.predict([cleaned])[0]).lower()
    return 1.0 if pred == target_class else 0.0

def top_supporting_lines(model, text: str, target_class: str) -> list[Tuple[str, float]]:
    candidates = _split_candidate_lines(text)
    scored: list[Tuple[str, float]] = []
    for line in candidates:
        score = _score_text_for_class(model, line, target_class)
        if score is not None:
            scored.append((line, score))

    scored.sort(key=lambda item: (item[1], len(item[0])), reverse=True)
    keep = _top_k_for_input(text, len(scored))
    return scored[:keep]

def preprocess_text(text: str) -> str:
    """
    Basic preprocessing to match training-time expectations:
    - lowercase
    - remove special characters via regex (keep letters, digits, whitespace)
    """
    text = (text or "").lower()
    text = re.sub(r"[^a-z0-9\s]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def _softmax(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    x = x - np.max(x)
    ex = np.exp(x)
    return ex / np.sum(ex)

@st.cache_resource(show_spinner=False)
def load_model(model_path: str):
    import joblib

    return joblib.load(model_path)

def predict_sentiment(model, text: str) -> PredictionResult:
    cleaned = preprocess_text(text)
    if not cleaned:
        raise ValueError("Empty text after preprocessing.")

    classes = getattr(model, "classes_", None)
    if classes is None:
        raise RuntimeError("Model does not expose `classes_`; cannot map predictions.")

    # Prefer decision_function for SVM-style models; fall back to predict_proba if available.
    confidence: Optional[float] = None
    raw_class: str

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba([cleaned])[0]
        idx = int(np.argmax(probs))
        raw_class = str(classes[idx])
        confidence = float(probs[idx])
    elif hasattr(model, "decision_function"):
        scores = model.decision_function([cleaned])
        scores = np.asarray(scores)
        if scores.ndim == 1:
            # Binary case -> two classes; map signed distance to pseudo-probability.
            # confidence is the model's certainty proxy, not calibrated probability.
            raw_pred = model.predict([cleaned])[0]
            raw_class = str(raw_pred)
            confidence = float(1.0 / (1.0 + np.exp(-abs(scores[0]))))
        else:
            # Multiclass case -> softmax over margins to get a usable confidence proxy.
            margins = scores[0]
            probs = _softmax(margins)
            idx = int(np.argmax(probs))
            raw_class = str(classes[idx])
            confidence = float(probs[idx])
    else:
        raw_class = str(model.predict([cleaned])[0])

    key = raw_class.lower()
    meta = SENTIMENT_META.get(key, None)
    if meta is None:
        # Gracefully handle unexpected class labels.
        meta = {
            "label": raw_class,
            "color": "#64748b",
            "bg": "rgba(100, 116, 139, 0.10)",
            "explanation": "Model returned an unrecognized label",
        }

    return PredictionResult(
        label=meta["label"],
        raw_class=key,
        confidence=confidence,
        explanation=meta["explanation"],
    )

def _require_ocr_deps():
    """
    Import OCR dependencies lazily so manual input mode works even if OCR libs
    aren't installed. Provides user-friendly error guidance.
    """
    try:
        import pytesseract  # noqa: F401
    except Exception as e:
        raise RuntimeError(
            "OCR requires `pytesseract` and the Tesseract engine installed.\n\n"
            "Windows quick steps:\n"
            "- Install Tesseract OCR\n"
            "- Ensure `tesseract` is on PATH (or set TESSERACT_CMD)\n\n"
            f"Import error: {e}"
        ) from e

    try:
        from PIL import Image  # noqa: F401
    except Exception as e:
        raise RuntimeError(f"OCR requires Pillow (`pillow`). Import error: {e}") from e

def _configure_tesseract_if_needed():
    """
    Optional: allow users to set a custom tesseract command path without code edits.
    """
    import os

    tesseract_cmd = os.environ.get("TESSERACT_CMD", "").strip()
    if tesseract_cmd:
        import pytesseract

        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

def ocr_image_bytes(image_bytes: bytes) -> str:
    _require_ocr_deps()
    _configure_tesseract_if_needed()

    from PIL import Image
    import pytesseract

    img = Image.open(io.BytesIO(image_bytes))
    # A small boost for OCR consistency on receipts/scans.
    img = img.convert("L")
    return (pytesseract.image_to_string(img) or "").strip()

def ocr_pdf_bytes(pdf_bytes: bytes, max_pages: int = 25) -> str:
    _require_ocr_deps()
    _configure_tesseract_if_needed()

    try:
        from pdf2image import convert_from_bytes
    except Exception as e:
        raise RuntimeError(
            "PDF OCR requires `pdf2image` and Poppler installed.\n\n"
            "Windows quick steps:\n"
            "- Install Poppler for Windows\n"
            "- Add Poppler `bin` folder to PATH\n\n"
            f"Import error: {e}"
        ) from e

    import pytesseract

    # Convert PDF pages to PIL Images (uses Poppler under the hood).
    # If Poppler is not on PATH, users can provide its folder via POPPLER_PATH.
    import os

    poppler_path = os.environ.get("POPPLER_PATH", "").strip() or None
    pages = convert_from_bytes(pdf_bytes, poppler_path=poppler_path)
    if not pages:
        return ""

    pages = pages[: max_pages or len(pages)]
    chunks = []
    for i, page in enumerate(pages, start=1):
        page = page.convert("L")
        text = (pytesseract.image_to_string(page) or "").strip()
        if text:
            chunks.append(text)
        # Keep memory lower by dropping references as we go.
        del page

    return "\n\n".join(chunks).strip()

def render_card(result: PredictionResult) -> None:
    meta = SENTIMENT_META.get(result.raw_class, None)
    color = (meta or {}).get("color", "#64748b")
    bg = (meta or {}).get("bg", "rgba(100, 116, 139, 0.10)")

    conf_html = ""
    if result.confidence is not None:
        conf_html = f"""
        <div class="fs-sub">
          Confidence: <span class="fs-mono">{result.confidence:.2%}</span>
        </div>
        """

    st.markdown(
        f"""
        <div class="fs-card" style="border-left-color:{color}; background:{bg};">
          <div class="fs-title" style="color:{color};">{result.label}</div>
          {conf_html}
          <div class="fs-sub">{result.explanation}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def render_supporting_lines(lines: list[Tuple[str, float]], result_label: str) -> None:
    st.markdown("#### Top supporting lines")
    if not lines:
        st.caption("No line-level explanation available for this input/model.")
        return

    min_confidence = 0.60
    filtered = [(line, score) for line, score in lines if score > min_confidence]
    if not filtered:
        st.caption(
            f"No lines exceeded {min_confidence:.0%} support confidence for **{result_label}** sentiment."
        )
        return

    st.caption(
        f"Showing {len(filtered)} lines most related to **{result_label}** sentiment "
        f"(support confidence > {min_confidence:.0%})."
    )
    for idx, (line, score) in enumerate(filtered, start=1):
        st.markdown(f"**{idx}.** {line}")
        st.caption(f"Support score: {score:.2%}")

def _inject_css():
    st.markdown(
        """
        <style>
          .block-container { padding-top: 2.2rem; padding-bottom: 2rem; }
          .fs-hero {
            padding: 1.1rem 1.2rem;
            border: 1px solid rgba(148, 163, 184, 0.30);
            border-radius: 18px;
            background: radial-gradient(1200px 300px at 10% 0%, rgba(56, 189, 248, 0.20), transparent 60%),
                        radial-gradient(1200px 300px at 90% 10%, rgba(168, 85, 247, 0.18), transparent 55%),
                        linear-gradient(180deg, rgba(2, 6, 23, 0.03), rgba(2, 6, 23, 0.00));
          }
          .fs-hero h1 { margin: 0 0 0.25rem 0; font-size: 2rem; letter-spacing: -0.02em; }
          .fs-hero p { margin: 0; color: rgba(15, 23, 42, 0.70); font-size: 1.02rem; }
          .fs-card {
            padding: 1.05rem 1.1rem;
            border-radius: 18px;
            border: 1px solid rgba(148, 163, 184, 0.25);
            border-left: 8px solid #64748b;
            box-shadow: 0 10px 30px rgba(2, 6, 23, 0.08);
          }
          .fs-title { font-size: 1.55rem; font-weight: 800; margin-bottom: 0.25rem; }
          .fs-sub { font-size: 0.98rem; color: rgba(15, 23, 42, 0.75); }
          .fs-mono { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; }
          .fs-help {
            padding: 0.9rem 1rem;
            border-radius: 16px;
            border: 1px dashed rgba(148, 163, 184, 0.55);
            background: rgba(248, 250, 252, 0.55);
          }
          @media (prefers-color-scheme: dark) {
            .fs-hero p, .fs-sub { color: rgba(226, 232, 240, 0.75) !important; }
            .fs-hero { border-color: rgba(148, 163, 184, 0.22); background:
              radial-gradient(1200px 300px at 10% 0%, rgba(56, 189, 248, 0.18), transparent 60%),
              radial-gradient(1200px 300px at 90% 10%, rgba(168, 85, 247, 0.16), transparent 55%),
              linear-gradient(180deg, rgba(2, 6, 23, 0.55), rgba(2, 6, 23, 0.10)); }
            .fs-card { border-color: rgba(148, 163, 184, 0.18); box-shadow: 0 12px 40px rgba(0,0,0,0.35); }
            .fs-help { background: rgba(2, 6, 23, 0.25); border-color: rgba(148, 163, 184, 0.30); }
          }
        </style>
        """,
        unsafe_allow_html=True,
    )

def init_state():
    st.session_state.setdefault("extracted_text", "")
    st.session_state.setdefault("show_full_text", False)

def clear_all():
    st.session_state["extracted_text"] = ""
    st.session_state["show_full_text"] = False

def main():
    st.set_page_config(
        page_title="Financial Sentiment Analyzer",
        layout="centered",
        initial_sidebar_state="collapsed",
    )

    _inject_css()
    init_state()

    st.markdown(
        """
        <div class="fs-hero">
          <h1>Financial Sentiment Analyzer</h1>
          <p>Analyze financial text using machine learning</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")

    with st.sidebar:
        st.markdown("### System setup")
        st.markdown("### Model")
        model_label = st.selectbox("Choose a model", list(MODEL_CHOICES.keys()))
        model_path = MODEL_CHOICES[model_label]
        st.caption(f"Using: `{model_path}`")
        st.markdown("### PDF OCR")
        st.caption("Optional: set Poppler folder if PDFs fail with page-count errors.")
        st.text_input(
            "Poppler path (sets POPPLER_PATH for this session)",
            value="",
            key="poppler_path_input",
            help="Example: C:\\path\\to\\poppler\\Library\\bin",
        )
        st.markdown(
            """
            <div class="fs-help">
              <div><b>OCR note (Windows)</b></div>
              <div style="margin-top:0.35rem;">
                - Install <b>Tesseract OCR</b> (required for images & PDFs)<br/>
                - Install <b>Poppler</b> (required for PDFs)<br/>
                - If Tesseract isn't on PATH, set env var <span class="fs-mono">TESSERACT_CMD</span>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    tabs = st.tabs(["Upload Document", "Enter Text"])

    extracted_text = ""

    with tabs[0]:
        uploaded = st.file_uploader(
            "Upload a financial document (PDF or image)",
            type=["pdf", "png", "jpg", "jpeg"],
            accept_multiple_files=False,
        )

        col_a, col_b, col_c = st.columns([1, 1, 1])
        with col_a:
            run_ocr = st.button("Extract Text (OCR)", use_container_width=True, disabled=uploaded is None)
        with col_b:
            show_example = st.button("Use Sample Text", use_container_width=True)
        with col_c:
            do_clear = st.button("Clear / Reset", use_container_width=True)

        if do_clear:
            clear_all()
            st.success("Cleared.")

        if show_example:
            st.session_state["extracted_text"] = SAMPLE_TEXT
            st.success("Loaded sample text. Scroll down to analyze.")

        if run_ocr and uploaded is not None:
            try:
                with st.spinner("Running OCR..."):
                    file_bytes = uploaded.getvalue()
                    ext = (uploaded.name.split(".")[-1] or "").lower()

                    if ext == "pdf":
                        poppler_path = (st.session_state.get("poppler_path_input", "") or "").strip()
                        if poppler_path:
                            import os

                            # pdf2image reads POPPLER_PATH; setting it here avoids requiring PATH configuration.
                            os.environ["POPPLER_PATH"] = poppler_path
                        text = ocr_pdf_bytes(file_bytes)
                    else:
                        text = ocr_image_bytes(file_bytes)

                if not text.strip():
                    st.error("OCR completed but returned empty text. Try a clearer scan or higher-resolution image.")
                else:
                    st.session_state["extracted_text"] = text
                    st.success("Text extracted successfully.")
            except Exception as e:
                st.error(f"OCR failed: {e}")

        extracted_text = st.session_state.get("extracted_text", "") or ""

    with tabs[1]:
        manual = st.text_area(
            "Enter financial text",
            placeholder="Enter financial news, report, or statement...",
            height=220,
        )

        col_a, col_b, col_c = st.columns([1, 1, 1])
        with col_a:
            use_manual = st.button("Use This Text", use_container_width=True)
        with col_b:
            show_example2 = st.button("Use Sample Text", use_container_width=True, key="sample2")
        with col_c:
            do_clear2 = st.button("Clear / Reset", use_container_width=True, key="clear2")

        if do_clear2:
            clear_all()
            st.success("Cleared.")

        if show_example2:
            st.session_state["extracted_text"] = SAMPLE_TEXT
            st.success("Loaded sample text. Scroll down to analyze.")

        if use_manual:
            st.session_state["extracted_text"] = manual or ""
            if (manual or "").strip():
                st.success("Text captured. Scroll down to analyze.")
            else:
                st.warning("No text entered.")

        extracted_text = st.session_state.get("extracted_text", "") or extracted_text

    st.divider()
    st.markdown("### Extracted Text")

    if not (extracted_text or "").strip():
        st.info("Upload a document and run OCR, or enter text manually to begin.")
    else:
        preview_len = 900
        preview = extracted_text[:preview_len]
        st.text_area("Preview", value=preview, height=180, disabled=True)

        col1, col2 = st.columns([1, 2])
        with col1:
            if st.button("Show Full Text", use_container_width=True):
                st.session_state["show_full_text"] = not st.session_state.get("show_full_text", False)
        with col2:
            st.caption(f"Showing first {min(len(extracted_text), preview_len)} of {len(extracted_text)} characters.")

        if st.session_state.get("show_full_text", False):
            st.text_area("Full Text", value=extracted_text, height=260, disabled=True)

    st.write("")
    st.markdown("### Prediction")

    col_left, col_right = st.columns([2, 1])
    with col_left:
        analyze = st.button("Analyze Sentiment", use_container_width=True)
    with col_right:
        st.caption("Tip: cleaner text → better results")

    if analyze:
        if not (extracted_text or "").strip():
            st.warning("No input detected. Please upload a document or enter text first.")
        else:
            try:
                with st.spinner("Loading model and analyzing..."):
                    model = load_model(model_path)
                    result = predict_sentiment(model, extracted_text)
                    supporting = top_supporting_lines(model, extracted_text, result.raw_class)

                render_card(result)
                render_supporting_lines(supporting, result.label)
            except Exception as e:
                st.error(f"Analysis failed: {e}")

    st.write("")
    st.caption("Built with NLP + ML")

if __name__ == "__main__":
    main()


