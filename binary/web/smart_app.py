import os
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image, UnidentifiedImageError

st.set_page_config(
    page_title="Diabetic Retinopathy Detector",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_DIR = os.path.dirname(__file__)
MODEL_CANDIDATES = [
    os.environ.get("MODEL_PATH"),
    os.path.abspath(os.path.join(APP_DIR, "..", "models", "quick_test_model.h5")),
    os.path.abspath(os.path.join(APP_DIR, "..", "..", "models", "quick_test_model_FIXED.h5")),
    os.path.abspath(os.path.join(APP_DIR, "..", "..", "models", "quick_test_model.h5")),
]
RESOLVED_MODEL_PATH = next((path for path in MODEL_CANDIDATES if path and os.path.exists(path)), None)

CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]


def inject_styles(theme_mode):
    dark_mode = theme_mode == "Black / Premium"
    if dark_mode:
        app_bg = (
            "radial-gradient(circle at top left, rgba(255, 255, 255, 0.06), transparent 24%), "
            "radial-gradient(circle at top right, rgba(255, 255, 255, 0.03), transparent 22%), "
            "linear-gradient(180deg, #08090c 0%, #111318 100%)"
        )
        hero_bg = "linear-gradient(135deg, rgba(18, 20, 26, 0.96), rgba(28, 31, 40, 0.96))"
        panel_bg = "rgba(15, 17, 22, 0.92)"
        output_bg = "linear-gradient(180deg, rgba(16, 18, 24, 0.96), rgba(11, 12, 15, 0.98))"
        heatmap_bg = "rgba(9, 10, 12, 0.98)"
        legend_bg = "rgba(18, 20, 26, 0.96)"
        border = "rgba(255, 255, 255, 0.10)"
        shadow = "rgba(0, 0, 0, 0.34)"
        primary = "#f4f7fb"
        secondary = "#c6d0db"
        muted = "#9aa6b3"
        badge_bg = "rgba(255, 255, 255, 0.08)"
        badge_fg = "#f4f7fb"
    else:
        app_bg = (
            "radial-gradient(circle at top left, rgba(0, 0, 0, 0.05), transparent 24%), "
            "radial-gradient(circle at top right, rgba(18, 65, 112, 0.10), transparent 22%), "
            "linear-gradient(180deg, #f6fbff 0%, #eef5ff 100%)"
        )
        hero_bg = "linear-gradient(135deg, rgba(255, 255, 255, 0.92), rgba(241, 247, 255, 0.92))"
        panel_bg = "rgba(255, 255, 255, 0.92)"
        output_bg = "linear-gradient(180deg, rgba(255,255,255,0.94), rgba(248,250,253,0.96))"
        heatmap_bg = "white"
        legend_bg = "#ffffff"
        border = "rgba(10, 10, 12, 0.08)"
        shadow = "rgba(10, 10, 12, 0.10)"
        primary = "#0b0c0f"
        secondary = "#3e4754"
        muted = "#5b6573"
        badge_bg = "#eceff3"
        badge_fg = "#111318"

    css = """
    <style>
    .stApp {
        background: %s;
    }
    .hero {
        padding: 1.5rem 1.7rem;
        border-radius: 24px;
        background: %s;
        border: 1px solid %s;
        box-shadow: 0 18px 44px %s;
        margin-bottom: 1rem;
    }
    .hero h1 {
        margin: 0;
        font-size: 2.15rem;
        line-height: 1.1;
        color: %s;
    }
    .hero p {
        margin: 0.55rem 0 0;
        color: %s;
        font-size: 1rem;
    }
    .report-shell {
        padding: 1.15rem 1.2rem;
        border-radius: 22px;
        border: 1px solid %s;
        background: %s;
        box-shadow: 0 12px 34px %s;
        margin: 1rem 0 0.75rem;
    }
    .report-top {
        display: flex;
        justify-content: space-between;
        gap: 1rem;
        align-items: flex-start;
        flex-wrap: wrap;
    }
    .report-kicker {
        text-transform: uppercase;
        letter-spacing: 0.14em;
        color: %s;
        font-size: 0.72rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .report-title {
        margin: 0;
        font-size: 1.35rem;
        color: %s;
    }
    .report-subtitle {
        margin: 0.3rem 0 0;
        color: %s;
        font-size: 0.95rem;
        max-width: 68ch;
    }
    .badge {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        padding: 0.45rem 0.75rem;
        border-radius: 999px;
        background: %s;
        color: %s;
        font-weight: 700;
        font-size: 0.84rem;
        border: 1px solid %s;
    }
    .badge.good { background: #eef8f0; color: #17562c; }
    .badge.warn { background: #fff7e8; color: #845500; }
    .badge.bad { background: #ffecef; color: #931822; }
    .metric-card {
        padding: 1rem 1.1rem;
        border-radius: 18px;
        background: %s;
        border: 1px solid %s;
        box-shadow: 0 10px 24px %s;
    }
    .heatmap-frame {
        padding: 1rem;
        border-radius: 18px;
        background: %s;
        border: 1px solid %s;
        box-shadow: 0 10px 22px %s;
    }
    .output-shell {
        padding: 1rem 1.05rem;
        border-radius: 22px;
        border: 1px solid %s;
        background: %s;
        box-shadow: 0 12px 34px %s;
        margin: 0.8rem 0 0.7rem;
    }
    .output-label {
        font-size: 0.74rem;
        text-transform: uppercase;
        letter-spacing: 0.16em;
        color: %s;
        font-weight: 800;
    }
    .output-title {
        margin: 0.08rem 0 0;
        font-size: 1.05rem;
        color: %s;
    }
    .heatmap-legend {
        display: flex;
        gap: 0.45rem;
        flex-wrap: wrap;
        margin-top: 0.45rem;
    }
    .legend-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        padding: 0.38rem 0.62rem;
        border-radius: 999px;
        border: 1px solid %s;
        background: %s;
        color: %s;
        font-size: 0.83rem;
        font-weight: 600;
    }
    .soft-note {
        color: %s;
        font-size: 0.92rem;
    }
    .status-ok {
        color: #116d34;
        font-weight: 700;
    }
    .status-bad {
        color: #9f1d24;
        font-weight: 700;
    }
    </style>
    """ % (
        app_bg,
        hero_bg,
        border,
        shadow,
        primary,
        secondary,
        border,
        panel_bg,
        shadow,
        muted,
        primary,
        secondary,
        badge_bg,
        badge_fg,
        border,
        panel_bg,
        border,
        shadow,
        heatmap_bg,
        border,
        shadow,
        border,
        output_bg,
        shadow,
        muted,
        primary,
        border,
        legend_bg,
        badge_fg,
        muted,
    )

    st.markdown(css, unsafe_allow_html=True)


def load_model():
    model_path = RESOLVED_MODEL_PATH
    if not model_path:
        return None, "No model file was found."

    try:
        import tensorflow as tf
    except ImportError:
        return None, "TensorFlow is not installed in the current environment."

    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        return model, None
    except Exception as e:
        return None, f"Failed to load model from {model_path}: {e}"


def assess_retina_image(image):
    """Return (is_retina_like, reason) for an uploaded image."""
    img = np.array(image.convert("RGB"))
    img = cv2.resize(img, (256, 256))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    avg_saturation = float(np.mean(hsv[:, :, 1]))
    avg_value = float(np.mean(hsv[:, :, 2]))

    h, w = gray.shape
    border = max(1, int(min(h, w) * 0.14))
    center = gray[border : h - border, border : w - border]
    border_mask = np.zeros_like(gray, dtype=bool)
    border_mask[:border, :] = True
    border_mask[-border:, :] = True
    border_mask[:, :border] = True
    border_mask[:, -border:] = True

    border_mean = float(np.mean(gray[border_mask]))
    center_mean = float(np.mean(center)) if center.size else float(np.mean(gray))
    bright_ratio = float(np.mean(gray > 20))

    if avg_saturation < 25:
        return False, "Low saturation suggests this is not a retinal fundus image."

    if avg_value < 35:
        return False, "The image is too dark to confidently identify as a retina photo."

    if bright_ratio > 0.92 and border_mean > 30:
        return False, "This looks like a normal photo or screenshot, not a fundus image."

    if center_mean - border_mean < 8:
        return False, "The image does not have the circular fundus appearance we expect."

    return True, ""


def preprocess_image(image):
    """Convert any uploaded image to RGB and resize it for the model."""
    rgb_image = image.convert("RGB")
    img = np.array(rgb_image)
    img = cv2.resize(img, (128, 128))
    img = img.astype(np.float32) / 255.0
    return img


def predict_image(model, image):
    input_tensor = np.expand_dims(preprocess_image(image), axis=0)
    raw = model.predict(input_tensor, verbose=0)
    raw_pred = np.array(raw)

    if raw_pred.ndim == 2 and raw_pred.shape[1] == 1:
        prediction = float(raw_pred[0, 0])
        return {
            "mode": "binary",
            "raw": raw_pred,
            "prediction": prediction,
            "label": "Diabetic Retinopathy Detected" if prediction > 0.7 else "No Diabetic Retinopathy Detected",
            "confidence": prediction if prediction > 0.7 else (1 - prediction),
        }

    probs = raw_pred[0]
    if probs.sum() > 1.001 or probs.sum() < 0.999:
        shifted = probs - np.max(probs)
        exp_probs = np.exp(shifted)
        probs = exp_probs / np.sum(exp_probs)

    top_idx = int(np.argmax(probs))
    return {
        "mode": "multiclass",
        "raw": raw_pred,
        "probabilities": probs,
        "label": CLASS_NAMES[top_idx],
        "confidence": float(probs[top_idx]),
        "top_idx": top_idx,
    }


def get_tf():
    import tensorflow as tf

    return tf


def find_last_conv_layer(model):
    tf = get_tf()
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    return None


def make_gradcam_heatmap(model, image, class_index=None):
    tf = get_tf()
    last_conv_layer_name = find_last_conv_layer(model)
    if not last_conv_layer_name:
        return None

    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output],
    )

    image_array = np.expand_dims(preprocess_image(image), axis=0)
    image_tensor = tf.convert_to_tensor(image_array)

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image_tensor)
        if predictions.shape[-1] == 1:
            score = predictions[:, 0]
        else:
            index = int(class_index or 0)
            score = predictions[:, index]

    grads = tape.gradient(score, conv_outputs)
    if grads is None:
        return None

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    max_value = tf.reduce_max(heatmap)
    if float(max_value) == 0.0:
        return None
    heatmap /= max_value
    return heatmap.numpy()


def make_saliency_heatmap(model, image, class_index=None):
    tf = get_tf()
    image_array = np.expand_dims(preprocess_image(image), axis=0)
    image_tensor = tf.convert_to_tensor(image_array)

    with tf.GradientTape() as tape:
        tape.watch(image_tensor)
        predictions = model(image_tensor, training=False)
        if predictions.shape[-1] == 1:
            score = predictions[:, 0]
        else:
            index = int(class_index or 0)
            score = predictions[:, index]

    grads = tape.gradient(score, image_tensor)
    if grads is None:
        return None

    saliency = tf.reduce_max(tf.abs(grads), axis=-1)[0].numpy()
    saliency = cv2.GaussianBlur(saliency, (0, 0), 3)
    saliency = saliency - np.min(saliency)
    max_value = float(np.max(saliency))
    if max_value == 0.0:
        return None
    return saliency / max_value


def generate_attention_heatmap(model, image, class_index=None):
    heatmap = None
    method = None

    try:
        heatmap = make_gradcam_heatmap(model, image, class_index)
        if heatmap is not None:
            method = "Grad-CAM"
    except Exception:
        heatmap = None

    if heatmap is None:
        try:
            heatmap = make_saliency_heatmap(model, image, class_index)
            if heatmap is not None:
                method = "Saliency fallback"
        except Exception:
            heatmap = None

    if heatmap is None:
        base = np.zeros((128, 128), dtype=np.float32)
        yy, xx = np.mgrid[:128, :128]
        center_x, center_y = 64, 64
        radius = 52.0
        fallback = np.exp(-(((xx - center_x) ** 2 + (yy - center_y) ** 2) / (2 * radius**2)))
        heatmap = fallback / np.max(fallback)
        method = "Center fallback"

    return heatmap, method


def create_heatmap_overlay(image, heatmap, alpha=0.42):
    rgb_image = image.convert("RGB")
    width, height = rgb_image.size
    base = np.array(rgb_image)
    heatmap = cv2.resize(np.uint8(255 * heatmap), (width, height))
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = np.uint8((1 - alpha) * base + alpha * heatmap)
    return Image.fromarray(overlay)


def build_medical_summary(result):
    if result["mode"] == "binary":
        if result["prediction"] > 0.7:
            return "Possible diabetic retinopathy signal. Clinical follow-up is recommended.", "bad"
        if result["prediction"] < 0.3:
            return "Low DR signal on this screening image.", "good"
        return "Borderline / uncertain screening result. Image quality review is advised.", "warn"

    return f"Top predicted class: {result['label']}.", "warn"


def generate_pdf_report(result, uploaded_file, image, model_path, heatmap_overlay, reason=None):
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.utils import ImageReader
        from reportlab.lib.units import mm
        from reportlab.pdfgen import canvas
    except Exception as exc:
        return None, f"PDF support is unavailable: {exc}"

    buffer = BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    page_width, page_height = A4
    left = 16 * mm
    right = page_width - 16 * mm
    top = page_height - 16 * mm

    def draw_wrapped_text(text, x, y, max_width, line_height=12, font_name="Helvetica", font_size=10, color_value=colors.HexColor("#23364a")):
        pdf.setFont(font_name, font_size)
        pdf.setFillColor(color_value)
        words = text.split()
        lines = []
        current = ""
        for word in words:
            candidate = word if not current else f"{current} {word}"
            if pdf.stringWidth(candidate, font_name, font_size) <= max_width:
                current = candidate
            else:
                if current:
                    lines.append(current)
                current = word
        if current:
            lines.append(current)
        for line in lines:
            pdf.drawString(x, y, line)
            y -= line_height
        return y

    def draw_image_box(pil_image, x, y, box_width, box_height, label):
        pdf.setStrokeColor(colors.HexColor("#c8d5e6"))
        pdf.setFillColor(colors.white)
        pdf.roundRect(x, y, box_width, box_height, 8, stroke=1, fill=1)
        pdf.setFillColor(colors.HexColor("#5c7084"))
        pdf.setFont("Helvetica-Bold", 9)
        pdf.drawString(x + 6, y + box_height - 12, label)

        if pil_image is None:
            pdf.setFont("Helvetica-Oblique", 9)
            pdf.drawString(x + 6, y + box_height / 2, "Heatmap not available")
            return

        preview = pil_image.copy()
        preview.thumbnail((int(box_width - 12), int(box_height - 22)))
        img_reader = ImageReader(preview)
        img_width, img_height = preview.size
        draw_x = x + (box_width - img_width) / 2
        draw_y = y + (box_height - img_height) / 2 - 4
        pdf.drawImage(img_reader, draw_x, draw_y, width=img_width, height=img_height, preserveAspectRatio=True, mask="auto")

    pdf.setTitle("Diabetic Retinopathy Screening Report")
    pdf.setAuthor("Streamlit Retina App")

    pdf.setFillColor(colors.HexColor("#0f2742"))
    pdf.setFont("Helvetica-Bold", 20)
    pdf.drawString(left, top - 4, "Diabetic Retinopathy Screening Report")
    pdf.setFont("Helvetica", 10)
    pdf.setFillColor(colors.HexColor("#52667b"))
    pdf.drawString(left, top - 18, "Generated by the Streamlit screening app")

    pdf.setFillColor(colors.HexColor("#eaf2ff"))
    pdf.roundRect(right - 44 * mm, top - 18, 42 * mm, 10 * mm, 5, stroke=0, fill=1)
    pdf.setFillColor(colors.HexColor("#12335d"))
    pdf.setFont("Helvetica-Bold", 10)
    pdf.drawCentredString(right - 23 * mm, top - 12, "SCREENING ONLY")

    pdf.setStrokeColor(colors.HexColor("#d7e1ee"))
    pdf.line(left, top - 26, right, top - 26)

    y = top - 40
    pdf.setFillColor(colors.HexColor("#0f2742"))
    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(left, y, "Study Summary")
    y -= 14
    pdf.setFont("Helvetica", 10)

    summary_rows = [
        f"File name: {uploaded_file.name}",
        f"File size: {uploaded_file.size / 1024:.1f} KB",
        f"Image mode: {image.mode}",
        f"Model file: {Path(model_path).name if model_path else 'unknown'}",
        f"Result label: {result['label']}",
        f"Confidence: {result['confidence']:.2%}",
        f"Interpretation: {reason or 'Accepted as fundus-like and analyzed.'}",
    ]
    for row in summary_rows:
        y = draw_wrapped_text(row, left, y, right - left, line_height=13)
        y -= 2

    y -= 4
    pdf.setStrokeColor(colors.HexColor("#d7e1ee"))
    pdf.line(left, y, right, y)
    y -= 16

    pdf.setFont("Helvetica-Bold", 12)
    pdf.setFillColor(colors.HexColor("#0f2742"))
    pdf.drawString(left, y, "Image Review")
    y -= 8

    image_box_height = 80 * mm
    image_box_width = (right - left - 8 * mm) / 2
    image_y = y - image_box_height
    draw_image_box(image, left, image_y, image_box_width, image_box_height, "Uploaded fundus image")
    draw_image_box(heatmap_overlay, left + image_box_width + 8 * mm, image_y, image_box_width, image_box_height, "Heatmap overlay")

    y = image_y - 14
    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(left, y, "Interpretation")
    y -= 13
    pdf.setFont("Helvetica", 10)
    interpretation = (
        "This report is a screening aid. The heatmap shows which regions influenced the model most strongly. "
        "It should be reviewed together with the original image and clinical context."
    )
    y = draw_wrapped_text(interpretation, left, y, right - left, line_height=13)

    if result.get("mode") == "multiclass":
        y -= 8
        pdf.setFont("Helvetica-Bold", 12)
        pdf.drawString(left, y, "Probability Distribution")
        y -= 13
        pdf.setFont("Helvetica", 10)
        for name, probability in zip(CLASS_NAMES, result.get("probabilities", [])):
            y = draw_wrapped_text(f"{name}: {float(probability):.2%}", left, y, right - left, line_height=13)

    pdf.setFont("Helvetica-Oblique", 9)
    pdf.setFillColor(colors.HexColor("#6b7c8f"))
    pdf.drawString(left, 14 * mm, "Generated by the Streamlit retina screening app. Not a medical diagnosis.")

    pdf.showPage()
    pdf.save()
    buffer.seek(0)
    return buffer.getvalue(), None


def build_report(result, uploaded_file, image, model_path):
    lines = [
        "Diabetic Retinopathy Screening Report",
        "",
        f"File name: {uploaded_file.name}",
        f"File size: {uploaded_file.size / 1024:.1f} KB",
        f"Image mode: {image.mode}",
        f"Model file: {Path(model_path).name if model_path else 'unknown'}",
        f"Prediction mode: {result['mode']}",
        f"Result label: {result['label']}",
        f"Confidence: {result['confidence']:.2%}",
    ]

    if result.get("mode") == "multiclass":
        lines.append("")
        lines.append("Probability distribution:")
        for name, probability in zip(CLASS_NAMES, result.get("probabilities", [])):
            lines.append(f"- {name}: {float(probability):.2%}")

    lines.append("")
    lines.append("Note: This is a screening aid, not a medical diagnosis.")
    return "\n".join(lines)


def render_sidebar(model_status_message):
    st.sidebar.header("App Status")
    theme_mode = st.sidebar.radio(
        "Theme",
        ["Light Mode", "Black / Premium"],
        index=0,
        help="Switch between a bright medical look and a premium black/white display.",
    )

    if RESOLVED_MODEL_PATH:
        st.sidebar.success(f"Model loaded: {Path(RESOLVED_MODEL_PATH).name}")
    else:
        st.sidebar.error("No model file found")

    if model_status_message:
        st.sidebar.caption(model_status_message)

    st.sidebar.divider()
    st.sidebar.subheader("What works best")
    st.sidebar.write("- JPG or PNG retinal fundus photos")
    st.sidebar.write("- Good lighting and centered eye images")
    st.sidebar.write("- One eye per image")

    st.sidebar.divider()
    st.sidebar.subheader("Output")
    st.sidebar.write("- Binary model: No DR vs DR detected")
    st.sidebar.write("- Multi-class model: severity probability table")

    st.sidebar.divider()
    st.sidebar.caption("Tip: wrong-image checks are intentionally strict to reduce false predictions.")

    return theme_mode


model, model_status_message = load_model()
theme_mode = render_sidebar(model_status_message)
inject_styles(theme_mode)

st.markdown(
    """
    <div class="hero">
        <h1>🩺 Diabetic Retinopathy Detection</h1>
        <p>Upload a retinal fundus image and the model will estimate DR severity with safer validation and clearer feedback.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

top_left, top_right = st.columns([2, 1])
with top_left:
    st.markdown("<div class='metric-card'><b>Fast review:</b> the app now rejects obvious non-retina images before prediction.</div>", unsafe_allow_html=True)
with top_right:
    st.markdown(
        f"<div class='metric-card'><b>Model:</b> {Path(RESOLVED_MODEL_PATH).name if RESOLVED_MODEL_PATH else 'not loaded'}</div>",
        unsafe_allow_html=True,
    )

uploaded_file = st.file_uploader("📤 Upload retinal image", type=["jpg", "jpeg", "png"], help="Use a fundus photo, not a normal photograph or screenshot.")

if uploaded_file is None:
    st.info("Upload a retinal image to start. The app will validate image type, then run prediction if the image looks like a fundus photo.")
    st.stop()

if model is None:
    st.error(
        "No model is available to run predictions. Check the sidebar for the loaded model status and restart the app after fixing the model path or TensorFlow environment."
    )
    st.stop()

try:
    image = Image.open(uploaded_file)
except UnidentifiedImageError:
    st.error("The uploaded file is not a readable image. Please upload a JPG or PNG retinal fundus image.")
    st.stop()
except Exception as e:
    st.error(f"Could not open the uploaded file: {e}")
    st.stop()

preview_col, info_col = st.columns([1.4, 1])
with preview_col:
    st.image(image, caption="🩺 Uploaded Image", use_container_width=True)

with info_col:
    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
    st.write("### Upload Summary")
    st.write(f"File name: `{uploaded_file.name}`")
    st.write(f"File size: {uploaded_file.size / 1024:.1f} KB")
    st.write(f"Image mode: `{image.mode}`")
    st.markdown("</div>", unsafe_allow_html=True)

analyze = st.button("🔍 Analyze", type="primary", use_container_width=True)

if not analyze:
    st.caption("Click Analyze after selecting an image.")
    st.stop()

with st.spinner("Analyzing the image..."):
    is_retina_like, reason = assess_retina_image(image)
    if not is_retina_like:
        st.error("⚠️ Wrong image type. Please upload a retinal fundus photo.")
        st.caption(reason)
        st.stop()

    try:
        result = predict_image(model, image)
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

st.markdown("### 🧾 Result")

result_tag_text, result_tag_class = build_medical_summary(result)
st.markdown(
    f"""
    <div class="report-shell">
        <div class="report-top">
            <div>
                <div class="report-kicker">Medical Screening Report</div>
                <h2 class="report-title">Automated DR Screening Summary</h2>
                <p class="report-subtitle">{result_tag_text}</p>
            </div>
            <div class="badge {result_tag_class}">{result['label']}</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if result["mode"] == "binary":
    confidence = result["confidence"]
    if result["prediction"] > 0.7:
        st.error(f"Screening result: **{result['label']}**")
        st.write(f"Confidence: **{confidence:.2%}**")
        st.warning("This suggests possible diabetic retinopathy. Please use clinical judgment.")
    elif result["prediction"] < 0.3:
        st.success(f"Screening result: **{result['label']}**")
        st.write(f"Confidence: **{confidence:.2%}**")
    else:
        st.warning("🤔 The model is uncertain on this image.")
        st.write(f"Confidence: **{confidence:.2%}**")
else:
    confidence = result["confidence"]
    top_idx = result["top_idx"]
    st.subheader(f"Severity Level: {result['label']}")
    st.write(f"Confidence: **{confidence:.2%}**")
    st.info("Higher-class predictions are not a diagnosis. Use this as a screening aid only.")

clinical_summary = (
    "The image was accepted as fundus-like and the model produced a prediction. "
    "If this result is being used for a real patient, treat it as a screening signal and confirm clinically."
)

if result["mode"] == "binary":
    if result["prediction"] > 0.7:
        clinical_summary = (
            "The model leans toward diabetic retinopathy. This does not confirm disease, but it does justify follow-up evaluation."
        )
    elif result["prediction"] < 0.3:
        clinical_summary = (
            "The model leans toward no diabetic retinopathy. Low-risk screening results still need clinical context."
        )
    else:
        clinical_summary = (
            "The model is uncertain on this image. Recheck image quality or consider another fundus photo before trusting the result."
        )
else:
    clinical_summary = (
        f"The model's strongest class is {result['label']}. Use the probability table below to compare nearby severity levels."
    )

st.info(clinical_summary)

heatmap = None
heatmap_overlay = None
heatmap_method = None
heatmap_error = None
try:
    heatmap, heatmap_method = generate_attention_heatmap(
        model,
        image,
        result.get("top_idx") if result.get("mode") == "multiclass" else None,
    )
    heatmap_overlay = create_heatmap_overlay(image, heatmap)
except Exception as exc:
    heatmap_error = f"Heatmap generation failed: {exc}"

st.markdown(
    """
    <div class="output-shell">
        <div class="output-header">
            <div>
                <div class="output-label">Output Section</div>
                <h3 class="output-title">Prediction and heatmap review</h3>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if heatmap_overlay is not None:
    heat_left, heat_right = st.columns([1, 1])
    with heat_left:
        st.markdown("<div class='heatmap-frame'>", unsafe_allow_html=True)
        st.image(image, caption="Original retinal image", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with heat_right:
        st.markdown("<div class='heatmap-frame'>", unsafe_allow_html=True)
        st.image(heatmap_overlay, caption="Grad-CAM attention heatmap", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="heatmap-legend">
            <span class="legend-pill">Red: strongest influence</span>
            <span class="legend-pill">Yellow: moderate influence</span>
            <span class="legend-pill">Blue: lower influence</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(f"Attention method used: {heatmap_method}")
else:
    st.warning(heatmap_error or "No heatmap could be produced for this image.")

left_col, right_col = st.columns([1, 1])
with left_col:
    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
    st.write("### Probability Distribution")
    if result.get("mode") == "binary":
        st.write(f"No DR probability: {1 - result['prediction']:.1%}")
        st.write(f"DR probability: {result['prediction']:.1%}")
    else:
        for name, p in zip(CLASS_NAMES, result.get("probabilities", [])):
            st.write(f"{name}: {p:.1%}")
    st.markdown("</div>", unsafe_allow_html=True)

with right_col:
    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
    st.write("### Technical Details")
    st.write(f"Model file: `{Path(RESOLVED_MODEL_PATH).name if RESOLVED_MODEL_PATH else 'unknown'}`")
    st.write(f"Model input: `(1, 128, 128, 3)`")
    st.write(f"Output class index: `{top_idx if result['mode'] == 'multiclass' else 'binary'}`")
    with st.expander("Show raw model output"):
        st.write(result["raw"])
    st.markdown("</div>", unsafe_allow_html=True)

report_text = build_report(result, uploaded_file, image, RESOLVED_MODEL_PATH)
pdf_bytes, pdf_error = generate_pdf_report(result, uploaded_file, image, RESOLVED_MODEL_PATH, heatmap_overlay, reason=reason)

pdf_col, text_col = st.columns([1, 1])
with pdf_col:
    if pdf_bytes is not None:
        st.download_button(
            "⬇️ Download PDF medical report",
            data=pdf_bytes,
            file_name=f"retina_medical_report_{Path(uploaded_file.name).stem}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
    else:
        st.warning(pdf_error or "PDF report is unavailable.")

with text_col:
    st.download_button(
        "⬇️ Download text report",
        data=report_text,
        file_name=f"retina_screening_report_{Path(uploaded_file.name).stem}.txt",
        mime="text/plain",
        use_container_width=True,
    )

with st.expander("Preview report"):
    st.code(report_text)

st.markdown("---")
st.markdown("📘 *Developed using TensorFlow and Streamlit*")
