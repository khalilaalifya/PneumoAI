from flask import Flask, render_template, request, send_file
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import cv2
from matplotlib import colormaps
from werkzeug.utils import secure_filename
from flask import url_for

from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image as RLImage
)

from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter

from datetime import datetime

app = Flask(__name__)

# =========================
# LOAD MODEL
# =========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
print("="*50)
print("BASE_DIR :", BASE_DIR)
print("STATIC   :", os.path.join(BASE_DIR, "static"))
print("="*50)

MODEL_PATH = os.path.join(BASE_DIR, "pneumonia_model.h5")

if not os.path.exists(MODEL_PATH):
    print("=" * 60)
    print("ERROR : pneumonia_model.h5 TIDAK DITEMUKAN")
    print("Lokasi yang dicari:")
    print(MODEL_PATH)
    print("=" * 60)
    sys.exit()

model = load_model(MODEL_PATH)

UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
HEATMAP_FOLDER = os.path.join(BASE_DIR, "static", "heatmaps")
PDF_FOLDER = os.path.join(BASE_DIR, "static", "reports")
app.config["UPLOAD_FOLDER"]=UPLOAD_FOLDER
app.config["HEATMAP_FOLDER"]=HEATMAP_FOLDER
app.config["PDF_FOLDER"]=PDF_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(app.config["HEATMAP_FOLDER"], exist_ok=True)
os.makedirs(app.config["PDF_FOLDER"], exist_ok=True)

# =========================
# PREPROCESS IMAGE
# =========================

def preprocess(img_path):

    img = image.load_img(img_path, target_size=(224, 224))

    img_array = image.img_to_array(img)

    img_array = np.expand_dims(img_array, axis=0)

    img_array = img_array / 255.0

    return img_array

# =========================
# MAKE GRADCAM
# =========================

def make_gradcam_heatmap(img_array, model):

    # cari layer conv terakhir
    last_conv_layer = None

    for layer in reversed(model.layers):

        try:
            if len(layer.output.shape) == 4:
                last_conv_layer = layer.name
                break
        except:
            continue

    grad_model = tf.keras.models.Model(
        model.inputs,
        [
            model.get_layer(last_conv_layer).output,
            model.output
        ]
    )

    with tf.GradientTape() as tape:

        conv_outputs, predictions = grad_model(img_array)

        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]

    heatmap = tf.squeeze(heatmap)

    heatmap = np.maximum(heatmap, 0)

    if np.max(heatmap) != 0:
        heatmap /= np.max(heatmap)

    return heatmap

# =========================
# SAVE HEATMAP
# =========================

def save_and_display_gradcam(img_path, heatmap, output_path):

    img = cv2.imread(img_path)

    img = cv2.resize(img, (224, 224))

    heatmap = np.uint8(255 * heatmap)

    jet = colormaps["jet"]

    jet_colors = jet(np.arange(256))[:, :3]

    jet_heatmap = jet_colors[heatmap]

    jet_heatmap = tf.keras.preprocessing.image.array_to_img(
        jet_heatmap
    )

    jet_heatmap = jet_heatmap.resize((224, 224))

    jet_heatmap = tf.keras.preprocessing.image.img_to_array(
        jet_heatmap
    )

    superimposed_img = jet_heatmap * 0.4 + img

    cv2.imwrite(
        output_path,
        np.uint8(superimposed_img)
    )

# =========================
# GENERATE PDF REPORT
# =========================

def generate_pdf_report(
    pdf_path,
    prediction,
    confidence,
    severity,
    analysis_time,
    xray_path,
    heatmap_path
):
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter
    )

    styles = getSampleStyleSheet()

    elements = []

    title = Paragraph(
        "<b>PneumoAI Analysis Report</b>",
        styles['Title']
    )

    elements.append(title)

    elements.append(Spacer(1, 20))

    info = f"""
    <b>Prediction:</b> {prediction}<br/>
    <b>Confidence:</b> {confidence}%<br/>
    <b>Severity Level:</b> {severity}<br/>
    <b>Analysis Time:</b> {analysis_time}<br/>
    """

    elements.append(
        Paragraph(info, styles['BodyText'])
    )

    elements.append(Spacer(1, 20))

    elements.append(
        Paragraph("<b>Original X-Ray</b>", styles['Heading2'])
    )

    elements.append(
        RLImage(xray_path, width=250, height=250)
    )

    elements.append(Spacer(1, 20))

    elements.append(
        Paragraph("<b>GradCAM Heatmap</b>", styles['Heading2'])
    )

    elements.append(
        RLImage(heatmap_path, width=250, height=250)
    )

    print("Generating PDF:", pdf_path)

    doc.build(elements)

    print("PDF berhasil dibuat")

# =========================
# HOME
# =========================

@app.route("/", methods=["GET", "POST"])

def index():

    prediction = None
    confidence = None
    img_path = None
    gradcam_path = None
    severity = None
    pdf_download = None
    analysis_time = None

    if request.method == "POST":

        file = request.files["file"]

        if file:

            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)

            file.save(filepath)

            print("Image saved :", filepath)
            print("Exists ?", os.path.exists(filepath))

            img_array = preprocess(filepath)

            pred = model.predict(img_array)

            print("Prediction Value:", pred)

            pred_value = float(pred.squeeze())

            analysis_time = datetime.now().strftime(
               "%d-%m-%Y %H:%M:%S"
     )

            # =====================
            # PREDICTION
            # =====================

            if pred_value > 0.5:

                prediction = "Pneumonia Detected"

                confidence = round(pred_value * 100, 2)

            else:

                prediction = "NORMAL"

                confidence = round((1 - pred_value) * 100, 2)

            severity = "none"
            
            if pred_value > 0.85:
                severity = "High"
            
            elif pred_value > 0.65:
                severity = "Medium"

            else:
                severity = "Low"

            if pred_value <= 0.5:
                
                prediction = "NORMAL"
                
                severity = "None"

            # =====================
            # GRADCAM
            # =====================

            heatmap = make_gradcam_heatmap(
                img_array,
                model
            )

            heatmap_filename = (
                "heatmap_" + filename
            )

            heatmap_path = os.path.join(
                app.config["HEATMAP_FOLDER"],
                heatmap_filename
            )

            save_and_display_gradcam(
            filepath,
            heatmap,
            heatmap_path
            )

            print("Heatmap :", heatmap_path)
            print("Heatmap Exists ?", os.path.exists(heatmap_path))

            # =====================
            # GENERATE PDF
            # =====================

            pdf_filename = (
                "report_" +
                filename.rsplit(".",1)[0] +
                ".pdf"
            )

            pdf_path = os.path.join(
                app.config["PDF_FOLDER"],
                pdf_filename
            )

            generate_pdf_report(
                pdf_path,
                prediction,
                confidence,
                severity,
                analysis_time,
                filepath,
                heatmap_path
            )

            pdf_download = "/" + pdf_path

            # cache breaker
            random_number = np.random.randint(1, 999999)

            img_path = url_for(
                "static",
                filename=f"uploads/{filename}"
            ) + f"?v={random_number}"

            gradcam_path = url_for(
                "static",
                filename=f"heatmaps/heatmap_{filename}"
            ) + f"?v={random_number}"

            pdf_download = url_for(
                "static",
                filename=f"reports/report_{filename.rsplit('.',1)[0]}.pdf"
            )
            
    print("="*60)
    print("IMG PATH      :", img_path)
    print("HEATMAP PATH  :", gradcam_path)
    print("PDF PATH      :", pdf_download)
    print("="*60)

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        severity=severity,
        analysis_time=analysis_time,
        img_path=img_path,
        gradcam_path=gradcam_path,
        pdf_download=pdf_download
)

# =========================
# RUN
# =========================

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5001)
