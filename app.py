from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import os

app = Flask(__name__)

# ✅ FIX CORS (VERY IMPORTANT)
CORS(app, resources={r"/*": {"origins": "*"}})

# ✅ Load model lazily
model = None

def load_model():
    global model
    if model is None:
        print("📦 Loading model...")
        MODEL_PATH = "waste_model.keras"
        model = tf.keras.models.load_model(MODEL_PATH)
        print("✅ Model loaded!")

classes = ["cardboard","glass","metal","paper","plastic","trash"]

def predict_image(img):
    img = img.resize((224,224))
    img = np.array(img) / 255.0
    img = img.reshape(1,224,224,3)

    prediction = model.predict(img)

    index = np.argmax(prediction)
    confidence = round(float(np.max(prediction)) * 100, 2)

    return classes[index], confidence


@app.route("/predict", methods=["POST"])
def predict():
    print("🔥 Request received")

    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    try:
        load_model()

        file = request.files["image"]
        image = Image.open(file).convert("RGB")

        prediction, confidence = predict_image(image)

        return jsonify({
            "prediction": prediction,
            "confidence": confidence
        })

    except Exception as e:
        print("❌ ERROR:", str(e))
        return jsonify({"error": str(e)}), 500


@app.route("/", methods=["GET"])
def home():
    return "✅ Flask Backend is Running"


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
