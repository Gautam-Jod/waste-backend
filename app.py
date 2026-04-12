from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image

app = Flask(__name__)
CORS(app)

# ✅ Load correct model
model = tf.keras.models.load_model("waste_model.keras")

# ✅ Class labels (MUST match training order)
classes = ["cardboard","glass","metal","paper","plastic","trash"]

# ✅ Prediction function
def predict_image(img):
    img = img.resize((224,224))
    img = np.array(img) / 255.0
    img = img.reshape(1,224,224,3)

    prediction = model.predict(img)

    index = np.argmax(prediction)
    confidence = round(float(np.max(prediction)) * 100, 2)

    return classes[index], confidence


# ✅ API Route
@app.route("/predict", methods=["POST"])
def predict():
    print("🔥 Request received")

    if "image" not in request.files:
        print("❌ No image uploaded")
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]

    try:
        # ✅ FIX: Convert to RGB
        image = Image.open(file).convert("RGB")

        prediction, confidence = predict_image(image)

        print(f"✅ Prediction: {prediction} ({confidence}%)")

        return jsonify({
            "prediction": prediction,
            "confidence": confidence
        })

    except Exception as e:
        print("❌ ERROR:", str(e))
        return jsonify({"error": "Prediction failed"}), 500


# ✅ Test route (VERY IMPORTANT)
@app.route("/", methods=["GET"])
def home():
    return "✅ Flask Backend is Running"


if __name__ == "__main__":
    app.run(debug=True)
