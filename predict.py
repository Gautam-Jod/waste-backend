import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# ✅ Load model
model = tf.keras.models.load_model("waste_model.keras")

classes = ['cardboard','glass','metal','paper','plastic','trash']

# ✅ Load image
img = image.load_img("test2.jpg", target_size=(224,224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) / 255

# ✅ Predict
prediction = model.predict(img_array)

# 🔥 Print full prediction array
print("\nRaw prediction values:")
for i in range(len(classes)):
    print(f"{classes[i]}: {round(float(prediction[0][i])*100,2)}%")

# ✅ Final prediction
predicted_class = classes[np.argmax(prediction)]
confidence = np.max(prediction)

print("\nFinal Prediction:", predicted_class)
print("Confidence:", round(float(confidence)*100,2), "%")
