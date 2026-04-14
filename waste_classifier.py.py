import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout

# =========================
# DATA GENERATOR
# =========================
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=25,
    zoom_range=0.25,
    horizontal_flip=True,
    shear_range=0.1
)

train_data = datagen.flow_from_directory(
    'dataset/',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

val_data = datagen.flow_from_directory(
    'dataset/',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

print("\n📊 Class Indices:", train_data.class_indices)

# =========================
# LOAD PRETRAINED MODEL
# =========================
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(224,224,3)
)

# 🔥 Freeze base model initially
base_model.trainable = False

# =========================
# CUSTOM HEAD
# =========================
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.7)(x)
predictions = Dense(6, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# =========================
# INITIAL TRAINING
# =========================
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

print("\n🚀 Starting Initial Training...\n")

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=10
)

# =========================
# FINE TUNING
# =========================
print("\n🔧 Starting Fine-Tuning...\n")

# Unfreeze top layers only
base_model.trainable = True

for layer in base_model.layers[:-100]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history_fine = model.fit(
    train_data,
    validation_data=val_data,
    epochs=5
)

# =========================
# SAVE MODEL
# =========================
model.save("waste_model.keras")

print("\n✅ Final Model Saved Successfully!")
