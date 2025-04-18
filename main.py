import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Path to your dataset folder
data_dir = "train"
img_size = (224, 224)
batch_size = 16

# Preprocess and split data
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2  # 80% for training, 20% for validation
)

# Training data
train_gen = datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

# Validation data
val_gen = datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# CNN Model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(3, activation='softmax')  # 3 categories: safe, moderate, dangerous
])

# Compile model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train model
model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=10
)

# Save model
model.save("water_pollution_model.h5")
