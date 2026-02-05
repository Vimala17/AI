import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle
import os
import json
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input

# --- 1. SETTING UP PATHS ---
save_path = r"C:\Users\Acer\OneDrive\PROJECTS\AgriSense AI\backend\models"
if not os.path.exists(save_path):
    os.makedirs(save_path)

# --- 2. CROP RECOMMENDATION MODEL TRAINING ---
print("\n--- Starting Crop Model Training ---")
csv_path = os.path.join(save_path, 'Crop_recommendation.csv')

if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    crop_model = RandomForestClassifier(n_estimators=100)
    crop_model.fit(X_train, y_train)

    with open(os.path.join(save_path, 'crop_model.pkl'), 'wb') as f:
        pickle.dump(crop_model, f)
    print("✅ Success: crop_model.pkl created!")
else:
    print(f"❌ Error: {csv_path} not found!")

# --- 3. PLANT DISEASE MODEL (Transfer Learning - MobileNetV2) ---
print("\n--- Starting Plant Disease Model Training with Transfer Learning ---")

DATASET_BASE = r"C:\Users\Acer\Downloads\archive (7)\New Plant Diseases Dataset(Augmented)\New Plant Diseases Dataset(Augmented)"
train_dir = os.path.join(DATASET_BASE, "train")



if os.path.exists(train_dir):
    # Data Preprocessing
    datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input, # MobileNet scaling handles ikkade avthundhi
        rotation_range=20,
        horizontal_flip=True,
        validation_split=0.2 
    )

    train_generator = datagen.flow_from_directory(
        train_dir, 
        target_size=(224, 224), 
        batch_size=32, 
        class_mode='categorical', 
        subset='training'
    )

    # Transfer Learning Logic
    # Munde train ayina MobileNetV2 ni base ga vaduthunnam
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False  # Base layers freeze cheshanu

    # Adding Custom Head
    x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    predictions = tf.keras.layers.Dense(len(train_generator.class_indices), activation='softmax')(x)

    disease_model = tf.keras.Model(inputs=base_model.input, outputs=predictions)
    disease_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Training - 5 epochs thappa kunda cheyandi
    print("Training the disease model... please wait.")
    disease_model.fit(train_generator, epochs=10                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 ) 
    
    # Save Model
    disease_model.save(os.path.join(save_path, 'plant_disease_model.h5'))
    
    # Save Labels Mapping (Idi app.py prediction ki chala avasaram)
    labels = {v: k for k, v in train_generator.class_indices.items()}
    with open(os.path.join(save_path, 'class_indices.json'), 'w') as f:
        json.dump(labels, f)
        
    print("✅ Success: plant_disease_model.h5 and class_indices.json created!")
else:
    print("❌ Error: Disease Dataset path not found.")

print("\n--- All Training Processes Completed Successfully ---")