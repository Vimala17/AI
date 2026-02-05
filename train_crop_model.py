import pandas as pd
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

data_path = r"C:\Users\Acer\OneDrive\PROJECTS\AgriSense AI\datasets\crop_recommendation.csv"

if not os.path.exists(data_path):
    print(f"❌ Error: డేటాసెట్ ఫైల్ ఈ పాత్ లో లేదు: {data_path}")
elif os.path.getsize(data_path) == 0:
    print(f"❌ Error: ఫైల్ దొరికింది కానీ అది ఖాళీగా (Empty) ఉంది!")
else:
    # 2. Load Dataset
    df = pd.read_csv(data_path)
    print("✅ Dataset loaded successfully!")

  
    X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
    y = df['label']

    # 4. Train Model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # 5. Save the Model
    save_path = r"C:\Users\Acer\OneDrive\PROJECTS\agriSense AI\backend\models\crop_model.pkl"
    with open(save_path, 'wb') as f:
        pickle.dump(model, f)

    print(f"✅ Success! Model saved at: {save_path}")