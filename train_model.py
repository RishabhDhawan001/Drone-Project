import os
import pandas as pd
import numpy as np
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder

BASE_DATA_DIR = os.path.join(os.getcwd(), r"C:\Users\DeLL\Desktop\vscode\drone_mp\data")
CSV_PATH = os.path.join(BASE_DATA_DIR, "landmarks.csv")
MODEL_PATH = os.path.join(os.getcwd(), "gesture_model.keras")

def load_dataset():
    df = pd.read_csv(CSV_PATH)
    df = df.dropna()
    gestures = df["gesture"].values
    le = LabelEncoder()
    y = le.fit_transform(gestures)
    X = df.iloc[:, 2:].values.astype("float32")
    return X, y, le

def build_model(input_dim, num_classes):
    model = keras.Sequential([
        keras.layers.Input(shape=(input_dim,)),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

def train():
    if not os.path.exists(CSV_PATH):
        print("dataset CSV not found. Run data_preprocess.py first.")
        return
    X, y, le = load_dataset()
    num_classes = len(np.unique(y))
    model = build_model(X.shape[1], num_classes)
    model.summary()
    model.fit(X, y, epochs=30, validation_split=0.2, batch_size=32)
    model.save(MODEL_PATH)
    le_path = os.path.join(os.getcwd(), "label_encoder.npy")
    np.save(le_path, le.classes_)
    print(f"model saved to {MODEL_PATH}")
    print(f"label classes saved to {le_path}")

if __name__ == "__main__":
    train()
