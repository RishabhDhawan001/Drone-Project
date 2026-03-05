import os
import csv
import cv2
import mediapipe as mp
import numpy as np
GESTURES = ["fist", "hand", "thumb", "yo", "peace", "index"]
DIRECTION_MAP = {
    "index": "right",
    "peace": "left",
    "hand": "up",
    "thumb": "down",
    "yo": "front",
    "fist": "back",
}

BASE_DATA_DIR = os.path.join(os.getcwd(), "data")
CROPPED_DIR = os.path.join(BASE_DATA_DIR, "cropped")
CSV_PATH = os.path.join(BASE_DATA_DIR, "landmarks.csv")

def ensure_dirs():
    os.makedirs(CROPPED_DIR, exist_ok=True)
    for g in GESTURES:
        os.makedirs(os.path.join(CROPPED_DIR, g), exist_ok=True)

def extract_landmarks(hand_landmarks):
    coords = []
    for lm in hand_landmarks.landmark:
        coords.append(lm.x)
        coords.append(lm.y)
    return coords

def process():
    ensure_dirs()
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)
    header = ["gesture", "direction"] + [f"x{i}" for i in range(21)] + [f"y{i}" for i in range(21)]
    with open(CSV_PATH, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)

        for gesture in GESTURES:
            src_dir = os.path.join(BASE_DATA_DIR, gesture)
            if not os.path.isdir(src_dir):
                print(f"warning: {src_dir} does not exist, skipping")
                continue

            for fname in os.listdir(src_dir):
                if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue
                path = os.path.join(src_dir, fname)
                img = cv2.imread(path)
                if img is None:
                    continue
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb)
                if not results.multi_hand_landmarks:
                    print(f"no hand found in {path}")
                    continue
                hand_landmarks = results.multi_hand_landmarks[0]

                h, w, _ = img.shape
                xs = [lm.x for lm in hand_landmarks.landmark]
                ys = [lm.y for lm in hand_landmarks.landmark]
                x_min, x_max = min(xs), max(xs)
                y_min, y_max = min(ys), max(ys)
                pad = 0.1
                x_min = max(0, x_min - pad)
                y_min = max(0, y_min - pad)
                x_max = min(1, x_max + pad)
                y_max = min(1, y_max + pad)
                x1, y1 = int(x_min * w), int(y_min * h)
                x2, y2 = int(x_max * w), int(y_max * h)

                crop = img[y1:y2, x1:x2]
                outpath = os.path.join(CROPPED_DIR, gesture, fname)
                cv2.imwrite(outpath, crop)

                flattened = extract_landmarks(hand_landmarks)
                row = [gesture, DIRECTION_MAP.get(gesture, "")] + flattened
                writer.writerow(row)

    hands.close()
    print(f"processing complete, landmarks written to {CSV_PATH}")

if __name__ == "__main__":
    process()
