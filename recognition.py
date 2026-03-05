import os
import cv2
import numpy as np
import mediapipe as mp
from tensorflow import keras

DIRECTION_MAP = {
    "index": "right",
    "peace": "left",
    "hand": "up",
    "thumb": "down",
    "yo": "front",
    "fist": "back",
}

MODEL_PATH = os.path.join(os.getcwd(), "gesture_model.keras")
LABELS_PATH = os.path.join(os.getcwd(), "label_encoder.npy")


def extract_landmarks(hand_landmarks):
    coords = []
    for lm in hand_landmarks.landmark:
        coords.append(lm.x)
        coords.append(lm.y)
    return np.array(coords, dtype="float32")


def main():

    if not os.path.exists(MODEL_PATH) or not os.path.exists(LABELS_PATH):
        print("Trained model or labels missing. Run train_model.py first.")
        return

    model = keras.models.load_model(MODEL_PATH)
    classes = np.load(LABELS_PATH, allow_pickle=True)
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Could not open webcam")
        return

    while True:

        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        label_text = ""

        if results.multi_hand_landmarks:

            for hand_landmarks in results.multi_hand_landmarks:

                mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

                feat = extract_landmarks(hand_landmarks).reshape(1, -1)

                pred = model.predict(feat, verbose=0)

                idx = np.argmax(pred)
                gesture = classes[idx]

                direction = DIRECTION_MAP.get(gesture, "")

                confidence = np.max(pred)

                label_text = f"{gesture} ({direction}) {confidence:.2f}"

                break

        if label_text:
            cv2.putText(
                frame,
                label_text,
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

        cv2.imshow("Gesture Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
