# Hand Gesture Recognition

This project builds a gesture recognition pipeline using MediaPipe, OpenCV and TensorFlow. It handles six gestures: `fist`, `hand`, `thumb`, `yo`, `peace`, and `index`.

## Setup

1. Create a virtual environment and install dependencies:

```bash
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

2. Place your image folders under `data/` with names matching the gestures (e.g. `data/fist`, etc.).

## Steps

1. **Preprocess** – extract hands, save crops and landmarks to CSV:
   ```bash
   python data_preprocess.py
   ```

2. **Train** – train a simple neural network using the CSV:
   ```bash
   python train_model.py
   ```

3. **Run** – open webcam and perform live recognition:
   ```bash
   python recognition.py
   ```
   Press `q` to quit.

## Labels & Directions

The gestures are mapped to directions as follows:

- `index` → right
- `peace` → left
- `hand` → up
- `thumb` → down
- `yo` → front
- `fist` → back

The direction is displayed alongside the predicted gesture in the live demo.

## Notes

- The preprocessing script only processes images that contain a detectable hand; any failures are logged to the console.
- Landmarks are normalized by the original image dimensions, so the network learns relative positions.
