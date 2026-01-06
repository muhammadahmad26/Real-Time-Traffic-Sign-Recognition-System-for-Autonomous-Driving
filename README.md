# 🚦 Traffic Sign Detection using YOLOv8

This project utilizes **YOLOv8 (Ultralytics)** to detect and classify traffic signs. It is trained on the **GTSRB (German Traffic Sign Recognition Benchmark)** dataset. The repository includes scripts for training the model, evaluating performance, and running real-time inference on images, webcam feeds, and YouTube videos.

## ✨ Features

*   **Object Detection:** Detects 43 different classes of traffic signs.
*   **Custom Training:** Includes scripts to train YOLOv8n on the GTSRB dataset.
*   **Model Evaluation:** Validation metrics to check precision, recall, and mAP.
*   **Multi-Source Inference:**
    *   Detect signs in static images.
    *   Real-time detection via Webcam.
    *   Process YouTube videos automatically (using `yt-dlp`).

## 🛠️ Prerequisites

Make sure you have Python installed. You will need to install the following libraries:

```bash
pip install ultralytics opencv-python matplotlib yt-dlp kagglehub pyyaml
```

*   `ultralytics`: For the YOLO model.
*   `opencv-python`: For image and video processing.
*   `yt-dlp`: For downloading YouTube videos for processing.
*   `kagglehub`: For easily downloading the dataset.

## 📂 Dataset Preparation

This project uses the **GTSRB** dataset.

1.  **Download Data:** You can download it automatically using the provided script or manually from Kaggle.
    ```python
    import kagglehub
    path = kagglehub.dataset_download("doganozcan/traffic-sign-gtrb")
    print("Path to dataset files:", path)
    ```

2.  **Data Configuration (`data.yaml`):**
    Create a `data.yaml` file to configure your dataset paths and class names.

    ```yaml
    path: /path/to/your/dataset  # Root directory
    train: train/images          # Training images path
    val: test/images             # Validation images path
    
    nc: 43                       # Number of classes
    names:
      - '20_speed'
      - '30_speed'
      - '50_speed'
      - '60_speed'
      - '70_speed'
      - '80_speed'
      - '80_lifted'
      - '100_speed'
      - '120_speed'
      - 'no_overtaking_general'
      - 'no_overtaking_trucks'
      - 'right_of_way_crossing'
      - 'right_of_way_general'
      - 'give_way'
      - 'stop'
      - 'no_way_general'
      - 'no_way_trucks'
      - 'no_way_one_way'
      - 'attention_general'
      - 'attention_left_turn'
      - 'attention_right_turn'
      - 'attention_curvy'
      - 'attention_bumpers'
      - 'attention_slippery'
      - 'attention_bottleneck'
      - 'attention_construction'
      - 'attention_traffic_light'
      - 'attention_pedestrian'
      - 'attention_children'
      - 'attention_bikes'
      - 'attention_snowflake'
      - 'attention_deer'
      - 'lifted_general'
      - 'turn_right'
      - 'turn_left'
      - 'turn_straight'
      - 'turn_straight_right'
      - 'turn_straight_left'
      - 'turn_right_down'
      - 'turn_left_down'
      - 'turn_circle'
      - 'lifted_no_overtaking_general'
      - 'lifted_no_overtaking_trucks'
    ```

## 🏋️ Training the Model

To train the model on the GTSRB dataset, run the following script.

**Note:** Ensure you update the `data` path in `model.train()` to point to your `data.yaml`.

```python
from ultralytics import YOLO

def train_yolo():
    # Load YOLOv8 pretrained model (nano version)
    model = YOLO("yolov8n.pt")

    # Train the model
    model.train(
        data="data.yaml",         # Path to your data.yaml
        epochs=50,
        imgsz=640,
        batch=16,
        workers=4,
        project="traffic_sign_detection",
        name="yolov8_traffic_signs",
        exist_ok=True
    )

if __name__ == "__main__":
    train_yolo()
```

After training, the best weights will be saved in `traffic_sign_detection/yolov8_traffic_signs/weights/best.pt`.

## 📊 Model Evaluation

To validate the model and see metrics (mAP, Precision, Recall):

```python
from ultralytics import YOLO

def evaluate_model():
    # Load your trained model
    model = YOLO("traffic_sign_detection/yolov8_traffic_signs/weights/best.pt")
    
    # Run validation
    metrics = model.val()
    
    print("Evaluation Results:")
    print(metrics)

if __name__ == "__main__":
    evaluate_model()
```

## 🚀 Inference / Prediction

### 1. Image Prediction
Detect traffic signs in a single image.

```python
from ultralytics import YOLO
import cv2

def predict_image(image_path):
    # Load the trained model
    # UPDATE THIS PATH to where your best.pt is located
    model = YOLO("best.pt")

    # Run prediction
    results = model(image_path, conf=0.5)

    # Show result
    annotated_image = results[0].plot()
    cv2.imshow("Traffic Sign Detection", annotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    predict_image("path/to/your/image.png")
```

### 2. YouTube Video Detection
This script downloads a YouTube video (if not present), processes it frame by frame, and saves the output with bounding boxes.

```python
from ultralytics import YOLO
import cv2
import yt_dlp

def process_youtube_video():
    # 1. Load Model
    model = YOLO("best.pt") 

    # 2. Setup YouTube Download
    youtube_url = "https://www.youtube.com/watch?v=wqctLW0Hb_0" # Example URL
    video_file = "youtube_video.mp4"

    ydl_opts = {
        'outtmpl': video_file,
        'format': 'mp4'
    }

    print("Downloading video...")
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([youtube_url])

    # 3. Open Video
    cap = cv2.VideoCapture(video_file)
    
    # Setup Output Writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(
        "detected_youtube_output.mp4",
        fourcc,
        30, # FPS
        (int(cap.get(3)), int(cap.get(4))) # Frame Size
    )

    print("Processing video...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Inference
        results = model(frame, conf=0.4)
        annotated_frame = results[0].plot()

        # Write to output file
        out.write(annotated_frame)
        
        # Display (Optional, can be slow on some machines)
        cv2.imshow("Traffic Sign Detection - YouTube", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("Done! Output saved to detected_youtube_output.mp4")

if __name__ == "__main__":
    process_youtube_video()
```

### 3. Webcam Detection
Connect to your webcam for real-time detection.

```python
from ultralytics import YOLO
import cv2

model = YOLO("best.pt")
cap = cv2.VideoCapture(0) # 0 for default webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.4)
    annotated_frame = results[0].plot()

    cv2.imshow("Traffic Sign Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
```

## ⚠️ Important Notes
1.  **Paths:** Ensure you replace file paths (e.g., `C:\\private\\AI...` or `/kaggle/input/...`) with the actual paths on your machine.
2.  **Hardware:** Training is recommended on a GPU. If using CPU, reduce the `batch` size and `workers` in the training script.
3.  **Dependencies:** `yt-dlp` is required for processing YouTube links and may need updating frequently as YouTube changes its API.

