# DancePose Module

The **DancePose** module is designed to detect a dancer’s body and extract keypoint coordinates from video frames.  
It serves as the foundation for motion analysis and performance evaluation in the *Coach Assistant* project.

---

## 📁 Project Structure and File Descriptions

- **configs/default.yaml** — Configuration file containing video path, model name, confidence thresholds, and output options.  
- **scripts/run_pose.py** — Main entry script that loads configuration, runs YOLOv8-Pose inference, and saves results.  
- **src/inference/pose_infer.py** — Core inference logic: loads the pretrained model and performs per-frame pose detection.  
- **src/viz/overlay.py** — Visualization utilities: draws skeletons and keypoints on video frames to produce overlay output.  
- **src/utils/io_utils.py** — Handles input/output operations such as directory creation, JSONL writing, and logging.  

---

## How to Use

1. Install the dependencies listed in the requirements section.
2. The data I used comes from this dataset:
https://google.github.io/aistplusplus_dataset/factsfigures.html
3. Open the configuration file 'configs/default.yaml' and modify paths for your video input and model weights if necessary.  
4. Run run_pose.py

---

## Notes

The current prototype supports single-person detection only.
Model weights (yolov8s-pose.pt or similar) will be downloaded automatically if not found locally.
The module is built on the Ultralytics YOLOv8-Pose framework and provides pose data for further dance-quality evaluation.
