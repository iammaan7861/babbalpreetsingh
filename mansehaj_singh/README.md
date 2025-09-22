
# Object Detection & Tracking with YOLOv8  
**Author:** Mansehaj Singh  

---

## 📌 Project Overview
This project implements **object detection and tracking** using the pretrained YOLOv8 model (Ultralytics).  
The model is fine-tuned on a custom dataset containing two classes:  

- `person`  
- `car`  

The pipeline covers:
- Dataset preparation (train/val/test split + annotations in YOLO format)  
- Model training with YOLOv8  
- Evaluation on validation/test sets  
- Demo inference on custom videos/images  

---

## 📂 Repository Structure

mansehaj_singh/
│── README.md # Project explanation
│── YOLOv8_object_detection.ipynb # Google Colab Notebook (training + inference)
│── Report.pdf # Documentation & report

---
🚀 How to Run

1️⃣ Clone the repository
git clone https://github.com/mansehaj-singh/labeller_project.git
cd labeller_project/mansehaj_singh

---

2️⃣ Open the Notebook

Upload YOLOv8_object_detection.ipynb to Google Colab and run step by step.
It covers:

Installing dependencies

Preparing dataset

Training with YOLOv8

Testing & inference
---
📊 Dataset

Total Images: 280

Train: 160

Validation: 40

Test: 80

Format: YOLOv8 (images + .txt annotations)

Classes: person, car

---

📈 Results

Model: yolov8m.pt (fine-tuned)

Epochs: 100

Image size: 640

Metric	Score
mAP50	X.XX
mAP50-95	X.XX
Precision	X.XX
Recall	X.XX

![Model tracking demo]("C:\Users\Asus\OneDrive\Desktop\Labeller_Project\car_vid.gif")
---

📝 Documentation

The complete project journey, challenges, and step-by-step guide are available in the Report.pdf

---

🙌 Acknowledgements

Ultralytics YOLOv8

Labellerr
 for dataset annotation and project guidelines


---










