# Object Tracking with Labellerr – Ankit Kumar

## 📌 Overview
This project is part of the **Labellerr Campus Hiring Assignment**.  
It demonstrates building an **end-to-end object tracker** for cars and persons using:
- **Labellerr** for dataset annotation
- **YOLOv8-Seg** for segmentation model training
- **ByteTrack** for multi-object tracking
- **Streamlit** for deployment demo

---

## 🚀 Project Workflow
1. **Dataset Preparation**
   - Uploaded ~100 raw images (cars & persons) to Labellerr
   - Annotated using **Segment Anything Model (SAM)**
   - Exported dataset in **COCO Instance Segmentation format**

2. **Data Preprocessing**
   - Used custom Colab notebook `prepare_labellerr_dataset_easy.ipynb`
   - Split dataset into **train/val/test**
   - Generated `data.yaml` for YOLOv8

3. **Model Training**
   - Framework: Ultralytics YOLOv8-Seg
   - Weights: `yolov8n-seg.pt`
   - Epochs: 50, Image size: 640, Batch: 8
   - Trained on Google Colab GPU

4. **Evaluation**
   - mAP50: ~0.72  
   - mAP50-95: ~0.58  
   - Precision: ~0.75  
   - Recall: ~0.70  
   - IoU (avg): ~0.65  

5. **Tracking with ByteTrack**
   - Integrated ByteTrack for real-time object tracking in video
   - Produced tracking results (`tracks.json` and annotated video)

6. **Deployment**
   - Built `streamlit_app.py` for demo
   - Upload video → runs YOLOv8-Seg + ByteTrack → shows live tracking

---

## 📂 Repository Structure
