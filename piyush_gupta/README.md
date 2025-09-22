# Image Segmentation Project

## Project Overview

This project focuses on **image segmentation** using a YOLO-based model. The workflow covers **data collection, manual labeling, model training, validation, and testing**.

---

## Dataset

* **Trained/Labelled Data**: `https://drive.google.com/drive/folders/1hD3CxS2ITz-XBsL0XAHGXP63A0F8LEbG?usp=sharing`
* **Validation Data**: `https://drive.google.com/drive/folders/1kTk3l1HNjWy_O6wPEW6WYrYvLv6lTgq3?usp=sharing`
* **Testing Unlabelled Data**: `https://drive.google.com/drive/folders/1rXHWN2rihxO9EzSxzn4PvLy-S8sgfDVm?usp=sharing`

> **Note:** Images were manually annotated using Labellerr.

---

## Video Demonstration

* **Project Demo Video Link:** `[PLACEHOLDER: Link to video demo]`

---

## Workflow

1. **Data Collection:**

   * Downloaded **114 raw images** containing pedestrians, vehicles, roads, and two-wheelers.

2. **Manual Labeling:**

   * Annotated all images using **Labellerr** to create the labeled dataset.

3. **Validation Dataset:**

   * Selected **53 images** for validation purposes.

4. **Model Training:**

   * Trained a YOLO segmentation model on the **114 training images**.

5. **Model Validation:**

   * Validated the model on the **53 validation images**.

6. **Evaluation Metrics:**

   * **Curve Placeholder:** `[PLACEHOLDER: Add training/validation curves here]`
   * **Confusion Matrix Placeholder:** `[PLACEHOLDER: Add confusion matrix image here]`

7. **Testing & Label Generation:**

   * Generated predictions for **20 test images** using `model.predict`.
   * Plan to verify predictions manually after fetching the **Client ID from Labellerr**.

---

## Notebook Link

* **Model Training Notebook:** `Training.ipynb in this folder`

---

## Results

* **Metrics and Graphs :**

  * Training/Validation loss curves
  * mAP scores per class
  * Confusion matrix

---

## Demo & Project Links

* **Demo Link:** `[PLACEHOLDER: Demo link]`
* **Project GitHub Link:** `[PLACEHOLDER: GitHub repository link]`

---

## Future Work

* Integrate automatic uploading of model-generated labels to Labellerr using **Client ID**.
* Expand dataset with more diverse images to improve model generalization.

