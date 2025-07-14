

# 📄 Multi-Document Photo Detection using YOLOv8

> A deep learning project to detect and extract **profile photos** from Indian government identity documents using **YOLOv8** — covering **PAN**, **Passport**, **Driving License**, and **Aadhaar Card**.

---

## 🚀 Project Overview

This project automates the process of detecting and extracting **profile photos** from various Indian identity documents using a single YOLOv8 object detection model trained on merged custom datasets. It improves document processing efficiency for KYC, onboarding, and OCR pipelines.

---

## 📌 Features

* ✅ Detects and extracts profile photos from:

  * PAN Cards
  * Passports
  * Driving Licenses
  * Aadhaar Cards
* 📦 Custom-trained YOLOv8n model with 4 classes
* 📸 Outputs cropped profile photos from raw document images
* 🔄 Merges multiple annotated datasets into one unified pipeline

---

## 🛠️ Tech Stack

* [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
* [Roboflow](https://roboflow.com/)
* Python (OpenCV, PIL, NumPy)
* Google Colab or Jupyter Notebook

---

## 📂 Dataset Structure

### ✅ Datasets Used (from Roboflow):

* **PAN Card:** `pan-twznp-xudzz`
* **Passport:** `passport-btyua-20czk`
* **Driving License:** `driving-license-annotation-qmxd3`
* **Aadhaar Card:** `aadhaar-card-details-mnomu`

### ✅ Class Mappings:

```yaml
0: PAN
1: Passport
2: Driving License
3: Aadhaar
```

### ✅ Directory after merging:

```
merged_dataset/
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
└── data.yaml
```

---

## ⚙️ Setup & Installation

```bash
!pip install roboflow ultralytics opencv-python --quiet
```

### Download datasets:

```python
from roboflow import Roboflow

rf = Roboflow(api_key="YOUR_API_KEY")
pan = rf.workspace("b-8pp1v").project("pan-twznp-xudzz").version(1).download("yolov8")
passport = rf.workspace("b-8pp1v").project("passport-btyua-20czk").version(1).download("yolov8")
dl = rf.workspace("b-8pp1v").project("driving-license-annotation-qmxd3").version(1).download("yolov8")
aadhaar = rf.workspace("b-8pp1v").project("aadhaar-card-details-mnomu").version(1).download("yolov8")
```

### Merge and relabel:

* Use custom Python script to copy and remap labels to class IDs `0–3`.
* Save everything into `merged_dataset/` with a proper `data.yaml`.

---

## 📄 Example `data.yaml`

```yaml
train: /content/merged_dataset/train/images
val: /content/merged_dataset/valid/images

nc: 4
names: ['PAN', 'Passport', 'Driving License', 'Aadhaar']
```

---

## 🧠 Model Training

```bash
!yolo task=detect mode=train model=yolov8n.pt data=/content/merged_dataset/data.yaml epochs=100 imgsz=640
```

Trained weights are saved to:

```
/content/runs/detect/train/weights/best.pt
```

---


## 📊 Results

* Accurate detections for all 4 document types.
* Cropped profile photos display clearly.
* Confidence thresholds tuned for robustness.

---



## 🎯 Future Improvements

* Add OCR layer to extract name, DOB, Aadhaar/PAN number
* Export model to ONNX or TensorFlow Lite
* Integrate with real-time ID verification APIs

---
