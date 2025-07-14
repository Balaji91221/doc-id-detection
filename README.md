
# 📄 Multi-Document Photo Detection using YOLOv8


> A deep learning project that detects and extracts profile photos from Indian government identity documents (PAN, Passport, Driving License) using YOLOv8 and Roboflow datasets.

---

## 🚀 Project Overview

This project leverages the YOLOv8 object detection model to automate the extraction of profile photos from scanned images of identity documents. It integrates multiple datasets—PAN, Passport, and Driving License—and trains a unified model to detect and crop the photo section from each document type.

---

## 📌 Features

* ✅ Detects and extracts profile photos from:

  * PAN cards
  * Passports
  * Driving Licenses
* 🔁 Merges multiple Roboflow datasets into a single YOLOv8 training set
* 📦 Custom-trained YOLOv8n model with 3 classes
* 🖼️ Outputs cropped profile photos using bounding box detection

---

## 🛠️ Tech Stack

* [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
* [Roboflow](https://roboflow.com/)
* Python (OpenCV, PIL)
* Google Colab / Jupyter Notebook

---

## 📂 Dataset Structure

Datasets were downloaded from Roboflow:

* **PAN Dataset** – `pan-twznp-xudzz`
* **Passport Dataset** – `passport-btyua-20czk`
* **Driving License Dataset** – `driving-license-annotation-qmxd3`

Merged and re-labeled as:

* Class `0`: PAN
* Class `1`: Passport
* Class `2`: Driving License

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

Download datasets:

```python
from roboflow import Roboflow

rf = Roboflow(api_key="YOUR_API_KEY")
pan = rf.workspace("b-8pp1v").project("pan-twznp-xudzz").version(1).download("yolov8")
passport = rf.workspace("b-8pp1v").project("passport-btyua-20czk").version(1).download("yolov8")
dl = rf.workspace("b-8pp1v").project("driving-license-annotation-qmxd3").version(1).download("yolov8")
```

Merge datasets and remap labels accordingly (see code in notebook).

---

## 🧠 Model Training

```bash
!yolo task=detect mode=train model=yolov8n.pt data=/content/merged_dataset/data.yaml epochs=100 imgsz=640
```

The trained model will be saved in:

```
runs/detect/train/weights/best.pt
```

---


## 📊 Results

* High confidence detections across all three ID types.
* Cropped profile photos displayed using PIL.
* Confidence threshold tuned to `0.15` for better sensitivity.

