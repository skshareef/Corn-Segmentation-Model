# 🌽 Corn-Segmentation-Model

This repository implements a deep learning model for **segmenting corn crop images** into three classes:
- 🌱 Green Vegetation  
- 🍂 Senescent Vegetation  
- 🟫 Background  

Inspired by the **SegVeg** architecture, this model utilizes a U-Net-like DeepLabV3 with a ResNet-50 backbone. It was developed and trained on **Google Colab**, optimized for segmentation of high-resolution crop images.

---

---

## 🧠 About the Model

We use a U-Net model with a ResNet-50 encoder, modified for **2-class semantic segmentation**. The output map categorizes each pixel in the image into:
- **Class 0:** Background  
- **Class 1:** Green Vegetation  

---

## 🚀 Quick Start

### 1. Clone and Install

```bash
git clone https://github.com/your-username/Corn-Segmentation-Model.git
cd Corn-Segmentation-Model
pip install -r requirements.txt

```

### 2. Train the Model
```bash
python train.py
```
### 3. Run Inference
```bash
python infer.py --img path/to/your_image.jpg

```



### ☁️ Google Colab
This project was trained on Google Colab using a free GPU. A Colab notebook version is included here:
📓 notebooks/demo_colab.ipynb
