# 🌽 Corn-Segmentation-Model

This repository implements a deep learning model for **segmenting corn crop images** into three classes:
- 🌱 Green Vegetation  
- 🍂 Senescent Vegetation  
- 🟫 Background  

Inspired by the **SegVeg** architecture, this model utilizes a U-Net-like DeepLabV3 with a ResNet-50 backbone. It was developed and trained on **Google Colab**, optimized for segmentation of high-resolution crop images.

---

## 📁 Repository Structure

Corn-Segmentation-Model/ ├── train.py # Main training script ├── infer.py # Inference/prediction script ├── config.py # Config for paths and hyperparameters ├── utils/ # Data loading, transforms, model logic │ ├── dataset.py │ ├── model.py │ ├── transforms.py │ └── helpers.py ├── outputs/ # Output folder (predictions, logs, checkpoints) │ ├── checkpoints/ │ ├── logs/ │ └── predictions/ ├── data/ # Directory for your dataset (not included) ├── notebooks/ # Optional: Colab/Jupyter notebooks └── README.md
