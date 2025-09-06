# 🌲 Cover Type Classification with Advanced Neural Networks

![Status](https://img.shields.io/badge/Status-Completed-brightgreen) ![Python](https://img.shields.io/badge/Python-3.11-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-2.1-orange) ![Accuracy](https://img.shields.io/badge/Accuracy-96%25-brightgreen)

This repository presents a **high-performance neural network solution** for predicting forest cover types based on cartographic features. The project leverages **deep learning best practices** including **layer normalization, dropout, and stochastic depth**, achieving strong generalization and stability during training.

---

## 📌 Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Data Preprocessing](#data-preprocessing)
- [Model Architecture](#model-architecture)
- [Training Strategy](#training-strategy)
- [Evaluation Metrics](#evaluation-metrics)
- [Results](#results)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

---

## 🌟 Project Overview
The **Cover Type Classification** task aims to predict forest cover type classes based on **54 cartographic variables** including elevation, slope, aspect, and soil type.  

Key achievements:
- Reduced the original 7 classes to 5 high-relevance classes.
- Applied **advanced neural network design** for high accuracy.
- Implemented **robust preprocessing and training pipeline** for stability and generalization.

---

## 📊 Dataset
Dataset: [Forest Cover Type dataset (UCI ML Repository)](https://archive.ics.uci.edu/ml/datasets/covertype)  

- **Features:** 54 features  
  - 10 numerical (e.g., Elevation, Slope, Aspect)  
  - 44 categorical (one-hot encoded for soil type and wilderness area)  
- **Target:** Cover type (classes 1, 2, 3, 6, 7)  

**Filtering and label mapping:**
```python
train_filtered = train_transformed[train_transformed["Cover_Type"].isin([1, 2, 3, 6, 7])]
X = train_filtered.drop(columns=["Id", "Cover_Type"])
y = train_filtered["Cover_Type"]
label_map = {1: 0, 2: 1, 3: 2, 6: 3, 7: 4}
y = y.map(label_map)
