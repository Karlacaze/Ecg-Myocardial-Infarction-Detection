# ECG Analysis and Classification Project

A comprehensive **machine learning** and **deep learning** framework for automated detection of cardiac conditions from electrocardiogram (ECG) signals.

---

## About

The **ECG Analysis and Classification Project** is an end-to-end framework for detecting cardiac conditions automatically from **electrocardiogram (ECG) signals** using **machine learning** and **deep learning** techniques.  

Built around the **PTB Diagnostic ECG Database**, this project integrates three main modules:  
1. **Data Processing and Visualization** – signal cleaning, normalization, and exploratory analysis of multi-lead ECG data.  
2. **Binary Classification Model** – a **ResNet-based CNN** trained to distinguish **Myocardial Infarction (MI)** from healthy controls.  
3. **Multi-Class Classification Model** – an extended architecture capable of identifying multiple cardiac abnormalities, including **MI** and **Bundle Branch Block**.  

The framework emphasizes **data quality**, **balanced training**, and **robust validation** through techniques such as **SMOTE**, **Focal Loss**, **K-Fold cross-validation**, and **ensemble learning**.  
Its extensive **data augmentation pipeline** simulates realistic ECG variations, enhancing model generalization across patient conditions.

This repository serves as a foundation for **academic research**, **clinical data exploration**, and **AI-driven diagnostic tool development** in cardiology.

---

## Overview

This repository presents **three complementary approaches** to ECG signal analysis and classification using the **PTB Diagnostic ECG Database**.  
It covers everything from **data exploration** to **deep learning model development**, forming a complete pipeline for cardiac condition detection.

---

## Project Structure

### 1. Data Processing and Visualization  
**File:** `random_forest_ml.ipynb`

Exploratory data analysis and preprocessing pipeline for ECG signals.

**Key Features**
- Multi-lead ECG extraction (15 leads)  
- Comprehensive visualization tools  
- Data quality assessment and preprocessing  
- Signal normalization and standardization  

---

### 2. Binary Classification Model  
**File:** `deep_learning.ipynb`

Deep learning model for **Myocardial Infarction (MI)** detection using a **ResNet-based CNN**.

**Key Features**
- **Task:** Binary classification (MI vs Non-MI)  
- **Dataset:** 511 samples (368 MI, 143 Non-MI)  
- **Architecture:** ResNet with residual blocks (64 → 128 → 256 → 512 filters)

**Advanced Techniques**
- SMOTE for class balancing  
- Focal Loss (γ = 2.0, α = 0.75)  
- Extensive data augmentation (9 techniques + mixup)  
- K-Fold cross-validation  
- Ensemble learning (5 models)

---

### 3. Multi-Class Classification Model  
**File:** `cnn_3_classes.ipynb`

Extended CNN model for classifying **multiple cardiac conditions**.

**Key Features**
- **Output Classes:**  
  - Healthy Controls  
  - Myocardial Infarction  
  - Bundle Branch Block  
- **Architecture:** Softmax activation + categorical cross-entropy  
- **SMOTE Adaptation:** Multi-class imbalance handling  
- **Metrics:** Per-class precision, recall, and F1-score  

---

## Dataset Source

**PTB Diagnostic ECG Database**

**Specifications**
- 15-lead ECG recordings  
- 5,000 time points per signal  
- Stratified train/validation/test splits  
- Multiple cardiac conditions represented  

---

## Technologies

- **Deep Learning:** TensorFlow, Keras  
- **Data Processing:** NumPy, Pandas, wfdb  
- **Visualization:** Matplotlib, Seaborn  
- **Machine Learning Tools:** scikit-learn, imbalanced-learn  
- **Class Balancing:** SMOTE  

---

## Getting Started

### Prerequisites
```bash
pip install numpy pandas matplotlib seaborn
pip install tensorflow keras scikit-learn
pip install imbalanced-learn wfdb
```

### Usage

**1. Data Exploration**
```bash
jupyter notebook random_forest_ml.ipynb
```

**2. Binary Classification (MI Detection)**
```bash
jupyter notebook deep_learning.ipynb
```

**3. Multi-Class Classification**
```bash
jupyter notebook cnn_3_classes.ipynb
```

---

## Model Performance

### Binary Classification
- **Metrics:** Accuracy, Precision, Recall, F1-Score, AUC-ROC  
- **Validation:** K-Fold cross-validation + ensemble predictions  
- **Training:** ~2.5M parameters, up to 100 epochs with early stopping  

### Multi-Class Classification
- **Metrics:** Confusion matrix and per-class classification report  
- **Classes:** 3 cardiac conditions  
- **Architecture:** ResNet base with modified output layer  

---

## Technical Highlights

### Data Augmentation Pipeline
- Gaussian noise injection (60%)  
- Amplitude scaling (60%)  
- Temporal shifting (40%)  
- Baseline wander simulation (40%)  
- Lead polarity inversion (25%)  
- Random cropping (30%)  
- Power line interference (30%)  
- Lead dropout (20%)  
- Time warping (25%)  
- Mixup augmentation (30%)  

---

### ResNet Architecture
```
Input (5000, 15)
 ├── Conv1D (64) + BatchNorm + MaxPool + Dropout
 ├── Residual Block (64)
 ├── Residual Block (128) × 2
 ├── Residual Block (256) × 2
 ├── Residual Block (512)
 ├── Global Average Pooling
 ├── Dense (256) + Dropout
 ├── Dense (128) + Dropout
 └── Output (Sigmoid / Softmax)
```

**Key Features**
- Class imbalance handling: SMOTE + Focal Loss  
- Regularization: L2, Dropout, Batch Normalization  
- Learning rate scheduling: Warm-up + decay strategy  
- Data augmentation: 10 diverse techniques  
- Ensemble learning: Multi-model aggregation  
- Robust validation: K-Fold + stratified splits  

---

## Citation

If you use this code or dataset, please cite:  
**PTB Diagnostic ECG Database (PhysioNet)**  

---


---

## License

This project is available for **academic and research purposes**.

---
