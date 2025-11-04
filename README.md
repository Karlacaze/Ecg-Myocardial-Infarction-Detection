ECG Analysis and Classification Project

A comprehensive machine learning and deep learning framework for automated detection of cardiac conditions from electrocardiogram (ECG) signals.

Overview

This repository presents three complementary approaches to ECG signal analysis and classification using the PTB Diagnostic ECG Database.
It covers everything from data exploration to deep learning model development, forming a complete pipeline for cardiac condition detection.

Project Structure
1. Data Processing and Visualization

File: random_forest_ml.ipynb

Exploratory data analysis and preprocessing pipeline for ECG signals.

Key Features

Multi-lead ECG extraction (15 leads)

Comprehensive visualization tools

Data quality assessment and preprocessing

Signal normalization and standardization

2. Binary Classification Model

File: deep_learning.ipynb

Deep learning model for Myocardial Infarction (MI) detection using a ResNet-based CNN.

Key Features

Task: Binary classification (MI vs Non-MI)

Dataset: 511 samples (368 MI, 143 Non-MI)

Architecture: ResNet with residual blocks (64 → 128 → 256 → 512 filters)

Advanced Techniques

SMOTE for class balancing

Focal Loss (γ = 2.0, α = 0.75)

Extensive data augmentation (9 techniques + mixup)

K-Fold cross-validation

Ensemble learning (5 models)

3. Multi-Class Classification Model

File: cnn_3_classes.ipynb

Extended CNN model for classifying multiple cardiac conditions.

Key Features

Output Classes:

Healthy Controls

Myocardial Infarction

Bundle Branch Block

Architecture: Softmax activation + categorical cross-entropy

SMOTE Adaptation: Multi-class imbalance handling

Metrics: Per-class precision, recall, and F1-score

Dataset Source

PTB Diagnostic ECG Database

Specifications

15-lead ECG recordings

5,000 time points per signal

Stratified train/validation/test splits

Multiple cardiac conditions represented

Technologies

Deep Learning: TensorFlow, Keras

Data Processing: NumPy, Pandas, wfdb

Visualization: Matplotlib, Seaborn

Machine Learning Tools: scikit-learn, imbalanced-learn

Class Balancing: SMOTE

Getting Started
Prerequisites
pip install numpy pandas matplotlib seaborn
pip install tensorflow keras scikit-learn
pip install imbalanced-learn wfdb

Usage

1. Data Exploration

jupyter notebook random_forest_ml.ipynb


2. Binary Classification (MI Detection)

jupyter notebook deep_learning.ipynb


3. Multi-Class Classification

jupyter notebook cnn_3_classes.ipynb

Model Performance
Binary Classification

Metrics: Accuracy, Precision, Recall, F1-Score, AUC-ROC

Validation: K-Fold cross-validation + ensemble predictions

Training: ~2.5M parameters, up to 100 epochs with early stopping

Multi-Class Classification

Metrics: Confusion matrix and per-class classification report

Classes: 3 cardiac conditions

Architecture: ResNet base with modified output layer
