ECG Analysis and Classification Project
A comprehensive machine learning and deep learning framework for automated cardiac condition detection from electrocardiogram (ECG) signals.
Overview
This repository contains three complementary approaches to ECG signal analysis and classification using the PTB Diagnostic ECG Database. The projects range from data exploration to advanced deep learning models, providing a complete pipeline for cardiac condition detection.
Project Structure
1. Data Processing and Visualization (random_forest_ml.ipynb)
Exploratory data analysis and preprocessing pipeline for ECG signals.
Key Features:

Multi-lead ECG signal extraction (15 leads)
Comprehensive visualization tools
Data quality assessment and preprocessing
Signal normalization and standardization

2. Binary Classification Model (deep_learning.ipynb)
Deep learning model for Myocardial Infarction (MI) detection using ResNet-based CNN.
Key Features:

Binary Classification: MI vs Non-MI
Dataset: 511 samples (368 MI, 143 Non-MI)
Architecture: ResNet with residual blocks (64→128→256→512 filters)
Advanced Techniques:

SMOTE for class balancing
Focal Loss (γ=2.0, α=0.75)
Extensive data augmentation (9 techniques + mixup)
K-Fold cross-validation
Ensemble learning (5 models)



3. Multi-Class Classification Model (cnn_3_classes.ipynb)
Extended CNN model for classifying multiple cardiac conditions.
Key Features:

Multi-Class Output: 3 cardiac conditions

Healthy Controls
Myocardial Infarction
Bundle Branch Block


Modified Architecture: Softmax activation + categorical crossentropy
Adapted SMOTE: Multi-class imbalance handling
Performance Metrics: Per-class precision, recall, F1-score

Dataset
Source: PTB Diagnostic ECG Database
Specifications:

15-lead ECG recordings
5,000 time points per signal
Stratified train/validation/test splits
Multiple cardiac conditions represented

Technologies

Deep Learning: TensorFlow, Keras
Data Processing: NumPy, Pandas, wfdb
Visualization: Matplotlib, Seaborn
ML Tools: scikit-learn, imbalanced-learn
Class Balancing: SMOTE

Getting Started
Prerequisites
bashpip install numpy pandas matplotlib seaborn
pip install tensorflow keras scikit-learn
pip install imbalanced-learn wfdb
Usage

Data Exploration

bash   jupyter notebook random_forest_ml.ipynb

Binary Classification (MI Detection)

bash   jupyter notebook deep_learning.ipynb

Multi-Class Classification

bash   jupyter notebook cnn_3_classes.ipynb
Model Performance
Binary Classification

Metrics: Accuracy, Precision, Recall, F1-Score, AUC-ROC
Validation: K-Fold cross-validation + ensemble predictions
Training: ~2.5M parameters, 100 epochs max with early stopping

Multi-Class Classification

Metrics: Per-class confusion matrix and classification report
Classes: 3 cardiac conditions
Architecture: Same ResNet base with modified output layer

Technical Highlights
Data Augmentation Pipeline

Gaussian noise injection (60%)
Amplitude scaling (60%)
Temporal shifting (40%)
Baseline wander simulation (40%)
Lead polarity inversion (25%)
Random cropping (30%)
Power line interference (30%)
Lead dropout (20%)
Time warping (25%)
Mixup augmentation (30%)

ResNet Architecture
Input (5000, 15)
├── Conv1D (64) + BatchNorm + MaxPool + Dropout
├── Residual Block (64)
├── Residual Block (128) × 2
├── Residual Block (256) × 2
├── Residual Block (512)
├── Global Average Pooling
├── Dense (256) + Dropout
├── Dense (128) + Dropout
└── Output (Sigmoid/Softmax)
Key Features

Class Imbalance Handling: SMOTE + Focal Loss
Regularization: L2, Dropout, Batch Normalization
Learning Rate Scheduling: Warmup + decay strategy
Data Augmentation: 10 diverse techniques
Ensemble Learning: Multiple model aggregation
Robust Validation: K-Fold + stratified splits

Citation
If you use this code or dataset, please cite:

PTB Diagnostic ECG Database (PhysioNet)

Authors
Karla - ECG Analysis and Deep Learning Implementation
License
This project is available for academic and research purposes.
Contributing
Contributions, issues, and feature requests are welcome!
Contact
For questions or collaboration opportunities, please open an issue.