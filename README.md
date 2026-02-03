# ECG-Based Biometric Authentication Using Self-Supervised Learning

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![License](https://img.shields.io/badge/License-MIT-green)

A robust ECG-based biometric authentication system employing Convolutional Neural Networks (CNN) and self-supervised contrastive learning for continuous user authentication.

## 🎯 Overview

This project implements an electrocardiogram (ECG)-based user authentication system that leverages:
- **1D CNN encoder** for automated feature extraction from raw ECG signals
- **Self-supervised contrastive learning** (Siamese network) for training without extensive labeled data
- **R-peak to R-peak (R2R) segmentation** for biologically-aligned signal processing
- **Pearson Correlation Coefficient (PCC)** similarity metric for authentication

### Key Achievements
- **Best Equal Error Rate (EER)**: ~13.35% on ECG-ID dataset
- **Optimal configuration**: Siamese network with training margin Λ = 0.50
- **Systematic hyperparameter tuning** across 12 different margin values
- **Resource-efficient** design suitable for IoT edge device deployment

## 📊 Dataset

**ECG-ID Database** [file:4]
- 310 ECG recordings from 90 unique individuals
- 20-second Lead I ECG signals
- Sampling rate: 500 Hz (resampled to 200 Hz)
- Resolution: 12-bit (±10mV range)
- Subject demographics: 44 male, 46 female, aged 13-75 years
- Temporal variability: Recordings collected over up to 6 months

## 🏗️ Architecture

### CNN Encoder
- **Input**: 1000-sample ECG segment (5 seconds at 200 Hz)
- **Architecture**: 6 convolutional blocks (Conv1D + BatchNorm + ReLU + MaxPooling)
- **Output**: 2034-dimensional feature vector
- **Parameters**: ~8.3 million

### Contrastive Learning Framework
The project explored two frameworks[file:4]:

1. **Triplet Loss** (Phase 2)
   - Trains on (Anchor, Positive, Negative) triplets
   - Goal: D(A,P) < D(A,N) - Λ

2. **Siamese Contrastive Loss** (Phase 3) ✅
   - Trains on positive/negative pairs
   - Shared-weight CNN encoders
   - Custom PCC-based similarity metric

## 🔬 Methodology

### Phase 1: Exploration
- Investigated ECG uniqueness through template matching
- Identified within-person variability challenges
- Confirmed need for learned feature representations

### Phase 2: Triplet Learning with NPD
- Implemented No Peak Detection (NPD) segmentation
- Trained with Triplet contrastive loss
- **Results**: High error rates (FAR: 21-51%, FRR: 20-45%)

### Phase 3: Siamese Learning with R2R
- Switched to R-peak to R-peak (R2R) segmentation
- Implemented Siamese contrastive learning
- Systematic hyperparameter tuning of margin Λ
- **Results**: Best EER of 13.35% with Λ = 0.50

### Signal Processing Pipeline
1. **Bandpass Filtering**: 0.5-40 Hz (Butterworth filter)
2. **Resampling**: 500 Hz → 200 Hz
3. **R-peak Detection**: Pan-Tompkins algorithm
4. **R2R Segmentation**: Extract & resample R-R intervals, concatenate 5 intervals
5. **Normalization**: Scale to [-512, 512]

## 📈 Results

### Performance Metrics (Best Model: Λ = 0.50)

| Metric | Value | Description |
|--------|-------|-------------|
| **EER** | **13.35%** | Equal Error Rate |
| **Accuracy** | 85.73% | Overall classification accuracy |
| **FAR** | 14.41% | False Acceptance Rate at EER |
| **FRR** | 14.15% | False Rejection Rate at EER |
| **Precision** | 86.61% | Positive predictive value |
| **Recall** | 85.85% | True Positive Rate |
| **F1 Score** | 86.23% | Harmonic mean of precision/recall |

### Hyperparameter Tuning Results

Training margin Λ significantly impacts authentication performance[file:4]:

| Training Λ | EER Threshold | EER Rate |
|-----------|---------------|----------|
| **0.50** | **0.9042** | **13.65%** |
| 0.60 | 0.9147 | 15.45% |
| 0.40 | 0.9061 | 15.66% |
| 0.70 | 0.9437 | 17.73% |
| 0.80 | 0.9510 | 17.38% |

## Getting Started

### Prerequisites
```bash
Python 3.8+
TensorFlow 2.x
NumPy
SciPy
Pandas
Matplotlib
scikit-learn
wfdb
