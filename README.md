# AI-Text-Detector (DetectAI)
Detect whether text is **human-written or AI-generated** using a **hybrid approach with RoBERTa and XGBoost**.

---

## 📌 Overview
This project focuses on classifying text into two categories:
- **Human-written**
- **AI-generated**

It uses:
- **RoBERTa** (Transformer-based model) for feature extraction and fine-tuning.
- **XGBoost** for classification on RoBERTa embeddings.

---

## ✅ Features
- Text preprocessing (cleaning, tokenization)
- Fine-tuned **RoBERTa classifier** using PyTorch
- Mixed Precision Training for speed
- **XGBoost classifier** on extracted embeddings
- Confusion matrix, accuracy, and classification report
- Kaggle-ready submission generation

---

## 🛠 Technologies Used
- **Python**
- **PyTorch**
- **HuggingFace Transformers**
- **XGBoost**
- **Scikit-learn**
- **Matplotlib & Seaborn**
- **Pandas & NumPy**
- **NLTK**

---

## 📂 Dataset
The dataset should include:
- `train.csv` → Columns: `text`, `generated`
- `test.csv` → Columns: `id`, `text`

Update the **`load_data()`** function paths for your dataset location.

---

## ⚡ Installation
Clone the repository:
```bash
git clone https://github.com/<your-username>/AI-Text-Detector.git
cd AI-Text-Detector
```
---

## Install dependencies:
```
pip install -r requirements.txt

```
---
## ▶ Usage
Run the Jupyter Notebook:

```bash

jupyter notebook datascape-duo_code.ipynb
Main steps:
Load dataset and preprocess text

Tokenize using RoBERTa

Train RoBERTa classifier

Evaluate with confusion matrix and metrics

Extract embeddings for XGBoost

Generate predictions and save CSV files
```
---
## 📊 Visualizations
Training vs Validation Loss curve

Training vs Validation Accuracy curve

Confusion Matrix for predictions

## 📦 Outputs
```
submission.csv → Predictions from RoBERTa

submission_xgboost.csv → Predictions from XGBoost

train_features.csv, val_features.csv, test_features.csv → Saved embeddings
```
