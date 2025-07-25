# 💳 Credit Risk Scoring App

A lightweight, interactive web application that predicts **credit default risk** using machine learning. Built with **Streamlit**, powered by **Gradient Boosting**, and deployed on **Hugging Face Spaces**.

> Determine whether a customer is likely to default on credit based on demographic and financial attributes — with model explainability via SHAP!

---

## Demo

👉 [Launch on Hugging Face Spaces](https://huggingface.co/spaces/aparnaaaw/credit_analysis)  

---

## About the Project

This project implements a **cost-sensitive credit scoring model** trained on German credit data. It integrates:

- Supervised Learning (Gradient Boosting Classifier)
- Class Imbalance Handling with **SMOTE**
- **Unsupervised Anomaly Detection** using Isolation Forest
- Full Scikit-learn **pipeline serialization**
- Model Explainability with **SHAP**
- **Streamlit UI** for real-time interaction
- Deployed via Docker on **Hugging Face Spaces**

---

## ⚙️ How It Works

The model uses a two-phase learning strategy:

### 1. 🔍 Unsupervised Anomaly Detection

Before training the supervised classifier, the model runs an **Isolation Forest** on the input data to detect outliers — potential credit risks not labeled in the dataset.

```python
iso_forest = IsolationForest(contamination=0.1, random_state=42)
outlier_labels = iso_forest.fit_predict(X)
```
- Anomalies (outliers) labeled -1 are treated as likely defaulters.
- These pseudo-labels are then combined with actual labels to refine the training dataset — effectively bootstrapping supervision in a semi-supervised fashion.

### 2. ✅ Supervised Learning with Gradient Boosting
- Once labels are finalized (real + inferred), the pipeline continues with:
- ColumnTransformer for preprocessing:
- Scaling numeric features
- Imputation + OneHotEncoding for categorical
- SMOTE for class balancing
- GradientBoostingClassifier for robust classification
- SHAP for interpretability

## 🧪 Model Training

To train the model:

```python
python train_model.py
```

The pipeline:
 - Loads and cleans the data
 - Applies Isolation Forest for pseudo-labeling
 - Combines with real labels
 - Applies SMOTE to balance
 - Trains a Gradient Boosting classifier
 - Saves the final pipeline + SHAP explainer

## 📈 SHAP Explainability
SHAP (SHapley Additive exPlanations) is used to explain model predictions. After each prediction, a waterfall plot displays feature-level contributions to the risk score.

## 🧪 Features
 - Form-based user input via Streamlit
 - Risk prediction with probability
 - Feature importance visualization
 - Integrated unsupervised + supervised modeling
 - Dockerized deployment

## 📦 Run Locally
```python
git clone https://github.com/your-username/credit-risk-scoring-app.git
cd credit-risk-scoring-app

pip install -r requirements.txt

streamlit run app.py
```
