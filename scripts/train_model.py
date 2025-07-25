import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
)
from sklearn.ensemble import GradientBoostingClassifier
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import IsolationForest
from preprocess import build_preprocessing_pipeline, load_data

# --- Load and Clean Data ---
df = load_data("../data/german_credit_data.csv")
df = df.dropna(subset=["Age", "Sex", "Job", "Housing", "Credit amount", "Duration"])

# --- Step 1: Anomaly Detection ---
X_anomaly = df.copy()
preprocessor = build_preprocessing_pipeline()
contamination_rate = (X_anomaly["Credit amount"] > X_anomaly["Credit amount"].quantile(0.95)).mean()

pipeline_anomaly = Pipeline([
    ("preprocessor", preprocessor),
    ("anomaly_detector", IsolationForest(contamination=contamination_rate, random_state=42))
])
pipeline_anomaly.fit(X_anomaly)

df["Anomaly"] = pipeline_anomaly.predict(X_anomaly)
df["Risk"] = df["Anomaly"].apply(lambda x: 1 if x == -1 else 0)
df.to_csv("../data/credit_data_with_anomalies.csv", index=False)

# --- PCA Visualization ---
X_transformed = pipeline_anomaly.named_steps["preprocessor"].transform(X_anomaly)
X_pca = PCA(n_components=2).fit_transform(X_transformed)
df["PCA1"], df["PCA2"] = X_pca[:, 0], X_pca[:, 1]

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x="PCA1", y="PCA2", hue="Anomaly", palette={1: "blue", -1: "red"})
plt.title("Anomaly Detection (PCA)")
plt.savefig("../outputs/pca_clusters.png")
plt.close()

# --- Step 2: Classification ---
X = df.drop(columns=["Anomaly", "Risk"])
y = df["Risk"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Fit the preprocessor on training data
preprocessor.fit(X_train)

# Balance training data
X_train_bal, y_train_bal = SMOTE(random_state=42).fit_resample(preprocessor.transform(X_train), y_train)

# Cost-sensitive classifier
classifier = GradientBoostingClassifier(random_state=42)
classifier.fit(X_train_bal, y_train_bal)

# Predict probabilities
X_test_transformed = preprocessor.transform(X_test)
y_proba = classifier.predict_proba(X_test_transformed)[:, 1]

# --- Tune Threshold ---
precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
best_threshold = thresholds[np.argmax(f1_scores)]

print(f"Optimal Threshold (F1): {best_threshold:.4f}")
y_pred_adjusted = (y_proba >= best_threshold).astype(int)

# --- Evaluation ---
report = classification_report(y_test, y_pred_adjusted, output_dict=True)
conf_matrix = confusion_matrix(y_test, y_pred_adjusted)
roc_auc = roc_auc_score(y_test, y_proba)

metrics_df = pd.DataFrame(report).transpose()
metrics_df["roc_auc"] = roc_auc
metrics_df.to_csv("../outputs/evaluation_threshold_adjusted.csv")

# --- Confusion Matrix Plot ---
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix (Threshold Adjusted)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("../outputs/conf_matrix_threshold_adjusted.png")
plt.close()

# --- Precision-Recall Curve ---
plt.figure(figsize=(8, 5))
plt.plot(recalls, precisions, label="Precision-Recall Curve")
plt.axvline(x=recalls[np.argmax(f1_scores)], color="red", linestyle="--", label="Best F1 Recall")
plt.axhline(y=precisions[np.argmax(f1_scores)], color="green", linestyle="--", label="Best F1 Precision")
plt.title("Precision-Recall Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend()
plt.savefig("../outputs/precision_recall_curve.png")
plt.close()

# --- SHAP Explanation ---
explainer = shap.Explainer(classifier, X_test_transformed)
shap_values = explainer(X_test_transformed)

# Helper to get transformed feature names
def get_feature_names(preprocessor):
    feature_names = []
    for name, transformer, columns in preprocessor.transformers_:
        if transformer == "drop":
            continue
        elif transformer == "passthrough":
            feature_names.extend(columns)
        else:
            try:
                names = transformer.get_feature_names_out(columns)
                feature_names.extend(names)
            except:
                feature_names.extend(columns)
    return feature_names

feature_names = get_feature_names(preprocessor)

# SHAP Summary Plot
if shap_values.values.ndim == 3:
    shap.summary_plot(shap_values[..., 1], features=X_test_transformed, feature_names=feature_names, show=False)
else:
    shap.summary_plot(shap_values, features=X_test_transformed, feature_names=feature_names, show=False)

plt.savefig("../outputs/shap_summary_adjusted_threshold.png")
plt.close()

# --- Save model and explainer ---
joblib.dump(classifier, "../models/threshold_tuned_model.pkl")
joblib.dump(preprocessor, "../models/preprocessor.pkl")
joblib.dump(explainer, "../models/shap_explainer.pkl")

print("Training complete. Model, preprocessor, SHAP, and evaluation artifacts saved.")
