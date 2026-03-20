"""
model/train.py — Train the Fraud Detection XGBoost model.

Critical fix vs original:
  The original code fit the StandardScaler on the FULL dataset before splitting,
  leaking test-set statistics (mean/std) into training. This inflates all reported
  metrics. The correct pipeline is: SPLIT FIRST → fit scaler on train only → transform.

Pipeline:
  1. Load creditcard.csv
  2. Train/test split (stratified, 80/20)          ← MUST come first
  3. Fit StandardScaler on X_train only             ← no leakage
  4. Transform X_test using training statistics     ← correct evaluation
  5. SMOTE oversampling on training set only
  6. Train XGBoost with tuned hyperparameters
  7. Evaluate on held-out test set
  8. Save model, scaler, and metrics.json
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
DATA_PATH    = os.path.join(BASE_DIR, "..", "data", "creditcard.csv")
MODEL_PATH   = os.path.join(BASE_DIR, "fraud_model.pkl")
SCALER_PATH  = os.path.join(BASE_DIR, "scaler.pkl")
METRICS_PATH = os.path.join(BASE_DIR, "metrics.json")


# ── 1. Load data ──────────────────────────────────────────────────────────────
print("📂  Loading dataset…")
df = pd.read_csv(DATA_PATH)
print(f"     Rows: {len(df):,} | Fraud rate: {df['Class'].mean()*100:.3f}%")

X = df.drop(columns=["Class"])
y = df["Class"]


# ── 2. Split FIRST (prevents data leakage into scaler/metrics) ────────────────
print("✂️   Splitting train/test (80/20, stratified)…")
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# ── 3. Fit scaler on training data only ───────────────────────────────────────
print("📏  Fitting StandardScaler on training data only…")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)     # fit + transform
X_test_scaled  = scaler.transform(X_test_raw)          # transform only — no leakage

X_train_df = pd.DataFrame(X_train_scaled, columns=X.columns)
X_test_df  = pd.DataFrame(X_test_scaled,  columns=X.columns)


# ── 4. SMOTE oversampling (only on training data) ─────────────────────────────
print("⚖️   Balancing classes with SMOTE…")
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train_df, y_train)
print(f"     After SMOTE: {y_train_res.value_counts().to_dict()}")


# ── 5. Train XGBoost ──────────────────────────────────────────────────────────
print("🚀  Training XGBoost classifier…")
model = XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    eval_metric="aucpr",
    random_state=42,
    n_jobs=-1,
)
model.fit(X_train_res, y_train_res)


# ── 6. Evaluate on held-out test set ─────────────────────────────────────────
print("\n📊  Evaluating on held-out test set…")
y_proba = model.predict_proba(X_test_df)[:, 1]
y_pred  = model.predict(X_test_df)

roc_auc = roc_auc_score(y_test, y_proba)
pr_auc  = average_precision_score(y_test, y_proba)
cm      = confusion_matrix(y_test, y_pred)
report  = classification_report(y_test, y_pred, output_dict=True)

tn, fp, fn, tp = cm.ravel()

print("─" * 50)
print(f"  ROC-AUC   : {roc_auc:.4f}")
print(f"  PR-AUC    : {pr_auc:.4f}")
print(f"  Precision : {report['1']['precision']:.4f}")
print(f"  Recall    : {report['1']['recall']:.4f}")
print(f"  F1-Score  : {report['1']['f1-score']:.4f}")
print(f"  Confusion Matrix:")
print(f"    TN={tn:,}  FP={fp}")
print(f"    FN={fn}  TP={tp}")
print(f"  False alarm rate : {fp/(fp+tn)*100:.2f}%")
print(f"  Fraud catch rate : {tp/(tp+fn)*100:.2f}%")
print("─" * 50)


# ── 7. Save artifacts ─────────────────────────────────────────────────────────
joblib.dump(model,  MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)

metrics = {
    "roc_auc":          round(roc_auc, 4),
    "pr_auc":           round(pr_auc, 4),
    "precision":        round(report["1"]["precision"], 4),
    "recall":           round(report["1"]["recall"], 4),
    "f1_score":         round(report["1"]["f1-score"], 4),
    "false_alarm_rate": round(fp / (fp + tn) * 100, 2),
    "fraud_catch_rate": round(tp / (tp + fn) * 100, 2),
    "confusion_matrix": {
        "tn": int(tn), "fp": int(fp),
        "fn": int(fn), "tp": int(tp),
    },
    "dataset": {
        "total_rows":   len(df),
        "fraud_count":  int(df["Class"].sum()),
        "fraud_rate_pct": round(df["Class"].mean() * 100, 4),
    },
}

with open(METRICS_PATH, "w") as f:
    json.dump(metrics, f, indent=2)

print(f"\n✅  Saved: {MODEL_PATH}")
print(f"✅  Saved: {SCALER_PATH}")
print(f"✅  Saved: {METRICS_PATH}")
print("\nTraining complete. Run the API with: uvicorn api.main:app --reload")
