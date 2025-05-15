import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
 
import torch
from deepod.models.tabular.slad import SLAD

df = pd.read_csv("")
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
 
# --- Separate normal and anomaly samples ---
X_normal = X_scaled[y == 0]
y_normal = y[y == 0]
 
X_anomaly = X_scaled[y == 1]
y_anomaly = y[y == 1]
 
# --- Manual Split: Anomalous Samples ---
n_anom = len(X_anomaly)
n_anom_train = max(1, int(0.01 * n_anom))
n_anom_val = max(1, int(0.01 * n_anom))
n_anom_test = n_anom - n_anom_train - n_anom_val
 
X_anom_train = X_anomaly[:n_anom_train]
y_anom_train = y_anomaly[:n_anom_train]
 
X_anom_val = X_anomaly[n_anom_train:n_anom_train + n_anom_val]
y_anom_val = y_anomaly[n_anom_train:n_anom_train + n_anom_val]
 
X_anom_test = X_anomaly[n_anom_train + n_anom_val:]
y_anom_test = y_anomaly[n_anom_train + n_anom_val:]
 
# --- Manual Split: Normal Samples ---
n_norm = len(X_normal)
n_norm_train = int(0.6 * n_norm)
n_norm_val = int(0.1 * n_norm)
n_norm_test = n_norm - n_norm_train - n_norm_val
 
X_norm_train = X_normal[:n_norm_train]
y_norm_train = y_normal[:n_norm_train]
 
X_norm_val = X_normal[n_norm_train:n_norm_train + n_norm_val]
y_norm_val = y_normal[n_norm_train:n_norm_train + n_norm_val]
 
X_norm_test = X_normal[n_norm_train + n_norm_val:]
y_norm_test = y_normal[n_norm_train + n_norm_val:]
 
# --- Final Splits ---
X_train = np.vstack([X_norm_train, X_anom_train])
y_train = np.concatenate([y_norm_train, y_anom_train])
 
X_val = np.vstack([X_norm_val, X_anom_val])
y_val = np.concatenate([y_norm_val, y_anom_val])
 
X_test = np.vstack([X_norm_test, X_anom_test])
y_test = np.concatenate([y_norm_test, y_anom_test])
 
# --- Shuffle sets ---
train_idx = np.random.permutation(len(X_train))
X_train, y_train = X_train[train_idx], y_train[train_idx]
 
val_idx = np.random.permutation(len(X_val))
X_val, y_val = X_val[val_idx], y_val[val_idx]
 
test_idx = np.random.permutation(len(X_test))
X_test, y_test = X_test[test_idx], y_test[test_idx]
 
# --- Train SLAD ---
print("Training SLAD...")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = SLAD(epochs=20, device=device)
model.fit(X_train, y_train)
 
# --- Inference on Validation ---
print("Selecting threshold using Youden's J statistic...")
val_scores = model.decision_function(X_val)
fpr, tpr, thresholds = roc_curve(y_val, val_scores)
j_scores = tpr - fpr
best_idx = np.argmax(j_scores)
best_threshold = thresholds[best_idx]
print(f"Best threshold: {best_threshold:.4f} (J = {j_scores[best_idx]:.4f})")

print("Evaluating on test set...")
test_scores = model.decision_function(X_test)
y_pred = (test_scores >= best_threshold).astype(int)
 
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
auc = roc_auc_score(y_test, test_scores)
print("\n--- Test Metrics ---")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"ROC AUC:   {auc:.4f}")