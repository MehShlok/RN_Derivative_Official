import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
import torch
from deepod.models.tabular.icl import ICL
import os
 
# ---------- CONFIG ----------
dataset_urls = []
 
output_csv = "icl_results.csv"
 
# ---------- UTILS ----------
def manual_split(X_scaled, y):
    X_normal = X_scaled[y == 0]
    y_normal = y[y == 0]
 
    X_anomaly = X_scaled[y == 1]
    y_anomaly = y[y == 1]
 
    # --- Anomalies ---
    n_anom = len(X_anomaly)
    n_train_anom = max(1, int(0.01 * n_anom))
    n_val_anom = max(1, int(0.01 * n_anom))
 
    X_anom_train = X_anomaly[:n_train_anom]
    y_anom_train = y_anomaly[:n_train_anom]
 
    X_anom_val = X_anomaly[n_train_anom:n_train_anom + n_val_anom]
    y_anom_val = y_anomaly[n_train_anom:n_train_anom + n_val_anom]
 
    X_anom_test = X_anomaly[n_train_anom + n_val_anom:]
    y_anom_test = y_anomaly[n_train_anom + n_val_anom:]
 
    # --- Normals ---
    n_norm = len(X_normal)
    n_train_norm = int(0.6 * n_norm)
    n_val_norm = int(0.1 * n_norm)
 
    X_norm_train = X_normal[:n_train_norm]
    y_norm_train = y_normal[:n_train_norm]
 
    X_norm_val = X_normal[n_train_norm:n_train_norm + n_val_norm]
    y_norm_val = y_normal[n_train_norm:n_train_norm + n_val_norm]
 
    X_norm_test = X_normal[n_train_norm + n_val_norm:]
    y_norm_test = y_normal[n_train_norm + n_val_norm:]
 
    # Combine
    X_train = np.vstack([X_norm_train, X_anom_train])
    y_train = np.concatenate([y_norm_train, y_anom_train])
 
    X_val = np.vstack([X_norm_val, X_anom_val])
    y_val = np.concatenate([y_norm_val, y_anom_val])
 
    X_test = np.vstack([X_norm_test, X_anom_test])
    y_test = np.concatenate([y_norm_test, y_anom_test])
 
    # Shuffle
    X_train, y_train = shuffle(X_train, y_train)
    X_val, y_val = shuffle(X_val, y_val)
    X_test, y_test = shuffle(X_test, y_test)
 
    return X_train, y_train, X_val, y_val, X_test, y_test
 
 
def shuffle(X, y):
    idx = np.random.permutation(len(X))
    return X[idx], y[idx]
 
 
def select_threshold_youden(y_true, scores):
    print("Selecting threshold using Youden's J statistic...")
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    return thresholds[best_idx]
 
 
# ---------- MAIN ----------
results = []
 
device = 'cuda' if torch.cuda.is_available() else 'cpu'
 
for url in dataset_urls:
    try:
        print(f"\n=== Processing Dataset: {os.path.basename(url)} ===")
        df = pd.read_csv(url)
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        X_train, y_train, X_val, y_val, X_test, y_test = manual_split(X_scaled, y)
        model = ICL(epochs=20, device=device)
        model.fit(X_train, y_train)
 
        # Threshold on val set using Youden's J
        val_scores = model.decision_function(X_val)
        threshold = select_threshold_youden(y_val, val_scores)

        test_scores = model.decision_function(X_test)
        y_pred = (test_scores >= threshold).astype(int)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, test_scores)
 
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1 Score:  {f1:.4f}")
        print(f"ROC AUC:   {auc:.4f}")
 
        results.append({
            "Dataset": os.path.basename(url),
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            "ROC AUC": auc
        })
 
    except Exception as e:
        print(f"Failed on dataset {url}: {e}")
        results.append({
            "Dataset": os.path.basename(url),
            "Precision": None,
            "Recall": None,
            "F1 Score": None,
            "ROC AUC": None,
            "Error": str(e)
        })
 
# Save to CSV
results_df = pd.DataFrame(results)
results_df.to_csv(output_csv, index=False)
print(f"\n✅ Results saved to '{output_csv}'")