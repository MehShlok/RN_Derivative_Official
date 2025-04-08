import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, auc, confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras.regularizers import l2

def run_experiment(df):
    # Extract features and labels
    train_features = df.iloc[:, :-1].values
    train_labels = df.iloc[:, -1].values

    # Remove NaNs
    nan_indices = np.isnan(train_features).any(axis=1)
    train_features = train_features[~nan_indices]
    train_labels = train_labels[~nan_indices]

    # Separate normal and anomaly classes
    normal_data = train_features[train_labels == 0]
    anomaly_data = train_features[train_labels == 1]

    # Shuffle before split
    np.random.shuffle(normal_data)
    np.random.shuffle(anomaly_data)

    if len(anomaly_data) == 0:
        print("Skipping: No anomaly samples present.")
        return None, None, None, None, None

    # Train/test split for normal
    len_train_normal = int(0.7 * len(normal_data))
    X_train_normal = normal_data[:len_train_normal]
    X_test_normal = normal_data[len_train_normal:]
    y_train_normal = np.zeros(len(X_train_normal))
    y_test_normal = np.zeros(len(X_test_normal))

    # Train/test split for anomaly
    if len(anomaly_data) < 7:
        len_train_anomaly = 1
    else:
        len_train_anomaly = max(1, int(0.15 * len(anomaly_data)))

    X_train_anomaly = anomaly_data[:len_train_anomaly]
    X_test_anomaly = anomaly_data[len_train_anomaly:]
    y_train_anomaly = np.ones(len(X_train_anomaly))
    y_test_anomaly = np.ones(len(X_test_anomaly))

    # Combine train and test sets
    X_train = np.vstack((X_train_normal, X_train_anomaly))
    y_train = np.hstack((y_train_normal, y_train_anomaly))
    X_test = np.vstack((X_test_normal, X_test_anomaly))
    y_test = np.hstack((y_test_normal, y_test_anomaly))

    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Check if stratification is safe (each class must have at least 2 samples)
    unique, counts = np.unique(y_test, return_counts=True)
    if len(unique) > 1 and np.all(counts >= 2):
        stratify_labels = y_test
    else:
        stratify_labels = None


    X_val_scaled, X_final_scaled, y_val, y_final = train_test_split(
        X_test_scaled, y_test, test_size=0.5, stratify=stratify_labels, random_state=42
    )

    # Custom weighted loss
    def custom_loss(alpha):
        def loss(y_true, y_pred):
            base_loss = tf.keras.losses.binary_crossentropy(y_true, y_pred)
            class_weights = tf.where(tf.equal(y_true, 1), 1.0, alpha)
            return tf.reduce_mean(class_weights * base_loss)
        return loss

    # Model definition
    input_shape = X_train.shape[1]
    model = Sequential([
        Dense(128, activation='relu', kernel_regularizer=l2(0.01), input_shape=(input_shape,)),
        BatchNormalization(),
        Dropout(0.5),
        Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])

    alpha = len(X_train_anomaly) / (len(X_train_normal) + 1e-6) # WEIGHT ASSIGNED TO THE NORMAL CLASS INSTANCES
    optimizer = Adam(learning_rate=0.0005)
    model.compile(optimizer=optimizer, loss=custom_loss(alpha), metrics=['accuracy'])

    # Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

    # Training
    model.fit(X_train_scaled, y_train,
              validation_data=(X_val_scaled, y_val),
              epochs=50, batch_size=32,
              callbacks=[early_stopping, reduce_lr])

    # Threshold selection using validation
    val_predictions = model.predict(X_val_scaled).flatten()
    if len(np.unique(y_val)) > 1:
        fpr, tpr, thresholds = roc_curve(y_val, val_predictions)
        roc_auc_val = auc(fpr, tpr)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
    else:
        roc_auc_val = np.nan
        optimal_threshold = 0.5

    # Final test evaluation
    final_predictions = model.predict(X_final_scaled).flatten()
    predicted_labels = (final_predictions >= optimal_threshold).astype(int)
    precision = precision_score(y_final, predicted_labels, zero_division=0)
    recall = recall_score(y_final, predicted_labels, zero_division=0)
    f1 = f1_score(y_final, predicted_labels, zero_division=0)
    cm = confusion_matrix(y_final, predicted_labels)

    if len(np.unique(y_final)) > 1:
        fpr_final, tpr_final, _ = roc_curve(y_final, final_predictions)
        roc_auc_final = auc(fpr_final, tpr_final)
    else:
        roc_auc_final = np.nan

    return precision, recall, f1, cm, roc_auc_final


# =========================
# Main loop for all datasets
# =========================

dataset_url = []
dataset_names = []

file_path = '' #For saving the required results

for d in range(len(dataset_url)):
    df = pd.read_csv(dataset_url[d]).dropna()
    Dataset = dataset_names[d]

    num_repeats = 5
    repeat_precisions = []
    repeat_recalls = []
    repeat_f1_scores = []
    repeat_confusion_matrices = []
    repeat_roc_aucs = []

    for r in range(num_repeats):
        tf.random.set_seed(42 + r)
        np.random.seed(42 + r)
        precision, recall, f1, cm, roc_auc = run_experiment(df)

        # Skip if anomalies were too few
        if precision is None:
            continue

        repeat_precisions.append(precision)
        repeat_recalls.append(recall)
        repeat_f1_scores.append(f1)
        repeat_confusion_matrices.append(cm)
        repeat_roc_aucs.append(roc_auc)

    if not repeat_precisions:
        print(f"Dataset {Dataset} skipped due to insufficient anomalies.\n")
        continue

    # Mean & std
    mean_precision = np.mean(repeat_precisions)
    std_precision = np.std(repeat_precisions)
    mean_recall = np.mean(repeat_recalls)
    std_recall = np.std(repeat_recalls)
    mean_f1 = np.mean(repeat_f1_scores)
    std_f1 = np.std(repeat_f1_scores)
    mean_roc_auc = np.nanmean(repeat_roc_aucs)

    # Save to file
    with open(file_path, 'a') as f:
        f.write(f"\nDataset: {Dataset}\n")
        f.write("Repeat, Precision, Recall, F1 Score, Confusion Matrix, ROC AUC\n")
        for i in range(len(repeat_precisions)):
            f.write(f"{i+1}, {repeat_precisions[i]}, {repeat_recalls[i]}, {repeat_f1_scores[i]},\n{repeat_confusion_matrices[i]}, {repeat_roc_aucs[i]}\n")

        f.write(f"\nSummary Statistics:\n")
        f.write(f"Mean Precision: {mean_precision}, Std Precision: {std_precision}\n")
        f.write(f"Mean Recall: {mean_recall}, Std Recall: {std_recall}\n")
        f.write(f"Mean F1 Score: {mean_f1}, Std F1 Score: {std_f1}\n")
        f.write(f"Mean ROC AUC: {mean_roc_auc}\n")
