import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import roc_curve, auc, recall_score, precision_score, f1_score
from sklearn.preprocessing import StandardScaler
import warnings
from datetime import datetime
 
def cblof(ts, small_cluster_threshold=0.02):
    """ CBLOF (Cluster-Based Local Outlier Factor) implementation. """
    kmeans = KMeans(n_clusters=5, random_state=0).fit(ts)
    labels = kmeans.labels_
 
    cluster_sizes = np.bincount(labels)
    large_clusters = np.where(cluster_sizes > small_cluster_threshold * len(ts))[0]
    small_clusters = np.where(cluster_sizes <= small_cluster_threshold * len(ts))[0]
    cblof_scores = np.zeros(len(ts))
    for i, x in enumerate(ts):
        cluster_id = labels[i]
        centroid = kmeans.cluster_centers_[cluster_id]
        dist = np.linalg.norm(x - centroid)
        if cluster_id in small_clusters:
            nearest_large_cluster = large_clusters[np.argmin([np.linalg.norm(x - kmeans.cluster_centers_[lc]) for lc in large_clusters])]
            dist_to_large = np.linalg.norm(x - kmeans.cluster_centers_[nearest_large_cluster])
            cblof_scores[i] = dist * dist_to_large  
        else:
            cblof_scores[i] = dist 
 
    return cblof_scores
 
def evaluate_dataset(X, y_true, dataset_name):
 
    try:
        cblof_scores = cblof(X)
 
        # Defining anomalies using the 95th percentile as the threshold
        anomaly_threshold = np.percentile(cblof_scores, 95)
        y_pred = (cblof_scores >= anomaly_threshold).astype(int)
 
        fpr, tpr, _ = roc_curve(y_true, cblof_scores)
        metrics = {
            'dataset': dataset_name,
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred),
            'auc_roc': auc(fpr, tpr),
            'n_samples': len(X),
            'n_features': X.shape[1],
            'anomaly_ratio': np.mean(y_true)
        }
        return metrics
 
    except Exception as e:
        print(f"Error processing dataset {dataset_name}: {str(e)}")
        return None
 
def process_multiple_datasets(dataset_urls):
    """
    Process multiple datasets and return combined results
    """
    results = []
 
    for url in dataset_urls:
        try:
            dataset_name = url.split('/')[-1].replace('.csv', '')
            print(f"\nProcessing dataset: {dataset_name}")
            data = pd.read_csv(url)
            X = data.iloc[:, :-1].values
            y_true = data.iloc[:, -1].values
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
 
            metrics = evaluate_dataset(X_scaled, y_true, dataset_name)
            if metrics:
                results.append(metrics)
                print(f"Results for {dataset_name}:")
                for key, value in metrics.items():
                    if isinstance(value, float):
                        print(f"{key}: {value:.4f}")
                    else:
                        print(f"{key}: {value}")
 
        except Exception as e:
            print(f"Error loading dataset {url}: {str(e)}")
            continue
 
    return results
 
if __name__ == '__main__':
    # Suppressing warnings
    warnings.filterwarnings('ignore')
    
    dataset_urls = []
    results = process_multiple_datasets(dataset_urls)
    if results:
        df_results = pd.DataFrame(results)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f'cblof_results_{timestamp}.csv'
        df_results.to_csv(output_file, index=False)
        print(f"\nResults saved to {output_file}")
 
        # Displaying summary statistics
        print("\nSummary Statistics:")
        print(df_results.describe())