import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import KernelDensity
from sklearn.metrics import roc_curve, auc, recall_score, precision_score, f1_score
 
def adjusted_cblof_with_kde(ts, small_cluster_threshold=0.02, min_bandwidth=1e-3, epsilon=1e-10):
    """
    Adjusted CBLOF with KDE-based density ratio correction and numerical stability improvements.
 
    Parameters:
    -----------
    ts : array-like
        Input time series data
    small_cluster_threshold : float
        Threshold for determining small clusters
    min_bandwidth : float
        Minimum bandwidth for KDE to prevent collapse
    epsilon : float
        Small constant to prevent division by zero
    """
    kmeans = KMeans(n_clusters=5, random_state=0).fit(ts)
    labels = kmeans.labels_
 
    cluster_sizes = np.bincount(labels)
    centroids = kmeans.cluster_centers_
 
    large_clusters = np.where(cluster_sizes > small_cluster_threshold * len(ts))[0]
    small_clusters = np.where(cluster_sizes <= small_cluster_threshold * len(ts))[0]
 
    if len(large_clusters) == 0:
        large_clusters = np.array([np.argmax(cluster_sizes)])
        small_clusters = np.array([i for i in range(len(cluster_sizes)) if i != large_clusters[0]])
 
    # Initializing KDE models with minimum bandwidth
    kde_models = {}
    for cluster_id in range(len(centroids)):
        cluster_points = ts[labels == cluster_id]
        # Adding small random noise to prevent singular matrices
        noise = np.random.normal(0, min_bandwidth/10, cluster_points.shape)
        cluster_points = cluster_points + noise
        bandwidth = max(min_bandwidth, 0.5)  # Ensuring minimum bandwidth
        kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(cluster_points)
        kde_models[cluster_id] = kde
 
    cblof_scores = np.zeros(len(ts))
    for i, x in enumerate(ts):
        cluster_id = labels[i]
        dist_to_centroid = np.linalg.norm(x - centroids[cluster_id])
 
        # Adding small noise to point for density estimation
        x_noisy = x + np.random.normal(0, min_bandwidth/10, x.shape)
 
        try:
            density_x = np.exp(kde_models[cluster_id].score_samples(x_noisy.reshape(1, -1)))[0]
            density_x = max(density_x, epsilon) 
        except:
            density_x = epsilon
 
        if cluster_id in large_clusters:
            cblof_scores[i] = (1 / density_x) * dist_to_centroid
        else:
            # Finding nearest large cluster
            nearest_large_cluster = large_clusters[np.argmin([np.linalg.norm(x - centroids[lc]) for lc in large_clusters])]
            dist_to_large_centroid = np.linalg.norm(x - centroids[nearest_large_cluster])
 
            try:
                density_large = np.exp(kde_models[nearest_large_cluster].score_samples(x_noisy.reshape(1, -1)))[0]
                density_large = max(density_large, epsilon)
                density_ratio = density_x / density_large
            except:
                density_ratio = 1.0
 
            cblof_scores[i] = density_ratio * dist_to_large_centroid
 
    # Clipping extreme values
    cblof_scores = np.clip(cblof_scores, 0, np.percentile(cblof_scores, 99.9))
 
    return cblof_scores
 
if __name__ == '__main__':
    dataset_url = []
 
    Dataset_names = ["ALOI"]
    results_file = 'anomaly_detection_results.csv'
    with open(results_file, 'w') as f:
        f.write("Dataset,Precision,Recall,F1-score,AUC-ROC\n")
 
    for d, url in enumerate(dataset_url):
        try:
            print(f"Processing dataset: {Dataset_names[d]}")
 
            data = pd.read_csv(url)
            X = data.iloc[:, :-1].values
            y_true = data.iloc[:, -1].values
            cblof_scores = adjusted_cblof_with_kde(X)
 
            anomaly_threshold = np.percentile(cblof_scores, 95)
            y_pred = (cblof_scores >= anomaly_threshold).astype(int)
            fpr, tpr, thresholds = roc_curve(y_true, cblof_scores)
            auc_score = auc(fpr, tpr)
            recall = recall_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)
 
            with open(results_file, 'a') as f:
                f.write(f"{Dataset_names[d]},{precision:.4f},{recall:.4f},{f1:.4f},{auc_score:.4f}\n")
            print(f"Completed {Dataset_names[d]}: Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}, AUC={auc_score:.4f}")
 
        except Exception as e:
            print(f"Error processing dataset {Dataset_names[d]}: {str(e)}")
            # Write error to results file
            with open(results_file, 'a') as f:
                f.write(f"{Dataset_names[d]},Error,Error,Error,Error\n")
 
    print(f"\nResults have been saved to {results_file}")