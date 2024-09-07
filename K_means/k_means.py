import warnings
import sys
import os
import pandas as pd
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

# Suppress future warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Add the root directory of the project to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.main import load_and_preprocess_data

def generate_cluster_names(cluster_centers, feature_names, thresholds=None):
    """Generate descriptive names for clusters based on feature values."""
    if thresholds is None:
        thresholds = {'high': 0.75, 'low': 0.25}
    
    cluster_names = []
    for center in cluster_centers:
        name_parts = []
        for i, feature_value in enumerate(center):
            feature_name = feature_names[i]
            high_threshold = np.percentile(cluster_centers[:, i], thresholds['high'])
            low_threshold = np.percentile(cluster_centers[:, i], thresholds['low'])
            if feature_value >= high_threshold:
                name_parts.append(f"High {feature_name.split()[0]}")
            elif feature_value <= low_threshold:
                name_parts.append(f"Low {feature_name.split()[0]}")
        
        cluster_name = ', '.join(name_parts[:2]) if name_parts else "Average"
        cluster_names.append(cluster_name)
    
    return cluster_names

def perform_clustering(n_clusters=5, random_state=42, save_path=None, sample_size=1000, max_features_for_visualization=5):
    """Perform K-Means clustering on the TikTok data and visualize the results."""
    # Load and preprocess the data
    tiktok_data = load_and_preprocess_data()

    # Sample the data if it's too large to handle
    if len(tiktok_data) > sample_size:
        tiktok_data = tiktok_data.sample(n=sample_size, random_state=random_state)

    numerical_columns = tiktok_data.select_dtypes(include='number').columns.tolist()

    # K-Means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    tiktok_data['KMeans_Cluster'] = kmeans.fit_predict(tiktok_data[numerical_columns])

    # Generate cluster names
    kmeans_cluster_names = generate_cluster_names(kmeans.cluster_centers_, numerical_columns)
    tiktok_data['KMeans_Cluster Name'] = tiktok_data['KMeans_Cluster'].map(lambda x: kmeans_cluster_names[x])

    # Calculate validation scores
    silhouette_avg = silhouette_score(tiktok_data[numerical_columns], tiktok_data['KMeans_Cluster'])
    davies_bouldin_avg = davies_bouldin_score(tiktok_data[numerical_columns], tiktok_data['KMeans_Cluster'])

    # Visualize clusters (limit the number of features to avoid memory issues)
    visualize_clusters(tiktok_data, numerical_columns[:max_features_for_visualization], save_path)

    # Output clustering results
    print_cluster_results(kmeans.cluster_centers_, numerical_columns, silhouette_avg, davies_bouldin_avg)

def visualize_clusters(data, features, save_path):
    """Create and save visualizations of the clusters."""
    unique_clusters = data['KMeans_Cluster Name'].nunique()
    palette = sns.color_palette("husl", n_colors=unique_clusters)

    # Limit the number of variables to plot
    limited_features = features[:5]  # Only plot the first 5 features
    sns.pairplot(data, hue='KMeans_Cluster Name', vars=limited_features, palette=palette)

    plt.suptitle('K-Means Clustering of TikTok Accounts', y=1.02)

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, 'k-means_clusters.png'))
        plt.close()
    else:
        plt.show()

def print_cluster_results(centroids, feature_names, silhouette, davies_bouldin):
    """Print the clustering results including centroids and validation scores."""
    print("\nK-Means Cluster Centroids:\n", pd.DataFrame(centroids, columns=feature_names))
    print(f"\nK-Means Silhouette Score: {silhouette}")
    print(f"K-Means Davies-Bouldin Index: {davies_bouldin}")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(base_dir, 'outcomes')
    perform_clustering(save_path=save_path)
