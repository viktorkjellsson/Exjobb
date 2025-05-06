from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
import sys
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.cluster import DBSCAN
import plotly.graph_objects as go
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform


def kmeans_clustering(normalized_pca_components, df, n_clusters):
    """
    Description: Perform K-Means clustering on the normalized PCA components and visualize the clusters.
    Args:
        normalized_pca_components (nparray): The PCA components normalized using StandardScaler.
        df (pd DataFrame): The original DataFrame including the timestamps.
        n_clusters (int): The number of clusters to create.

    Returns:
        data_with_KMeans (pd DataFrame): The original DataFrame including the timestamps with the addition of the cluster labels.
    """
   
    kmeans = KMeans(n_clusters, random_state=42)
    clusters = kmeans.fit_predict(normalized_pca_components)

    # Add cluster labels to your original data (without overwriting)
    data_with_KMeans = df.copy()  # Make a copy to preserve the original DataFrame

    # Insert the clusters as the second column (at index 1)
    data_with_KMeans.insert(1, 'Cluster', clusters)

    # Count the number of data points assigned to each cluster
    cluster_counts = {i: sum(clusters == i) for i in range(n_clusters)}

    # Plot the clusters
    plt.figure(figsize=(10, 6))
    scatter = sns.scatterplot(x=normalized_pca_components[:, 0], y=normalized_pca_components[:, 1], hue=clusters, palette="viridis", s=100, alpha=0.7)

    # Create custom labels for the legend with the cluster counts
    legend_labels = [f'Cluster {i} (n={cluster_counts[i]})' for i in range(n_clusters)]
    handles, _ = scatter.get_legend_handles_labels()

    # Set the custom labels in the legend
    plt.legend(handles=handles, labels=legend_labels, title='Cluster')

    # Label the axes
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title("PCA + KMeans Clustering")

    # Show the plot
    plt.show()

    # Show the updated DataFrame with the Cluster column as the second column
    return data_with_KMeans


def gmm_clustering(normalized_pca_components, df, n_clusters):
    """
    Description: Perform GMM clustering on the normalized PCA components and visualize the clusters.
    Args:
        normalized_pca_components (nparray): The PCA components normalized using StandardScaler.
        df (pd DataFrame): The original DataFrame including the timestamps.
        n_clusters (int): The number of clusters to create.

    Returns:
        data_with_gmm (pd DataFrame): The original DataFrame including the timestamps with the addition of the cluster labels.
    """
   
    gmm = GaussianMixture(n_clusters, random_state=42)
    clusters = gmm.fit_predict(normalized_pca_components)

    # Extract cluster probabilities
    probabilities = gmm.predict_proba(normalized_pca_components)

    # Get the probability of the assigned cluster
    assigned_prob = probabilities[np.arange(len(clusters)), clusters]

    # Add cluster labels to your original data (without overwriting)
    data_with_gmm = df.copy()  # Make a copy to preserve the original DataFrame

    # Insert the clusters as the second column (at index 1)
    data_with_gmm.insert(1, 'Cluster', clusters)

    data_with_gmm.insert(2, 'Assigned_Cluster_Prob', assigned_prob)  # Insert probability for the assigned cluster


    # Count the number of data points assigned to each cluster
    cluster_counts = {i: sum(clusters == i) for i in range(n_clusters)}

    # Plot the clusters
    plt.figure(figsize=(10, 6))
    scatter = sns.scatterplot(x=normalized_pca_components[:, 0], y=normalized_pca_components[:, 1], hue=clusters, palette="viridis", s=100, alpha=0.7)

    # Create custom labels for the legend with the cluster counts
    legend_labels = [f'Cluster {i} (n={cluster_counts[i]})' for i in range(n_clusters)]
    handles, _ = scatter.get_legend_handles_labels()

    # Set the custom labels in the legend
    plt.legend(handles=handles, labels=legend_labels, title='Cluster')

    # Label the axes
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title("PCA + GMM Clustering")

    # Show the plot
    plt.show()

    # Show the updated DataFrame with the Cluster column as the second column
    return data_with_gmm

def kl_divergence(mu0, cov0, mu1, cov1):
    d = mu0.shape[0]
    cov1_inv = np.linalg.inv(cov1)
    trace_term = np.trace(cov1_inv @ cov0)
    diff = mu1 - mu0
    quad_term = diff.T @ cov1_inv @ diff
    log_det_term = np.log(np.linalg.det(cov1) / np.linalg.det(cov0))
    return 0.5 * (trace_term + quad_term - d + log_det_term)

def jeffreys_divergence(mu0, cov0, mu1, cov1):
    return 0.5 * (kl_divergence(mu0, cov0, mu1, cov1) + kl_divergence(mu1, cov1, mu0, cov0))

def merge_clusters_by_divergence(dpgmm, labels, threshold):
    unique_labels = np.unique(labels)
    if len(unique_labels) <= 1:
        return labels

    means = dpgmm.means_
    covariances = dpgmm.covariances_
    label_indices = [label for label in unique_labels if np.sum(labels == label) > 0]

    distance_matrix = np.zeros((len(label_indices), len(label_indices)))
    for i, idx_i in enumerate(label_indices):
        for j, idx_j in enumerate(label_indices):
            if i < j:
                dist = jeffreys_divergence(means[idx_i], covariances[idx_i], means[idx_j], covariances[idx_j])
                distance_matrix[i, j] = distance_matrix[j, i] = dist

    # Compute condensed distance matrix
    condensed_distance = squareform(distance_matrix, checks=False)
    Z = linkage(condensed_distance, method='average')
    new_cluster_ids = fcluster(Z, t=threshold, criterion='distance')
    label_map = {old: new for old, new in zip(label_indices, new_cluster_ids)}
    merged_labels = np.array([label_map.get(label, -1) for label in labels])

    return merged_labels


def streaming_dpgmm_clustering(normalized_pca_components, df, prior, n_points, window_size, step_size, max_components, merge_threshold, merge_within_window):
    all_labels = np.full(len(df), -1)
    all_probs = np.zeros(len(df))
    all_results = []

    # === Initial fit ===
    initial_data = normalized_pca_components[:n_points]
    dpgmm_init = BayesianGaussianMixture(
        n_components=max_components,
        covariance_type='full',
        weight_concentration_prior_type='dirichlet_process',
        weight_concentration_prior=prior,
        max_iter=1000,
        tol=1e-3,
        init_params='kmeans',
        random_state=42
    )
    dpgmm_init.fit(initial_data)
    initial_labels = dpgmm_init.predict(initial_data)
    if merge_within_window:
        initial_labels = merge_clusters_by_divergence(dpgmm_init, initial_labels, merge_threshold)
    initial_probs = dpgmm_init.predict_proba(initial_data)
    initial_max_probs = initial_probs[np.arange(len(initial_labels)), initial_labels]
    all_labels[:n_points] = initial_labels
    all_probs[:n_points] = initial_max_probs

    print(f"Initial fit => Clusters used: {np.sum(dpgmm_init.weights_ > 0.01)}")

    # === Streaming windows ===
    for start in range(n_points, len(df) - window_size + 1, step_size):
        end = start + window_size
        window_data = normalized_pca_components[start:end]
        memory_data = normalized_pca_components[:end]

        dpgmm = BayesianGaussianMixture(
            n_components=max_components,
            covariance_type='full',
            weight_concentration_prior_type='dirichlet_process',
            weight_concentration_prior=prior,
            max_iter=1500,
            tol=1e-3,
            init_params='kmeans',
            random_state=42
        )
        dpgmm.fit(memory_data)

        labels = dpgmm.predict(window_data)
        if merge_within_window:
            labels = merge_clusters_by_divergence(dpgmm, labels, merge_threshold)

        probs = dpgmm.predict_proba(window_data)
        max_probs = probs[np.arange(len(labels)), labels]

        all_labels[start:end] = labels
        all_probs[start:end] = max_probs

        active_clusters = np.sum(dpgmm.weights_ > 0.01)
        print(f"Window {start}-{end} => Active clusters: {active_clusters}, Top 5 weights: {np.round(dpgmm.weights_[:5], 3)}")

        all_results.append({
            'start': start,
            'end': end,
            'labels': labels,
            'probs': max_probs
        })

    # === Final window ===
    final_start = start + step_size
    if final_start < len(df):
        final_data = normalized_pca_components[final_start:]
        memory_data = normalized_pca_components[:]

        dpgmm_final = BayesianGaussianMixture(
            n_components=max_components,
            covariance_type='full',
            weight_concentration_prior_type='dirichlet_process',
            weight_concentration_prior=prior,
            max_iter=1000,
            tol=1e-3,
            init_params='kmeans',
            random_state=42
        )
        dpgmm_final.fit(memory_data)

        final_labels = dpgmm_final.predict(final_data)
        if merge_within_window:
            final_labels = merge_clusters_by_divergence(dpgmm_final, final_labels, merge_threshold)

        final_probs = dpgmm_final.predict_proba(final_data)
        final_max_probs = final_probs[np.arange(len(final_labels)), final_labels]

        all_labels[final_start:] = final_labels
        all_probs[final_start:] = final_max_probs

        print(f"Final window {final_start}-{len(df)} => Active clusters: {np.sum(dpgmm_final.weights_ > 0.01)}")

        all_results.append({
            'start': final_start,
            'end': len(df),
            'labels': final_labels,
            'probs': final_max_probs
        })
    else:
        dpgmm_final = dpgmm  # fallback

    # === Merge clusters outside the window loop === (if enabled)
    if not merge_within_window:
            # === Merge Clusters using Jeffrey's Divergence ===
        unique_labels = np.unique(all_labels)
        if -1 in unique_labels:
            unique_labels = unique_labels[unique_labels != -1]

        if len(unique_labels) > 1:
            means = dpgmm_final.means_
            covariances = dpgmm_final.covariances_
            label_indices = [label for label in unique_labels if np.sum(all_labels == label) > 0]

            distance_matrix = np.zeros((len(label_indices), len(label_indices)))
            for i, idx_i in enumerate(label_indices):
                for j, idx_j in enumerate(label_indices):
                    if i < j:
                        dist = jeffreys_divergence(means[idx_i], covariances[idx_i], means[idx_j], covariances[idx_j])
                        distance_matrix[i, j] = distance_matrix[j, i] = dist

        sns.heatmap(distance_matrix, cmap='viridis')
        plt.title("Jeffrey's Divergence Between Cluster Gaussians")
        plt.show()

        Z = linkage(distance_matrix, method='average')
        new_cluster_ids = fcluster(Z, t=merge_threshold, criterion='distance')
        label_map = {old: new for old, new in zip(label_indices, new_cluster_ids)}
        all_labels = np.array([label_map.get(label, -1) for label in all_labels])


    # === Return DataFrame with results ===
    df_result = df.copy()
    df_result.insert(1, 'Cluster', all_labels)
    df_result.insert(2, 'Assigned_Cluster_Prob', all_probs)

    used_clusters = np.unique(all_labels)
    used_clusters = used_clusters[used_clusters != -1]
    palette = sns.color_palette('viridis', len(used_clusters))
    cluster_color_map = {label: palette[i] for i, label in enumerate(used_clusters)}
    cluster_color_map[-1] = (0.6, 0.6, 0.6)

    # === Plot clusters with counts in legend ===
    plt.figure(figsize=(10, 6))
    ax = sns.scatterplot(
        x=normalized_pca_components[:, 0],
        y=normalized_pca_components[:, 1],
        hue=all_labels,
        palette=cluster_color_map,
        s=60,
        alpha=0.7,
        legend='full'
    )

    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.title("Streaming PCA + DPGMM Clustering")

    handles, labels = ax.get_legend_handles_labels()
    label_counts = pd.Series(all_labels).value_counts().sort_index()

    new_labels = []
    for lbl in labels:
        try:
            cluster_id = int(lbl)
            count = label_counts.get(cluster_id, 0)
            new_labels.append(f"Cluster {cluster_id} ({count} samples)")
        except ValueError:
            new_labels.append(lbl)

    ax.legend(handles=handles, labels=new_labels, title='Cluster', loc='upper right')
    plt.show()

    return df_result, all_results