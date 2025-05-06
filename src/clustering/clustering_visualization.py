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
import matplotlib.cm as cm
import matplotlib.colors as mcolors


def plot_clusters_over_time(data_with_clusters, method) -> None:
    """
    Plot the assignment to clusters over time, only displaying active clusters.
    
    Args:
        data_with_clusters (pd.DataFrame): DataFrame with 'Timestamp' and 'Cluster' columns.
        method (str): Name of the clustering method for the plot title.
    """
    # Convert timestamps
    data_with_clusters['Timestamp'] = pd.to_datetime(data_with_clusters['Timestamp'])

    # Identify active clusters (those actually used)
    active_clusters = sorted(data_with_clusters['Cluster'].dropna().unique())

    # Create Plotly figure
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=data_with_clusters['Timestamp'], 
        y=data_with_clusters['Cluster'], 
        mode='markers+lines',
        marker=dict(
            size=6, 
            color=data_with_clusters['Cluster'], 
            colorscale='Viridis',
            colorbar=dict(title='Cluster')
        ),
        line=dict(width=0.5, color='gray')
    ))

    # Update layout
    fig.update_layout(
        title=f'Assignment to Clusters Over Time with {method} Clustering',
        xaxis_title='Time',
        yaxis_title='Cluster',
        xaxis_tickangle=-45,
        yaxis=dict(
            tickmode='array',
            tickvals=active_clusters,
            ticktext=[str(c) for c in active_clusters]
        )
    )

    fig.show()

def plot_cluster_mean_and_std(data_with_clusters, clusters_to_keep, method) -> None:
    """
    Plot the mean strain values for each cluster with uncertainty (standard deviation).

    Args:
        data_with_clusters (pd.DataFrame): DataFrame with 'Timestamp' and 'Cluster' columns,
                                           other columns should be numeric sensor data (e.g., distances).
        clusters_to_keep (list): List of cluster labels to keep (e.g., [0, 1, 2]) or ['all'] to keep all.
        method (str): Name of clustering method for plot title.
    """
    # Get unique clusters and normalize types
    cluster_col = data_with_clusters['Cluster'].dropna()
    all_clusters = sorted(cluster_col.unique())  # Use all unique cluster labels for consistent color mapping

    # Normalize user input
    if clusters_to_keep == ['all']:
        clusters_to_keep = [str(c) for c in all_clusters]

    else:
        clusters_to_keep = [str(c) for c in clusters_to_keep]

    # Drop timestamp for clustering-related stats
    df_mean = data_with_clusters.drop(columns='Timestamp').groupby('Cluster').mean()
    df_std = data_with_clusters.drop(columns='Timestamp').groupby('Cluster').std()

    # Filter to desired clusters
    df_mean = df_mean.loc[df_mean.index.astype(str).isin(clusters_to_keep)]
    df_std = df_std.loc[df_std.index.astype(str).isin(clusters_to_keep)]

    x_values = pd.to_numeric(df_mean.columns, errors='coerce')
    average_std = df_std.mean(axis=1)

    # Setup Viridis colormap
    cmap = cm.get_cmap('viridis')
    norm = mcolors.Normalize(vmin=min(all_clusters), vmax=max(all_clusters))  # Normalize using all clusters

    plt.figure(figsize=(30, 6))

    for cluster in df_mean.index:
        cluster_num = int(cluster)  # original numeric cluster label
        color = cmap(norm(cluster_num))

        y_mean = df_mean.loc[cluster]
        y_std = df_std.loc[cluster]

        plt.plot(x_values, y_mean,
                 label=f'Cluster {cluster} - Mean, Avg. std: {average_std[cluster]:.2f}',
                 linewidth=2, color=color)

        plt.fill_between(x_values,
                         y_mean - y_std,
                         y_mean + y_std,
                         alpha=0.2, color=color)

    plt.xlabel('Distance [m]')
    plt.ylabel('Strain (Mean Value)')
    plt.title(f'{method} Clustering Centroids with Uncertainty (Standard Deviation)')
    plt.legend(title='Cluster')
    plt.grid(True)
    plt.show()







    # # Plot
    # plt.figure(figsize=(30, 6))
    # for i, cluster in enumerate(df_mean.index):
    #     plt.plot(x_values, df_mean.loc[cluster], label=f'Cluster {cluster} - Mean, Avg. std: {average_std[cluster]:.2f}', linewidth=2)
    #     plt.fill_between(x_values,
    #                      df_mean.loc[cluster] - df_std.loc[cluster],
    #                      df_mean.loc[cluster] + df_std.loc[cluster],
    #                      alpha=0.3)

    # plt.xlabel('Distance [m]')
    # plt.ylabel('Strain (Mean Value)')
    # plt.title(f'{method} Clustering Centroids with Uncertainty (Standard Deviation)')
    # plt.legend(title='Cluster')
    # plt.grid(True)
    # plt.show()








# def plot_cluster_mean_and_std(data_with_clusters, clusters_to_keep, method) -> None:
#     """
#     Plot the mean strain values for each cluster with uncertainty (standard deviation) using Plotly.

#     Args:
#         data_with_clusters (pd.DataFrame): DataFrame with 'Timestamp' and 'Cluster' columns,
#                                            other columns should be numeric sensor data (e.g., distances).
#         clusters_to_keep (list): List of cluster labels to keep (e.g., [0, 1, 2]) or ['all'] to keep all.
#         method (str): Name of clustering method for plot title.
#     """
#     cluster_col = data_with_clusters['Cluster'].dropna()
#     unique_clusters = cluster_col.astype(str).unique()

#     if clusters_to_keep == ['all']:
#         clusters_to_keep = unique_clusters
#     else:
#         clusters_to_keep = [str(c) for c in clusters_to_keep]

#     # Compute mean and std by cluster
#     df_mean = data_with_clusters.drop(columns='Timestamp').groupby('Cluster').mean()
#     df_std = data_with_clusters.drop(columns='Timestamp').groupby('Cluster').std()

#     # Filter clusters
#     df_mean = df_mean.loc[df_mean.index.astype(str).isin(clusters_to_keep)]
#     df_std = df_std.loc[df_std.index.astype(str).isin(clusters_to_keep)]

#     x_values = pd.to_numeric(df_mean.columns, errors='coerce')
#     average_std = df_std.mean(axis=1)

#     # Assign colors from the Viridis colormap
#     viridis_colors = plotly.colors.sequential.Viridis
#     num_clusters = len(df_mean)
#     color_scale = [viridis_colors[int(i * (len(viridis_colors) - 1) / (num_clusters - 1))] for i in range(num_clusters)]

#     fig = go.Figure()

#     for i, cluster in enumerate(df_mean.index):
#         y_mean = df_mean.loc[cluster].values
#         y_std = df_std.loc[cluster].values
#         color = color_scale[i]

#         # 1. Uncertainty band (plotted first so it's behind the line)
#         fig.add_trace(go.Scatter(
#             x=np.concatenate([x_values, x_values[::-1]]),
#             y=np.concatenate([y_mean + y_std, (y_mean - y_std)[::-1]]),
#             fill='toself',
#             fillcolor=color.replace('rgb', 'rgba').replace(')', ', 0.05)'),
#             line=dict(color='rgba(255,255,255,0)'),
#             hoverinfo='skip',
#             showlegend=False
#         ))

#         # 2. Mean line
#         fig.add_trace(go.Scatter(
#             x=x_values,
#             y=y_mean,
#             mode='lines',

#             line=dict(color=color, width=1.5),
#             name=f'Cluster {cluster} - Mean, Avg. std: {average_std[cluster]:.2f}'
#         ))

#     fig.update_layout(
#         title=f'{method} Clustering Centroids with Uncertainty (Standard Deviation)',
#         xaxis_title='Distance [m]',
#         yaxis_title='Strain (Mean Value)',
#         legend_title='Cluster',
#         template='plotly_white'
#     )

#     fig.show()