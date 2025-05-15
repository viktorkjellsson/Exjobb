from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import sys
import plotly.graph_objects as go
import seaborn as sns

# Add the root project directory to the Python path
project_root = Path.cwd().parent  # This will get the project root since the notebook is in 'notebooks/'
sys.path.append(str(project_root))

def plot_clusters_over_time(data_with_clusters, cluster_color_map, method, beam_id, save_dir, save) -> None:
    """
    Plot the assignment to clusters over time, only displaying active clusters.
    
    Args:
        data_with_clusters (pd.DataFrame): DataFrame with 'Timestamp' and 'Cluster' columns.
        method (str): Name of the clustering method for the plot title.
    """
    # Convert timestamps
    data_with_clusters['Timestamp'] = pd.to_datetime(data_with_clusters['Timestamp'])

    # Sort data by timestamp to ensure chronological order
    data_with_clusters = data_with_clusters.sort_values(by='Timestamp')

    # Identify active clusters (those actually used)
    active_clusters = sorted(data_with_clusters['Cluster'].dropna().unique())

    # Create Plotly figure
    fig = go.Figure()

    # Get unique timestamps
    unique_timestamps = data_with_clusters['Timestamp'].unique()

    # Initialize lists to store data for lines
    line_x = []
    line_y = []

    # Add one trace per timestamp using specified colors
    for timestamp in unique_timestamps:
        timestamp_data = data_with_clusters[data_with_clusters['Timestamp'] == timestamp]
        
        for cluster in active_clusters:
            cluster_data = timestamp_data[timestamp_data['Cluster'] == cluster]
            if not cluster_data.empty:
                color = cluster_color_map.get(cluster, 'gray')

                # Convert RGB tuple (0-1) to hex if needed
                if isinstance(color, tuple):
                    color = f'rgba({int(color[0]*255)}, {int(color[1]*255)}, {int(color[2]*255)}, 1)'

                fig.add_trace(go.Scatter(
                    x=cluster_data['Timestamp'],
                    y=[cluster] * len(cluster_data),
                    mode='markers',
                    marker=dict(size=6, color=color),
                    opacity=1.0,
                    showlegend=False  # Hide legend for individual markers
                ))

                # Append data for lines
                line_x.extend(cluster_data['Timestamp'].tolist())
                line_y.extend([cluster] * len(cluster_data))

    # Add lines connecting consecutive markers (chronologically ordered)
    fig.add_trace(go.Scatter(
        x=line_x,
        y=line_y,
        mode='lines',
        line=dict(color='black', width=0.5),
        opacity=0.5,
        showlegend=False,  # Hide legend for lines
    ))

    # Add separate traces for legend entries
    for cluster in active_clusters:
        color = cluster_color_map.get(cluster, 'gray')

        # Convert RGB tuple (0-1) to hex if needed
        if isinstance(color, tuple):
            color = f'rgba({int(color[0]*255)}, {int(color[1]*255)}, {int(color[2]*255)}, 1)'

        fig.add_trace(go.Scatter(
            x=[None],  # Dummy data for legend
            y=[None],  # Dummy data for legend
            mode='markers',
            name=f'Cluster {cluster}',
            marker=dict(size=6, color=color),
            showlegend=True
        ))

    # Update layout
    fig.update_layout(
        title=f'Assignment to Clusters Over Time with {method} Clustering for Beam {beam_id}',
        xaxis_title='Time',
        yaxis_title='Cluster',
        font=dict(size=20),
        yaxis=dict(
            tickmode='array',
            tickvals=active_clusters,
            ticktext=[str(c) for c in active_clusters]
        ),
            margin=dict(b=120),  # Add bottom margin in pixels
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=-0.5,
                xanchor='center',
                x=0.5
            )
        )

    if save == True:
        fig.write_image(str(save_dir), format='pdf', width=1500, height=500, scale=1)
        fig.show()
    else:
        fig.show()


def plot_cluster_mean_and_std(data_with_clusters, clusters_to_keep, cluster_color_map, method, beam, beam_id, save_dir=None, save=False):
    """
    Plotly version of cluster mean + standard deviation bands.

    Args:
        data_with_clusters (pd.DataFrame): Must contain 'Timestamp', 'Cluster' + sensor columns (as distances).
        clusters_to_keep (list): List of clusters to plot, or ['all'].
        cluster_color_map (dict): Maps int cluster_id to RGB tuples.
        method (str): Title method (e.g., 'DPGMM').
        save_dir (str): Where to save (only if save=True).
        save (bool): Whether to export as .pdf.
    """
    # Normalize clusters
    cluster_col = data_with_clusters['Cluster'].dropna()
    all_clusters = sorted(cluster_col.unique())

    if clusters_to_keep == ['all']:
        clusters_to_keep = [str(c) for c in all_clusters]
    else:
        clusters_to_keep = [str(c) for c in clusters_to_keep]

    # Compute stats
    df_mean = data_with_clusters.drop(columns='Timestamp').groupby('Cluster').mean()
    df_std = data_with_clusters.drop(columns='Timestamp').groupby('Cluster').std()

    df_mean = df_mean.loc[df_mean.index.astype(str).isin(clusters_to_keep)]
    df_std = df_std.loc[df_std.index.astype(str).isin(clusters_to_keep)]

    x_values = pd.to_numeric(df_mean.columns, errors='coerce')
    average_std = df_std.mean(axis=1)

    # Create Plotly figure
    fig = go.Figure()

    for cluster in df_mean.index:
        cluster_num = int(cluster)
        color_rgb = cluster_color_map.get(cluster_num, (0.6, 0.6, 0.6))
        color_hex = f'rgb({int(color_rgb[0]*255)}, {int(color_rgb[1]*255)}, {int(color_rgb[2]*255)})'

        y_mean = df_mean.loc[cluster]
        y_std = df_std.loc[cluster]

        # Std band (transparent fill)
        fig.add_trace(go.Scatter(
            x=x_values,
            y=y_mean + y_std,
            mode='lines',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip'
        ))

        fig.add_trace(go.Scatter(
            x=x_values,
            y=y_mean - y_std,
            mode='lines',
            line=dict(width=0),
            fill='tonexty',
            fillcolor=color_hex.replace('rgb', 'rgba').replace(')', ',0.2)'),
            name=f'Cluster {cluster} ± std',
            hoverinfo='skip',
            showlegend=False
        ))

        # Mean line
        fig.add_trace(go.Scatter(
            x=x_values,
            y=y_mean,
            mode='lines',
            line=dict(color=color_hex, width=2),
            name=f'Cluster {cluster} - Mean (avg std: {average_std[cluster]:.2f})'
        ))

    support_coords = list(beam.values())
    annotations = list(beam.keys())
    # Add vertical lines at chosen x-values
    for x_val, annotation in zip(support_coords, annotations):
        fig.add_shape(
            type="line",
            x0=x_val, x1=x_val,
            y0=min(df_mean.min()), y1=max(df_mean.max()),
            line=dict(color="grey", width=2, dash="dash")
        )
        
        fig.add_annotation(
            x=x_val,
            y=min(df_mean.min())*1.2,  # Slightly below plot
            text=annotation,
            showarrow=False,
            font=dict(size=20, color="black"),
            xanchor="center"
        )

        fig.update_layout(
            title=f'Mean Strain Distributions with Standard Deviation from {method} Clusters for Beam {beam_id}',
            xaxis_title='Distance [m]',
            yaxis_title='Strain (Mean Value)',
            legend_title='Cluster',
            font=dict(size=20),
            height=550,  # Optionally increase plot height
            width=1500,
            margin=dict(b=120),  # Add bottom margin in pixels
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=-0.5,
                xanchor='center',
                x=0.5
            )
        )

    if save:
        fig.write_image(str(save_dir), format='pdf', width=1500, height=500, scale=1)
        fig.show()
    else:
        fig.show()

def plot_dpgmm_clusters_simple(
    df,
    normalized_pca_components,
    all_labels,
    cluster_color_map, 
    save_dir,
    save, 
    beam_id
):
    # Add cluster labels to the DataFrame
    df['Cluster'] = all_labels

    fig, ax = plt.subplots(figsize=(7, 6))

    sns.scatterplot(
        x=normalized_pca_components[:, 0],
        y=normalized_pca_components[:, 1],
        hue=all_labels,
        palette=cluster_color_map,
        s=60,
        alpha=0.7,
        ax=ax,
        legend=False
    )
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_title(f"DPGMM Clustering on First Two PCA Components for Beam {beam_id}")

    # Create custom legend
    unique_labels = sorted(set(all_labels))
    handles = [
        plt.Line2D([0], [0], marker='o', color='w', label=f"Cluster {lbl} ({(df['Cluster'] == lbl).sum()} samples)",
                   markerfacecolor=cluster_color_map[lbl], markersize=8)
        for lbl in unique_labels
    ]
    ax.legend(handles=handles, title='Cluster', loc='best')

    plt.tight_layout()

    if save:
        plt.savefig(save_dir, format='pdf', bbox_inches='tight')
        plt.show()
    else:
        plt.show()


def plot_dpgmm_clusters_subplot(
    df,
    normalized_pca_components,
    all_labels,
    cluster_color_map, 
    num_components_to_plot,
    save_dir,
    save, 
    beam_id
):
    # Add cluster labels to the DataFrame
    df['Cluster'] = all_labels

    num_components_total = normalized_pca_components.shape[1]

    # Use only up to the specified number of PCA components
    num_components = min(num_components_total, num_components_to_plot)
    components = list(range(num_components))
    combinations = [(i, j) for i in components for j in components if i < j]

    # Setup subplot grid
    n_plots = len(combinations)
    if num_components == 2:
        n_cols = 1
    else:
        n_cols = 3
    n_rows = (n_plots + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))

    if isinstance(axes, plt.Axes):
        axes = [axes]
    else:
        axes = axes.flatten()

    for idx, (i, j) in enumerate(combinations):
        ax = axes[idx]
        sns.scatterplot(
            x=normalized_pca_components[:, i],
            y=normalized_pca_components[:, j],
            hue=all_labels,
            palette=cluster_color_map,
            s=60,
            alpha=0.7,
            ax=ax,
            legend=False
        )
        ax.set_xlabel(f"PC {i + 1}")
        ax.set_ylabel(f"PC {j + 1}")
        ax.set_title(f"Principal Components {i + 1} & {j + 1}")

    # Remove unused axes
    for idx in range(len(combinations), len(axes)):
        fig.delaxes(axes[idx])

    # Create legend from unique labels
    unique_labels = sorted(set(all_labels))
    handles = [
        plt.Line2D([0], [0], marker='o', color='w', label=f"Cluster {lbl} ({(df['Cluster'] == lbl).sum()} samples)",
                   markerfacecolor=cluster_color_map[lbl], markersize=8)
        for lbl in unique_labels
    ]

    fig.legend(
        handles=handles,
        title='Cluster',
        loc='upper right',
        bbox_to_anchor=(1.15, 1)
    )

    fig.suptitle(f"PCA + DPGMM Clustering with First {num_components} Components for Beam {beam_id}", fontsize=16, y=1.02)
    plt.tight_layout()

    if save==True:
        plt.savefig(save_dir, format='pdf', bbox_inches='tight')
        plt.show()
    else:
        plt.show()

def plot_dpgmm_clusters(
    df,
    normalized_pca_components,
    all_labels,
    cluster_color_map, 
    num_components_to_plot,
    save_dir,
    save, 
    beam_id
):
    
    if num_components_to_plot == 2:
        plot_dpgmm_clusters_simple(
            df,
            normalized_pca_components,
            all_labels,
            cluster_color_map,
            save_dir,
            save, 
            beam_id
        )
    else:
        plot_dpgmm_clusters_subplot(
            df,
            normalized_pca_components,
            all_labels,
            cluster_color_map,
            num_components_to_plot,
            save_dir,
            save, 
            beam_id
        )
        

# def plot_dpgmm_clusters(
#     df,
#     normalized_pca_components,
#     all_labels,
#     cluster_color_map,
#     num_components_to_plot
# ):
#     # Add cluster labels to DataFrame
#     df['Cluster'] = all_labels

#     num_components_total = normalized_pca_components.shape[1]
#     num_components = min(num_components_total, num_components_to_plot)
#     components = list(range(num_components))
#     combinations = [(i, j) for i in components for j in components if i < j]

#     # Setup subplot grid
#     n_plots = len(combinations)
#     n_cols = 3
#     n_rows = (n_plots + n_cols - 1) // n_cols

#     fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=[
#         f"Principal Components {i + 1} & {j + 1}" for i, j in combinations
#     ])

#     # Scatter plot for each PCA component combination
#     for idx, (i, j) in enumerate(combinations):
#         row, col = divmod(idx, n_cols)

#         for cluster_label in sorted(set(all_labels)):
#             # cluster_mask = (df['Cluster'] == cluster_label)
#             fig.add_trace(
#                 go.Scatter(
#                     x=normalized_pca_components[:, i],
#                     y=normalized_pca_components[:, j],
#                     mode="markers",
#                     marker=dict(
#                         color=cluster_color_map[cluster_label],  # Maintain same colors as sns
#                         size=6,
#                         opacity=0.7
#                     ),
#                     name=f"Cluster {cluster_label}",
#                     legendgroup=str(cluster_label),
#                     hoverinfo="text",
#                     text=[f"Sample {idx}, Cluster {cluster_label}" for idx in range(len(df))]
#                 ),
#                 row=row + 1,
#                 col=col + 1
#             )

#     # Layout adjustments
#     fig.update_layout(
#         title_text=f"PCA + DPGMM Clustering with first {num_components} components",
#         showlegend=True,
#         width=1000,
#         height=600
#     )

#     fig.show()


    

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

# def plot_cluster_mean_and_std(data_with_clusters, clusters_to_keep, cluster_color_map, method, save_dir, save) -> None:
#     """
#     Plot the mean strain values for each cluster with uncertainty (standard deviation).

#     Args:
#         data_with_clusters (pd.DataFrame): DataFrame with 'Timestamp' and 'Cluster' columns,
#                                            other columns should be numeric sensor data (e.g., distances).
#         clusters_to_keep (list): List of cluster labels to keep (e.g., [0, 1, 2]) or ['all'] to keep all.
#         method (str): Name of clustering method for plot title.
#     """
#     # Get unique clusters and normalize types
#     cluster_col = data_with_clusters['Cluster'].dropna()
#     all_clusters = sorted(cluster_col.unique())  # Use all unique cluster labels for consistent color mapping

#     # Normalize user input
#     if clusters_to_keep == ['all']:
#         clusters_to_keep = [str(c) for c in all_clusters]

#     else:
#         clusters_to_keep = [str(c) for c in clusters_to_keep]

#     # Drop timestamp for clustering-related stats
#     df_mean = data_with_clusters.drop(columns='Timestamp').groupby('Cluster').mean()
#     df_std = data_with_clusters.drop(columns='Timestamp').groupby('Cluster').std()

#     # Filter to desired clusters
#     df_mean = df_mean.loc[df_mean.index.astype(str).isin(clusters_to_keep)]
#     df_std = df_std.loc[df_std.index.astype(str).isin(clusters_to_keep)]

#     x_values = pd.to_numeric(df_mean.columns, errors='coerce')
#     average_std = df_std.mean(axis=1)

#     # Setup Viridis colormap
#     cmap = cm.get_cmap('tab10')
#     norm = mcolors.Normalize(vmin=min(all_clusters), vmax=max(all_clusters))  # Normalize using all clusters

#     plt.figure(figsize=(30, 6))

#     for cluster in df_mean.index:
#         cluster_num = int(cluster)  # original numeric cluster label
#         color = cluster_color_map.get(cluster_num, (0.6, 0.6, 0.6))  # Fallback to gray

#         if isinstance(color, tuple):
#             color = (color[0], color[1], color[2])

#         y_mean = df_mean.loc[cluster]
#         y_std = df_std.loc[cluster]

#         plt.plot(x_values, y_mean,
#                  label=f'Cluster {cluster} - Mean, Avg. std: {average_std[cluster]:.2f}',
#                  linewidth=2, color=color)

#         plt.fill_between(x_values,
#                          y_mean - y_std,
#                          y_mean + y_std,
#                          alpha=0.2, color=color)

#     plt.xlabel('Distance [m]')
#     plt.ylabel('Strain (Mean Value)')
#     plt.title(f'{method} Clustering Centroids with Uncertainty (Standard Deviation)')
#     plt.legend(title='Cluster')
#     plt.grid(True)

#     if save == True:
#         plt.savefig(save_dir, format='pdf', bbox_inches='tight')
#         plt.show()
#     else:
#         plt.show()