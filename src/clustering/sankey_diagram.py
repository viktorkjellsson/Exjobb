import pandas as pd
from collections import defaultdict
import plotly.graph_objects as go
import matplotlib.pyplot as plt

timestamp_to_x = {}

def build_sankey_links_from_cluster_dict(cluster_dict):
    links = defaultdict(int)
    
    timesteps = sorted(cluster_dict.keys())

    for t in range(len(timesteps) - 1):
        current_clusters = cluster_dict[timesteps[t]]
        next_clusters = cluster_dict[timesteps[t + 1]]

        for i in range(min(len(current_clusters), len(next_clusters))):
            src = f"{timesteps[t].strftime('%Y-%m-%d')}_Cluster: {current_clusters[i]}"
            tgt = f"{timesteps[t + 1].strftime('%Y-%m-%d')}_Cluster: {next_clusters[i]}"
            links[(src, tgt)] += 1  # Or use weights if needed

    return links

def prepare_sankey_data(links):
    all_nodes = set()
    for src, tgt in links:
        all_nodes.add(src)
        all_nodes.add(tgt)

    all_nodes = sorted(all_nodes)
    node_indices = {node: idx for idx, node in enumerate(all_nodes)}

    source = [node_indices[src] for (src, _) in links]
    target = [node_indices[tgt] for (_, tgt) in links]
    value = [links[(src, tgt)] for (src, tgt) in links]

    return all_nodes, source, target, value

def assign_node_positions(nodes):
    timestamp_to_nodes = defaultdict(list)

    for node in nodes:
        t_str = node.split('_')[0]
        t_dt = pd.to_datetime(t_str)
        timestamp_to_nodes[t_dt].append(node)

    x = []
    y = []

    # Get a sorted list of unique cluster labels
    unique_clusters = sorted(set(node.split('_')[1] for node in nodes))

    # Dynamically calculate y-spacing based on the number of unique clusters
    y_spacing = 1.0 / len(unique_clusters)  # Dynamic spacing based on cluster count
    cluster_to_y = {cluster: i * y_spacing for i, cluster in enumerate(unique_clusters)}

    for node in nodes:
        t_str = node.split('_')[0]
        t_dt = pd.to_datetime(t_str)
        cluster_label = node.split('_')[1]  # Extract cluster label
        
        cluster_y = cluster_to_y[cluster_label]

        # Map the timestamp to a normalized x position
        x.append(timestamp_to_x[t_dt])
        y.append(cluster_y)

    return x, y


# 4. Plot the Sankey diagram with axis labels
def plot_sankey(nodes, source, target, value, title, save_path, save):
    global timestamp_to_x

    # Parse datetime objects from node names
    timestamps = sorted(set(pd.to_datetime(n.split('_')[0]) for n in nodes))
    timestamp_to_x = {t: i / (len(timestamps) - 1) for i, t in enumerate(timestamps)}

    # Get positions for nodes (x, y)
    x_positions, y_positions = assign_node_positions(nodes)

    # Extract cluster labels for axis annotations
    cluster_labels = [node.split('_')[1] for node in nodes]

    # Create axis labels for x and y
    datetime_labels = [t.strftime("%Y-%m-%d") for t in timestamps]
    cluster_labels_sorted = sorted(set(cluster_labels))  # Y-axis labels

    # Extract cluster labels
    cluster_labels_raw = [node.split('_')[1] for node in nodes]
    unique_clusters = sorted(set(cluster_labels_raw))

    tab10_colors = plt.get_cmap('tab10').colors
    color_palette = ['rgb({},{},{})'.format(int(r*255), int(g*255), int(b*255)) for r, g, b in tab10_colors]
    cluster_to_color = {cluster: color_palette[i % len(color_palette)] for i, cluster in enumerate(unique_clusters)}
    node_colors = [cluster_to_color[cluster] for cluster in cluster_labels_raw]

    fig = go.Figure(data=[go.Sankey(
        arrangement="snap",  # Ensures x/y positions are respected
        node=dict(
            pad=15,
            thickness=10,
            label=["" for _ in cluster_labels_raw],  # Hide labels
            color=node_colors,
            x=[timestamp_to_x[pd.to_datetime(n.split('_')[0])] for n in nodes],
            y=y_positions,
        ),
        link=dict(
            source=source,
            target=target,
            value=value,
        )
    )])

    # Add timestamp labels along the very bottom of the full figure using 'paper' coords
    for i, timestamp in enumerate(datetime_labels):
        fig.add_annotation(
            x=i / (len(datetime_labels) - 1),  # Normalized position
            y=-0.3,  # Bottom of the figure canvas
            xref='paper',
            yref='paper',
            text=timestamp,
            showarrow=False,
            font=dict(size=12),
            xanchor="center",
            textangle=90
        )

    # Dummy scatter traces for legend only
    legend_scatter = []
    for cluster_label, color in cluster_to_color.items():
        legend_scatter.append(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(color=color, size=10),
            name=cluster_label,
            showlegend=True
        ))

    fig.add_traces(legend_scatter)

    # Update layout with enough bottom margin to fit timestamp labels
    fig.update_layout(
        title_text=title,
        font_size=12,
        width=1000,
        height=650,
        margin=dict(b=180, t=50, l=100, r=150),
        showlegend=True,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        plot_bgcolor='white'
    )

    fig.show()

    if save == True:
        fig.write_image(save_path, format="pdf")
    else:
        print("Sankey diagram not saved. Set 'save' to True to save the figure.")