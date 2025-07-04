import matplotlib.pyplot as plt
import numpy as np
import string

def plot_2d_scatter(X_embedded, cluster_labels):

    # Prepare color map for clusters (excluding noise)
    unique_labels = np.unique(cluster_labels)
    non_noise_labels = [label for label in unique_labels if label != -1]
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))

    plt.figure(figsize=(7, 6))

    # Scatter plot by cluster label
    for idx, label in enumerate(unique_labels):
        indices = np.where(cluster_labels == label)
        label_str = f"Cluster {label}" if label != -1 else "Noise"

        # Set light grey color for noise
        color = colors[idx] if label != -1 else '#bcbcbc'

        plt.scatter(X_embedded[indices, 0], X_embedded[indices, 1],
                    s=50, alpha=0.7, color=color, label=label_str)

    # Annotate aptamers 0 to 4
    highlight_aptamer_indices = [0, 1, 2, 3, 4]
    for i in highlight_aptamer_indices:
        x, y = X_embedded[i]
        plt.annotate(str(i), (x, y), fontsize=12, weight='bold', color='black',
                     textcoords="offset points", xytext=(0, 5), ha='center')

    # Final plot formatting
    # plt.title("Clustered Aptamers (HDBSCAN or Other Clustering)", fontsize=14)
    plt.tight_layout()
    plt.grid(True)
    plt.axis('off')
    plt.show()

    
def plot_cluster_counts(cluster_labels):
    # Sort all labels in increasing order
    unique_labels = sorted(set(cluster_labels))

    # Find the clusters to which the first five most common aptamers belong
    # Note: df -> topic -> cluster_labels -> high_light clusters retains the fact that the first aptamers are the top counts
    highlight_indices = range(5)
    highlight_clusters = set(cluster_labels[i] for i in highlight_indices)

    # Count occurrences of aptamers in each cluster
    counts = {label: np.sum(cluster_labels == label) for label in unique_labels}
    sizes = [counts[label] for label in unique_labels]

    # Create bar plot
    plt.figure(figsize=(10, 6))
    spacing_multiplier = 1.5
    x_positions = np.arange(len(counts)) * spacing_multiplier
    bars = plt.bar(x_positions, sizes)

    # Color highlighted bars
    count = 0
    for i, bar in enumerate(bars):
        height = bar.get_height()
        cluster_label = unique_labels[i]

        # Signify which clusters are home to top counts
        colors = ["#3d010e", "#79021c", "#b6042a", "#f50538"]
        if cluster_label in highlight_clusters:
            bar.set_facecolor(colors[count])
            count += 1

        # Display number of aptamers in each cluster above bar
        plt.text(bar.get_x() + bar.get_width() / 2.0, height, f'{height:.0f}',
                ha='center', va='bottom', fontsize=15)

    plt.yscale('log')
    plt.yticks([])
    # Add one to every label so we don't begion at 0
    temp_labels = [f'C$_{{{i+1}}}$' for i in unique_labels]
    # Change the font size
    plt.xticks(x_positions, temp_labels, fontsize=15)
    plt.xlabel('Clusters', fontsize=20)
    plt.ylabel('Number of sequences', fontsize=30)
    plt.tight_layout()
    plt.show()
