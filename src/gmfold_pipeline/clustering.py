from sklearn.cluster import SpectralClustering

def spect_clustering(X, n_clusters):
    # Run clustering
    # Start measuring time
    # Run spectral clustering algorithm
    spectral_clustering = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors', random_state=42)

    # Fit the model and predict cluster labels
    cluster_labels = spectral_clustering.fit_predict(X)
    return cluster_labels