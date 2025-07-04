from sklearn.cluster import SpectralClustering
import numpy as np

def spect_clustering(X, n_clusters):
    # Run clustering
    # Start measuring time
    # Run spectral clustering algorithm
    spectral_clustering = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors', random_state=42)

    # Fit the model and predict cluster labels
    cluster_labels = spectral_clustering.fit_predict(X)
    return cluster_labels

def overlaps(new_cluster_labels, spectral_cluster_labels, df):

  # Identify the cluster label for aptamer 0
  for i in range(6):
    cluster_apt = new_cluster_labels[i]

    # Get indices of all aptamers in the same cluster
    same_cluster_indices = [k for k, label in enumerate(new_cluster_labels) if label == cluster_apt]
    spectral_cluster_overlap = [j for j, label in enumerate(spectral_cluster_labels) if label == spectral_cluster_labels[i] and j in same_cluster_indices]

    print("There are " + str(len(spectral_cluster_overlap)) + " in total out of " + str(len(same_cluster_indices)) + " clustered with aptamer " + str(i) + " here and by GMFold")

    # Print sequences of those aptamers
    isomorphic_overlap = []
    dif_overlap = []
    for idx in spectral_cluster_overlap:
        seq = df.loc[idx, 'Sequence']
        moztkin = df.loc[idx, 'd_b']
        bof = df.loc[idx, 'energy_faces']
        if np.array_equal(moztkin, df.loc[i, 'd_b']) and np.array_equal(bof, df.loc[i, 'energy_faces']):
          isomorphic_overlap.append(idx)
        else:
          dif_overlap.append(idx)
    
    non_overlap = []
    for j in same_cluster_indices:
      if j not in spectral_cluster_overlap:
        non_overlap.append(j)

    print(f"Aptamer {i}: {df.loc[i, 'Sequence']}")
    print(f"Isomorphic sequences among overlapping: {isomorphic_overlap}")
    print(f"Non-isomorphic sequences among overlapping: {dif_overlap}")
    print(f"Non-overlapping: {non_overlap} \n")