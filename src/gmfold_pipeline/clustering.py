from sklearn.cluster import SpectralClustering
import numpy as np
import pandas as pd
import hdbscan
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

def spect_clustering(X, n_clusters):
    # Run clustering
    # Start measuring time
    # Run spectral clustering algorithm
    spectral_clustering = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors', random_state=42)

    # Fit the model and predict cluster labels
    cluster_labels = spectral_clustering.fit_predict(X)
    return cluster_labels

def display_results(results): 
  df_results = pd.DataFrame(
      results,
      columns=["min_size", "eps", "min_sample", "n_clusters", "noise", "sil_score", "ch_score", "db_score"]
  )
  print(df_results.to_string(index=False))

def hdbscan_scoring(topic):
  results = []
  for min_size in [3,5,8,10,15]:
    for epsilon in [0.0, 0.1, 0.2]:
      for min_sample in [1, 2, 3, None]:
        clusterer = hdbscan.HDBSCAN(min_samples=min_sample, min_cluster_size=min_size, cluster_selection_epsilon = epsilon, metric='euclidean')  # adjust min_cluster_size as needed
        labels = clusterer.fit_predict(topic)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        # Compute silhouette score (excluding noise points)
        if n_clusters > 1 and len(set(labels)) > 1:
            mask = labels != -1
            sil_score = silhouette_score(topic[mask], labels[mask])
            ch_score = calinski_harabasz_score(topic[mask], labels[mask])
            db_score = davies_bouldin_score(topic[mask], labels[mask])
        else:
            sil_score = float('nan')  # Not enough clusters to compute score
        results.append((min_size, epsilon, min_sample, n_clusters, n_noise, sil_score, ch_score, db_score))
  display_results(results)
  return

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