import hdbscan

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
  return results
