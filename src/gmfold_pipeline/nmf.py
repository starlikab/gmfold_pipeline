from sklearn.decomposition import NMF

def run_nmf( X, n_components ):
    init = "nndsvda"
    nmf= NMF(
    n_components=n_components,
    random_state=42,
    init=init,
    beta_loss="frobenius",
    alpha_W=0.00005,
    alpha_H=0.00005,
    l1_ratio=0.99,
    ).fit(X)
    # Get the topic distribution for each point in X
    topic_distribution = nmf.transform(X)
    W = nmf.components_
    # Create a label list where each element is the topic associated with the corresponding point
    labels = [topic_probs.argmax() for topic_probs in topic_distribution]
    # plot_top_words(nmf, np.asarray(words)[mask], n_top_words, "Topics in LDA model")
    return labels, topic_distribution,  W