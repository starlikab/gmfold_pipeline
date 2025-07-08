from sklearn.decomposition import NMF
import numpy as np
import matplotlib.pyplot as plt

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

def plot_nmf_reconstruction_error(X, topic_range):
    raw_errors = []
    rel_errors = []

    total_norm = np.linalg.norm(X, 'fro')  # Compute norm of the original data for normalization

    for k in topic_range:
        nmf = NMF(
            n_components=k,
            random_state=42,
            beta_loss="frobenius",
            alpha_W=0.00005,
            alpha_H=0.00005,
            l1_ratio=0.99,
        )
        W = nmf.fit_transform(X)
        H = nmf.components_
        error = nmf.reconstruction_err_
        rel_error = error / total_norm

        raw_errors.append(error)
        rel_errors.append(rel_error)

    # Plot both absolute and relative errors
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(topic_range, raw_errors, marker='o')
    plt.title("Raw NMF Reconstruction Error")
    plt.xlabel("Number of Topics")
    plt.ylabel("Frobenius Norm Error")

    plt.subplot(1, 2, 2)
    plt.plot(topic_range, rel_errors, marker='o', color='green')
    plt.title("Relative NMF Error (Normalized)")
    plt.xlabel("Number of Topics")
    plt.ylabel("Relative Error (0–1)")

    plt.tight_layout()
    plt.show()