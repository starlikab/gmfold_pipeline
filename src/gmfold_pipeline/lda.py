# Function to run LDA
from sklearn.decomposition import LatentDirichletAllocation

def run_lda( X, n_components ):
    learning_method = 'online'
    lda = LatentDirichletAllocation(
        n_components=n_components,
        learning_decay=0.7,
        learning_offset=10.0,
        max_iter=10,
        batch_size=128,
        evaluate_every=-1,
        random_state=42
    ).fit(X)

    Theta = lda.transform(X)
    Phi = lda.components_
    labels = [topic_probs.argmax() for topic_probs in Theta]
    return labels, Theta,  Phi,  n_components
