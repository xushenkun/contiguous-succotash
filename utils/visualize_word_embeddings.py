import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

from utils.batch_loader import BatchLoader

if __name__ == "__main__":

    prefix = 'poem'
    word_is_char = True 

    batch_loader = BatchLoader('../../', prefix, word_is_char)

    if not os.path.exists('../../data/' + batch_loader.prefix + 'word_embeddings.npy'):
        raise FileNotFoundError("word embeddings file was't found")

    pca = PCA(n_components=2)
    word_embeddings = np.load('../../data/' + batch_loader.prefix + 'word_embeddings.npy')
    word_embeddings_pca = pca.fit_transform(word_embeddings)
       
    words = batch_loader.idx_to_word

    fig, ax = plt.subplots()
    fig.set_size_inches(150, 150)
    x = word_embeddings_pca[:, 0]
    y = word_embeddings_pca[:, 1]
    ax.scatter(x, y)

    for i, word in enumerate(words):
        ax.annotate(word, (x[i], y[i]))

    fig.savefig(batch_loader.prefix+'word_embedding.png', dpi=100)
