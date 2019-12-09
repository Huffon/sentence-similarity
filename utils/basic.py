import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, \
    manhattan_distances


def angular_distance(src, tgt):
    cos_sim = cosine_similarity(src, tgt)
    np.fill_diagonal(cos_sim, 1)
    distance_ = 1 - (np.arccos(cos_sim) / math.pi)
    return distance_


def cosine_sim(src, tgt):
    similarity_ = cosine_similarity(src, tgt)
    return similarity_


def manhattan_dist(src, tgt):
    distance_ = manhattan_distances(src, tgt)
    return distance_


def euclidean_dist(src, tgt):
    distance_ = euclidean_distances(src, tgt)
    return distance_


def inner_product(src, tgt):
    similarity_ = np.inner(src, tgt)
    return similarity_


def vector_summation(sentences):
    sent_len = sentences.shape[1]
    summed_sentence_ = sentences.sum(axis=1) / sent_len
    return summed_sentence_


def plot_similarity(sentences, similarity, method):
    max_sim = np.max(similarity)
    min_sim = np.min(similarity)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)

    ax.matshow(similarity, 
               vmin=min_sim, 
               vmax=max_sim, 
               interpolation='nearest', 
               cmap='Greens')

    for (i, j), z in np.ndenumerate(similarity):
        ax.text(j, i, '{:0.2f}'.format(z), ha='center', va='center', fontsize=12)

    ax.tick_params(labelsize=15)
    ax.set_xticklabels(['']+sentences)
    ax.set_yticklabels(['']+sentences)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.set_title(f'Sentence Similarity using {method.upper()}')

    plt.show()
    plt.close()
