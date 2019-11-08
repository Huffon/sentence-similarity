import math
import numpy as np
import seaborn as sns
from numpy import arccos
from numpy.linalg import norm
import matplotlib.pyplot as plt
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

    sns.set(font_scale=0.8)
    graph = sns.heatmap(
            similarity,
            vmin=min_sim,
            vmax=max_sim,
            annot=True,
            square=True,
            fmt='1g',
            cmap="coolwarm",
            cbar=False)

    graph.set_xticklabels(sentences, rotation=0)
    graph.set_yticklabels(sentences, rotation=0)
    graph.set_title(f'Sentence Similarity using {method}')
    plt.show()
