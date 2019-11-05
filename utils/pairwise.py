import numpy as np
from utils.basic import cosine_similarity


def pairwise_cosine_similarity(src, tgt):
    sim_matrix = np.zeros([len(src), len(tgt)])

    for i in range(len(src)):
        for j in range(len(tgt)):
            sim_matrix[i][j] = cosine_similarity(src[i], tgt[j])

    max_sim = np.amax(sim_matrix, axis=1)
    print(max_sim.shape)

    return max_sim


def pairwise_cosine_similarity_idf(src, tgt):
    sim_matrix = np.zeros([len(src), len(tgt)])

    for i in range(len(src)):
        for j in range(len(tgt)):
            sim_matrix[i][j] = cosine_similarity(src[i], tgt[j])

    max_sim = np.amax(sim_matrix, axis=1)

    if use_idf:
        pass

    return max_sim


def bert_pairwise_cosine_similarity(src, tgt):
    p, _, _ = score([src], [tgt], lang='en')

    return p


def bert_pairwise_cosine_similarity_idf(src, tgt):
    p, _, _ = score([src], [tgt], lang='en')

    return p
