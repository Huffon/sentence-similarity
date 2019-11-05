import numpy as np
from utils.basic import cosine_similarity


def pairwise_cos_sim(src, tgt):
    sim_matrix = np.zeros([len(src), len(tgt)])

    for i in range(len(src)):
        for j in range(len(tgt)):
            sim_matrix[i][j] = cosine_similarity(src[i], tgt[j])

    max_sim = np.amax(sim_matrix, axis=1)

    return max_sim


def pairwise_cos_sim_idf(src, tgt):
    sim_matrix = np.zeros([len(src), len(tgt)])

    for i in range(len(src)):
        for j in range(len(tgt)):
            sim_matrix[i][j] = cosine_similarity(src[i], tgt[j])

    max_sim = np.amax(sim_matrix, axis=1)

    return max_sim


def bert_pairwise_cos_sim(src, tgt, idf=False):
    from bert_score import score
    if idf:
        p, _, _ = score([src], [tgt], lang='en', idf=True)
        return p.item()

    p, _, _ = score([src], [tgt], lang='en')
    return p.item()
