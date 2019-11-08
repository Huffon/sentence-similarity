import numpy as np
from utils.basic import cosine_sim


# Not implemented yet
def pairwise_cos_sim(src, tgt):
    sim_matrix = np.zeros([len(src), len(tgt)])

    for i in range(len(src)):
        for j in range(len(tgt)):
            sim_matrix[i][j] = cosine_sim(src[i], tgt[j])

    max_sim = np.amax(sim_matrix, axis=1)

    return max_sim


# Not implemented yet
def pairwise_cos_sim_idf(src, tgt):
    sim_matrix = np.zeros([len(src), len(tgt)])

    for i in range(len(src)):
        for j in range(len(tgt)):
            sim_matrix[i][j] = cosine_sim(src[i], tgt[j])

    max_sim = np.amax(sim_matrix, axis=1)

    return max_sim


def bert_pairwise_cos_sim(sentences, idf=False):
    import itertools
    from bert_score import score
    src_len = len(sentences)

    refs = [[sentence] * src_len for sentence in sentences]
    refs = list(itertools.chain(*refs))
    hyps = sentences * src_len

    if idf:
        p, _, _ = score(refs, hyps, lang='en', idf=True)
        p = p.reshape(src_len, -1)
        return p.detach().numpy()

    p, _, _ = score(refs, hyps, lang='en')
    p = p.reshape(src_len, -1)
    return p.detach().numpy()
