import itertools

import numpy as np
from bert_score import score

from utils.basic import cosine_sim


def bert_pairwise_cos_sim(sentences, idf=False):
    """BERTScore similarity func
    """
    src_len = len(sentences)

    refs = [[sentence] * src_len for sentence in sentences]
    refs = list(itertools.chain(*refs))
    hyps = sentences * src_len

    if idf:
        p, _, _ = score(refs, hyps, lang="en", idf=True)
        p = p.reshape(src_len, -1)
        return p.detach().numpy()

    p, _, _ = score(refs, hyps, lang="en")
    p = p.reshape(src_len, -1)
    return p.detach().numpy()
