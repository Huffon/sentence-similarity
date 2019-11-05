import math
import numpy as np
from numpy.linalg import norm
from numpy import arccos


def angular_cosine_similarity(src, tgt):
    similarity_ = (1-np.arccos(np.dot(src, tgt)/(np.linalg.norm(src)*np.linalg.norm(tgt)))/3.1416)
    return similarity_

def cosine_similarity(src, tgt):
    similarity_ = np.dot(src, tgt) / (norm(src) * norm(tgt))
    return similarity_


def euclidean_distance(src, tgt):
    distance_ = math.sqrt(
        sum([(x - y) ** 2 for x, y in zip(src, tgt)])
    )
    return distance_


def inner_product(src, tgt):
    similarity = np.inner(src, tgt)
    return similarity


def vector_summation(sentence):
    sent_len = sentence.shape[0]
    sentence = sentence.sum(axis=0) / sent_len
    return sentence
