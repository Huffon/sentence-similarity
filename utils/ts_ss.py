import math
import numpy as np
from numpy.linalg import norm
from utils.basic import euclidean_dist, cosine_sim


def theta(src, tgt):
    similarity_ = cosine_sim(src, tgt)
    np.fill_diagonal(similarity_, 1)
    theta_ = np.arccos(similarity_) + math.radians(10)
    return theta_


def magnitude_difference(src, tgt):
    src_len = len(src)
    src_norm = norm(src, axis=1).repeat(src_len).reshape(src_len, src_len)
    tgt_norm = norm(tgt, axis=1)
    difference_ = np.abs(src_norm - tgt_norm)
    return difference_


def triangle_area_similarity(src, tgt, theta_):
    src_norm = norm(src, axis=1)
    tgt_norm = norm(tgt, axis=1)
    triangle_similarity_ = (src_norm * tgt_norm * np.sin(theta_)) / 2
    return triangle_similarity_


def sector_area_similarity(src, tgt, theta_):
    distance_ = euclidean_dist(src, tgt)
    difference_ = magnitude_difference(src, tgt)
    sector_similarity_ = math.pi * ((distance_ + difference_) ** 2)
    sector_similarity_ *= (theta_ / 360)
    return sector_similarity_


def triangle_sector_similarity(src, tgt):
    theta_ = theta(src, tgt)
    triangle_similarity_ = triangle_area_similarity(src, tgt, theta_)
    sector_similarity_ = sector_area_similarity(src, tgt, theta_)
    ts_ss_ = triangle_similarity_ * sector_similarity_
    return ts_ss_
