import math
import numpy as np
from numpy.linalg import norm
from utils.basic import euclidean_distance, cosine_similarity


def theta(src, tgt):
    similarity_ = cosine_similarity(src, tgt)
    theta_ = np.arccos(similarity_) + math.radians(10)
    return theta_


def magnitude_difference(src, tgt):
    src_norm = norm(src)
    tgt_norm = norm(tgt)
    difference_ = np.abs(src_norm - tgt_norm)
    return difference_


def triangle_area_similarity(src, tgt, theta_):
    src_norm = norm(src)
    tgt_norm = norm(tgt)
    triangle_similarity_ = (src_norm * tgt_norm * np.sin(theta_)) / 2
    return triangle_similarity_


def sector_area_similarity(src, tgt, theta_):
    distance_ = euclidean_distance(src, tgt)
    difference_ = magnitude_difference(src, tgt)
    sector_similarity_ = math.pi * (distance_ + difference_) ** 2
    sector_similarity_ *= (theta_ / 360)
    return sector_similarity_


def triangle_sector_similarity(src, tgt):
    theta_ = theta(src, tgt)
    triangle_similarity_ = triangle_area_similarity(src, tgt, theta_)
    sector_similarity_ = sector_area_similarity(src, tgt, theta_)
    ts_ss_ = triangle_similarity_ * sector_similarity_
    return ts_ss_
