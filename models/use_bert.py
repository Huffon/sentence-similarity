from utils.basic import *
from utils.ts_ss import triangle_sector_similarity

from bert_score import score


class BERTCalculator:
    def __init__(self, config):
        self.source = config.source
        self.target = config.target
        self.method = config.method

    def calculate(self):
        methods = {
            'cosine': cosine_similarity,
            'euclidean': euclidean_distance,
            'inner': inner_product,
            'ts-ss': triangle_sector_similarity,
            'pairwise': bert_pairwise_cosine_similarity,
            'pairwise-idf': 5
        }

        if self.method not in methods:
            return False

        if 'pairwise' not in self.method:
            embed_source = vector_summation(embed_source)
            embed_target = vector_summation(embed_target)

        method = methods[self.method]
        print(f'[Calculating] Calculating similarity between sentences...')
        similarity = method(embed_source, embed_target)
        return similarity
