from utils.basic import *
from utils.ts_ss import triangle_sector_similarity
from utils.pairwise import *

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
            'pairwise': bert_pairwise_cosine_sim,
            'pairwise-idf': bert_pairwise_cosine_sim
        }

        if self.method not in methods:
            return False

        if 'pairwise' in self.method:
            if 'idf' in self.method:
                return bert_pairwise_cosine_sim(self.source, self.target, idf=True)
            return bert_pairwise_cosine_sim(self.source, self.target)

        method = methods[self.method]
        print(f'[Calculating] Calculating similarity between sentences...')
        similarity = method(embed_source, embed_target)
        return similarity
