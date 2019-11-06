from utils.basic import *
from utils.ts_ss import triangle_sector_similarity
from utils.pairwise import bert_pairwise_cos_sim

from bert_score import score
from sentence_transformers import SentenceTransformer


class BERTCalculator:
    def __init__(self, config):
        self.source = config.source
        self.target = config.target
        self.method = config.method
        self.verbose = config.verbose

    def calculate(self):
        methods = {
            'cosine': cosine_similarity,
            'manhattan': manhattan_distance,
            'euclidean': euclidean_distance,
            'angular': angular_distance,
            'inner': inner_product,
            'ts-ss': triangle_sector_similarity,
            'pairwise': bert_pairwise_cos_sim,
            'pairwise-idf': bert_pairwise_cos_sim
        }

        if self.method not in methods:
            return False

        if 'pairwise' in self.method:
            if 'idf' in self.method:
                return bert_pairwise_cos_sim(self.source, self.target, idf=True)
            return bert_pairwise_cos_sim(self.source, self.target)

        model = SentenceTransformer('bert-base-nli-mean-tokens')

        if self.verbose:
            print(f'[Embedding] Now embedding sentence...')
        embed_source = model.encode([self.source])[0]
        embed_target = model.encode([self.target])[0]

        method = methods[self.method]
        if self.verbose:
            print(f'[Calculating] Calculating similarity between sentences...')
        similarity = method(embed_source, embed_target)
        return similarity
