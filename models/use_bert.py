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

    def calculate(self):
        methods = {
            'cosine': cosine_similarity,
            'euclidean': euclidean_distance,
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

        print(f'[Embedding] Now embedding sentence...')
        embed_source = model.encode([self.source])[0]
        embed_target = model.encode([self.target])[0]

        method = methods[self.method]
        print(f'[Calculating] Calculating similarity between sentences...')
        similarity = method(embed_source, embed_target)
        return similarity
