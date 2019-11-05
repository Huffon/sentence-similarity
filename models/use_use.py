from utils.basic import *
from utils.ts_ss import triangle_sector_similarity

import tensorflow as tf
import tensorflow_hub as hub


module_url = "https://tfhub.dev/google/universal-sentence-encoder/2"


class USECalculator:
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
            'pairwise': 4,
            'pairwise-idf': 5
        }

        if self.method not in methods:
            return False

        model = hub.Module(module_url)

        print(f'[Embedding] Now embedding sentence...')
        embed_source = model(self.source)
        embed_target = model(self.target)

        method = methods[self.method]
        print(f'[Calculating] Calculating similarity between sentences...')
        similarity = method(embed_source, embed_target)
        return similarity
