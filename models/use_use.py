from absl import logging
from utils.basic import *
from utils.ts_ss import triangle_sector_similarity

import tensorflow as tf
import tensorflow_hub as hub

logging.set_verbosity(logging.INFO)
module_url = "https://tfhub.dev/google/universal-sentence-encoder/2"


class USECalculator:
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
        }

        if 'pairwise' in self.method:
            print(f'[ERROR] Pairwise similarity is not supported with DAN')
            return False

        if self.method not in methods:
            return False

        embed = hub.Module(module_url)
        if self.verbose:
            print(f'[LOGGING] Now embedding sentence...')
        sentences = [self.source, self.target]

        with tf.Session() as session:
            session.run(
                [tf.compat.v1.global_variables_initializer(),
                 tf.compat.v1.tables_initializer()])
            embeddings = session.run(embed(sentences))
            embed_source = np.array(embeddings[0])
            embed_target = np.array(embeddings[1])

        method = methods[self.method]
        if self.verbose:
            print(f'[LOGGING] Calculating similarity between sentences...')
        similarity = method(embed_source, embed_target)
        return similarity
