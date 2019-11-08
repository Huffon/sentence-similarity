from absl import logging
from utils.basic import *
from utils.ts_ss import triangle_sector_similarity

import tensorflow as tf
import tensorflow_hub as hub

logging.set_verbosity(logging.INFO)
module_url = "https://tfhub.dev/google/universal-sentence-encoder/2"


class USECalculator:
    def __init__(self, config, sentences):
        self.sentences = sentences
        self.method = config.method
        self.verbose = config.verbose

    def calculate(self):
        methods = {
            'cosine': cosine_sim,
            'manhattan': manhattan_dist,
            'euclidean': euclidean_dist,
            'angular': angular_distance,
            'inner': inner_product,
            'ts-ss': triangle_sector_similarity,
        }

        if self.method not in methods:
            print(f'[ERROR] The method you chosen is not supported yet.')
            return False

        model = hub.Module(module_url)
        if self.verbose:
            print(f'[LOGGING] Now embedding sentence...')

        with tf.Session() as session:
            session.run(
                [tf.compat.v1.global_variables_initializer(),
                 tf.compat.v1.tables_initializer()])
            embeddings = session.run(model(self.sentences))

        method = methods[self.method]

        if self.verbose:
            print(f'[LOGGING] Calculating similarity between sentences...')

        similarity = method(embeddings, embeddings)
        plot_similarity(self.sentences, similarity, self.method)
