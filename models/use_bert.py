from utils.basic import *
from utils.ts_ss import triangle_sector_similarity
from utils.pairwise import bert_pairwise_cos_sim
from sentence_transformers import SentenceTransformer


class BERTCalculator:
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
            'pairwise': bert_pairwise_cos_sim,
            'pairwise-idf': bert_pairwise_cos_sim
        }

        if self.method not in methods:
            print(f'[ERROR] The method you chosen is not supported yet.')
            return False

        if 'pairwise' in self.method:
            if 'idf' in self.method:
                similarity = bert_pairwise_cos_sim(self.sentences, idf=True)
                plot_similarity(self.sentences, similarity, self.method)
                return

            similarity = bert_pairwise_cos_sim(self.sentences)
            plot_similarity(self.sentences, similarity, self.method)
            return

        model = SentenceTransformer('bert-base-nli-mean-tokens')

        if self.verbose:
            print(f'[LOGGING] Now embedding sentence...')

        embed_sentences = np.asarray(model.encode(self.sentences))
        method = methods[self.method]

        if self.verbose:
            print(f'[LOGGING] Calculating similarity between sentences...')

        similarity = method(embed_sentences, embed_sentences)
        plot_similarity(self.sentences, similarity, self.method)
