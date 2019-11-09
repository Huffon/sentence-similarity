from utils.basic import *
from utils.ts_ss import triangle_sector_similarity

from spacy.lang.en import English
from spacy.tokenizer import Tokenizer
from allennlp.modules.elmo import Elmo, batch_to_ids

options_file = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
weight_file = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"


class ELMoCalculator:
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

        nlp = English()
        tokenizer = Tokenizer(nlp.vocab)

        sentences = [[tok.text for tok in tokenizer(sentence)]
                     for sentence in self.sentences]

        char_ids = batch_to_ids(sentences)

        elmo = Elmo(options_file, weight_file, 1, dropout=0)

        if self.verbose:
            print(f'[LOGGING] Now embedding sentence...')

        embeddings = elmo(char_ids)['elmo_representations'][0].squeeze(0)
        embeddings = embeddings.detach().numpy()

        if 'pairwise' not in self.method:
            summed_embeddings = vector_summation(embeddings)

        method = methods[self.method]

        if self.verbose:
            print(f'[LOGGING] Calculating similarity between sentences...')

        similarity = method(summed_embeddings, summed_embeddings)
        plot_similarity(self.sentences, similarity, self.method)
