from utils.basic import *
from utils.ts_ss import triangle_sector_similarity
from utils.pairwise import pairwise_cos_sim, pairwise_cos_sim_idf

from spacy.lang.en import English
from spacy.tokenizer import Tokenizer
from allennlp.modules.elmo import Elmo, batch_to_ids

options_file = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
weight_file = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"


class ELMoCalculator:
    def __init__(self, config):
        self.source = config.source
        self.target = config.target
        self.method = config.method

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

        if 'pairwise' in self.method:
            print(f'[ERROR] Pairwise similarity is not supported with ELMo yet')
            return False

        if self.method not in methods:
            return False

        nlp = English()
        tokenizer = Tokenizer(nlp.vocab)

        tokenized_source = [tok.text for tok in tokenizer(self.source)]
        tokenized_target = [tok.text for tok in tokenizer(self.target)]

        source_ids = batch_to_ids([tokenized_source])
        target_ids = batch_to_ids([tokenized_target])

        elmo = Elmo(options_file, weight_file, 1, dropout=0)

        print(f'[Embedding] Now embedding sentence...')
        embed_source = elmo(source_ids)['elmo_representations'][0].squeeze(0)
        embed_target = elmo(target_ids)['elmo_representations'][0].squeeze(0)

        embed_source = embed_source.detach().numpy()
        embed_target = embed_target.detach().numpy()

        if 'pairwise' not in self.method:
            embed_source = vector_summation(embed_source)
            embed_target = vector_summation(embed_target)

        method = methods[self.method]
        print(f'[Calculating] Calculating similarity between sentences...')
        similarity = method(embed_source, embed_target)
        return similarity
