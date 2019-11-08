import argparse
from models.use_use import USECalculator
from models.use_elmo import ELMoCalculator
from models.use_bert import BERTCalculator


def main(config):
    models = {
        'use': USECalculator,
        'elmo': ELMoCalculator,
        'bert': BERTCalculator,
    }

    if config.model not in models:
        print(f'[ERROR] The model you chosen is not supported yet.')
        return

    if config.verbose:
        print(f'[LOGGING] Loading the corpus...')

    f = open('corpus.txt', 'r', encoding='utf-8')
    sentences = f.readlines()
    sentences = [sentence.replace('\n', '') for sentence in sentences]

    model = models[config.model](config, sentences)

    if config.verbose:
        print(f'[LOGGING] You chose the "{config.model.upper()}" as a model.\n'
              f'[LOGGING] You chose the "{config.method.upper()}" as a method.')

    model.calculate()

    if config.verbose:
        print(f'[LOGGING] Terminating the program...')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Sentence similarity calculator'
    )
    parser.add_argument('--model', type=str, default='use',
                        choices=['use', 'elmo', 'bert'])
    parser.add_argument('--method', type=str, default='cosine',
                        choices=['cosine', 'manhattan', 'euclidean', 'inner',
                                 'ts-ss', 'angular', 'pairwise', 'pairwise-idf'])
    parser.add_argument('--verbose', type=bool, default=True)
    args = parser.parse_args()
    main(args)
