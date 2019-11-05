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

    model = models[config.model](config)
    similarity = model.calculate()

    if not similarity:
        print(f'[ERROR] The method you chosen is not supported.')
        return

    print(f'Similarity using [{config.model}] with. [{config.method}] between\n'
          f'\t source> "{config.source}"\n'
          f'\t target> "{config.target}" is | {similarity: .5f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Sentence similarity calculator'
    )
    parser.add_argument('--source', type=str, default='I ate an apple')
    parser.add_argument('--target', type=str, default='I went to the Apple')
    parser.add_argument('--model', type=str, default='use',
                        choices=['use', 'elmo', 'bert'])
    parser.add_argument('--method', type=str, default='cosine',
                        choices=['cosine', 'euclidean', 'inner', 'ts-ss',
                                 'pairwise', 'pairwise-idf'])
    args = parser.parse_args()
    main(args)
