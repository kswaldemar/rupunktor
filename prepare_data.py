import os
import sys
import argparse

import numpy as np

from rupunktor import converter, corpus_build
from rupunktor.pos_tagger import PosTagger
from rupunktor.utils import pickle_load, pickle_save

CORPUS_TYPE = corpus_build.StemCorpus


def main(args):
    dest_dir = args.dest_directory
    if not os.path.exists(dest_dir):
        print('- Create directory {}'.format(dest_dir))
        os.makedirs(dest_dir)

    if not os.path.isdir(dest_dir):
        print('Error: path {} is not directory'.format(dest_dir))
        return

    preprocessed_file = os.path.join(dest_dir, 'preprocessed.txt')
    fixlen_file = os.path.join(dest_dir, 'fixlen_preprocessed.txt')
    corpus_file = os.path.join(dest_dir, 'corpus.pkl')
    data_file = os.path.join(dest_dir, 'data.npy')
    # Second input - morphological tags, created together with data.npy
    tags_file = os.path.join(dest_dir, 'morph_tags.npy')

    if not os.path.exists(preprocessed_file):
        if not args.input_file:
            print('Error: Failed to create {}, input file not specified'.format(preprocessed_file))
            return
        print('= Create file {} from file {}'.format(preprocessed_file, args.input_file))
        conv = converter.Converter()
        with open(args.input_file, 'r') as in_f:
            with open(preprocessed_file, 'w') as out_f:
                conv.convert_file(in_f, out_f)
        print('= Created {}'.format(preprocessed_file))
    else:
        print('= File {} already exists'.format(preprocessed_file))

    if not os.path.exists(corpus_file):
        print('= Build corpus file {}'.format(corpus_file))
        with open(preprocessed_file, 'r') as sentences_file:
            corp = CORPUS_TYPE()
            corp.build(sentences_file, args.corpus_vocabulary)
            pickle_save(corpus_file, corp)
    else:
        print('= File {} already exists'.format(corpus_file))

    if not os.path.exists(fixlen_file):
        print('= Build fixed length sentence file {}'.format(fixlen_file))
        with open(preprocessed_file, 'r') as sentences_file:
            with open(fixlen_file, 'w') as out_f:
                converter.convert_to_fixed_len_sentences(sentences_file, out_f, args.seq_len,
                                                         norm_info=(5, 30))
    else:
        print('= File {} already exists'.format(fixlen_file))

    if not os.path.exists(data_file):
        print('= Build X, Y vectors data file {}'.format(data_file))
        with open(fixlen_file, 'r') as sentences_file:
            corp = pickle_load(corpus_file)
            data = converter.apply_encoder(corp, sentences_file, log_every=100000)
            np.save(data_file, data)
    else:
        print('= File {} already exists'.format(data_file))

    if not os.path.exists(tags_file):
        if args.build_tags:
            print('= Build morphological tags data file {}'.format(tags_file))
            with open(fixlen_file, 'r') as sentences_file:
                tagger = PosTagger()
                data = converter.apply_encoder(tagger, sentences_file, log_every=20000)
                np.save(tags_file, data)
    else:
        print('= File {} already exists'.format(tags_file))

    return True


parser = argparse.ArgumentParser(description='Prepare data to suitable format for rupunktor')
parser.add_argument('dest_directory',
                    help='Directory to write all processed data')
parser.add_argument('--file', dest='input_file', metavar='FILENAME',
                    help='File with unprocessed sentences')
parser.add_argument('--seq_len', type=int, metavar='N', default=50,
                    help='Sentence length to create output data vectors')
parser.add_argument('--corpus_vocabulary', type=int, metavar='N', default=50000,
                    help='Size of corpus vocabulary')
parser.add_argument('--build_tags', action="store_true",
                    help='Create morphological tag file (Warning: May take long time)')

if __name__ == '__main__':
    args = parser.parse_args()
    if not main(args):
        sys.exit(1)
