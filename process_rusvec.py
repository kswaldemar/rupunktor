"""
    Preprocess word2vec file models from http://rusvectores.org/ru/models/
    to format
        legend (['ADJ', 'ADV', 'INTJ', 'NOUN', 'NUM', 'PROPN', 'SYM', 'VERB'])
        word_to_index mapping
        numpy vector of embedding
"""
import sys
import gzip
import pickle
import numpy as np


def decode_corpora(in_fname, out_fname):
    print('Open RusVectors file {}'.format(in_fname))
    with gzip.open(in_fname) as f:
        lines = f.readlines()
    num_words, vec_dim = map(int, lines[0].split())
    print('Number of words {}, vector dimension {}'.format(num_words, vec_dim))
    print('Read words...')
    words = []
    vectors = []
    for line in lines[1:]:
        word, *vec = line.split()
        words.append(word.decode('utf8'))
        vectors.append(list(map(float, vec)))
    print('Actual words/vectors count {}'.format(len(words)))

    parts = set()
    for w in words:
        parts.add(w.split('_')[1])
    parts = list(parts)
    parts.sort()
    print('Word parts: {!r}'.format(parts))

    print('Create dictionary...')
    word_dict = {}
    for idx, w in enumerate(words):
        word, part = w.split('_')
        if word not in word_dict:
            word_dict[word] = [-1 for _ in range(len(parts))]
        word_dict[word][parts.index(part)] = idx

    print('Pickling tuple (legend, word_mapping, numpy_vectors) to file {}'.format(out_fname))
    with gzip.GzipFile(out_fname, 'w') as f:
        pickle.dump((parts, word_dict, np.array(vectors)), f)


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Usage: rusvec_converter.py <path to ruscorpora model> <converted filename>')
    else:
        decode_corpora(sys.argv[1], sys.argv[2])