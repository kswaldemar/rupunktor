from operator import itemgetter
from collections import defaultdict

import numpy as np
from Stemmer import Stemmer

from rupunktor.converter import Tag, PUNKT_TAGS


def _freq_sorter(t):
    return t[1], t[0]


class Corpus:
    def __init__(self):
        self.word_to_idx = {}
        self.idx_to_word = []

    @property
    def vocab_size(self):
        return len(self.word_to_idx)

    def encode_word(self, word):
        return self.word_to_idx.get(word.lower(), len(self.idx_to_word) - 1)

    def build(self, sentences, vocabulary_size=50000, log_every=100000):
        vocab = defaultdict(int)
        print('= Start building vocabulary')
        for i, s in enumerate(sentences, 1):
            for tok in s.lower().split():
                if tok in PUNKT_TAGS:
                    continue
                vocab[tok.lower()] += 1
            if i % log_every == 0:
                print('--- Processed {} sentences'.format(i))

        print('= Built vocabulary with size {}'.format(len(vocab)))
        if vocabulary_size < len(vocab):
            print('= Trim it to {}'.format(vocabulary_size))
        word_freq = list(map(itemgetter(0), sorted(vocab.items(), key=_freq_sorter, reverse=True)))
        word_freq = word_freq[:vocabulary_size]
        print('Top 10 most frequent words: {}'.format(', '.join(word_freq[:10])))
        print('Top 10 least frequent words: {}'.format(', '.join(word_freq[-10:])))

        print('= Building word to index mapping')
        if Tag.NUM not in word_freq:
            word_freq[-2] = Tag.NUM

        if Tag.ENG not in word_freq:
            word_freq[-1] = Tag.ENG

        assert Tag.EOS not in word_freq
        word_freq.append(Tag.EOS)

        assert Tag.UNK not in word_freq
        word_freq.append(Tag.UNK)

        self.idx_to_word.clear()
        self.word_to_idx.clear()
        for w in word_freq:
            self.word_to_idx[w] = len(self.idx_to_word)
            self.idx_to_word.append(w)

        print('= Built mappings')
        print('idx_to_word size = {}, word_to_idx size = {}'.format(len(self.idx_to_word), len(self.word_to_idx)))

    def encode_sentence(self, sentence):
        """
            Encode preprocessed sentence to x, y index vectors
        :param sentence:
        :return: pair of X, Y numpy arrays
        """
        tokens = sentence.lower().split()
        x_vec = []
        y_vec = []
        expected_punct = False
        for tok in tokens:
            if tok in PUNKT_TAGS:
                if expected_punct:
                    y_vec.append(PUNKT_TAGS.index(tok))
                    expected_punct = False
            else:
                if expected_punct:
                    y_vec.append(0)
                x_vec.append(self.encode_word(tok))
                expected_punct = True
        if expected_punct:
            y_vec.append(0)
        return np.asarray(x_vec), np.asarray(y_vec)


class StemCorpus(Corpus):
    def __init__(self):
        super().__init__()
        self.stemmer = Stemmer('russian')

    def __getstate__(self):
        return self.word_to_idx, self.idx_to_word

    def __setstate__(self, state):
        self.stemmer = Stemmer('russian')
        self.word_to_idx, self.idx_to_word = state

    def encode_word(self, word):
        stem_form = self.stemmer.stemWord(word.lower())
        return self.word_to_idx.get(stem_form, len(self.idx_to_word) - 1)

    def build(self, sentences, vocabulary_size=50000, log_every=100000):
        print('= Start building vocabulary')
        vocab = defaultdict(int)
        saved_sentences = []
        for i, s in enumerate(sentences, 1):
            line = s.lower().split()
            for tok in line:
                if tok in PUNKT_TAGS:
                    continue
                stem_form = self.stemmer.stemWord(tok.lower())
                vocab[stem_form] += 1
            if i % log_every == 0:
                print('--- Processed {} sentences'.format(i))
            saved_sentences.append(line)

        print('= Built vocabulary with size {}'.format(len(vocab)))
        if vocabulary_size < len(vocab):
            print('= Trim it to {}'.format(vocabulary_size))
        word_freq = list(map(itemgetter(0), sorted(vocab.items(), key=_freq_sorter, reverse=True)))
        word_freq = word_freq[:vocabulary_size]

        print('Top 10 most frequent words: {}'.format(', '.join(word_freq[:10])))
        print('Top 10 least frequent words: {}'.format(', '.join(word_freq[-10:])))

        print('= Building word to index mapping')
        if Tag.NUM not in word_freq:
            word_freq[-2] = Tag.NUM

        if Tag.ENG not in word_freq:
            word_freq[-1] = Tag.ENG

        assert Tag.EOS not in word_freq
        word_freq.append(Tag.EOS)

        assert Tag.UNK not in word_freq
        word_freq.append(Tag.UNK)

        self.idx_to_word.clear()
        self.word_to_idx.clear()
        for w in word_freq:
            self.word_to_idx[w] = len(self.idx_to_word)
            self.idx_to_word.append(w)

        print('= Built mappings')
        print('idx_to_word size = {}, word_to_idx size = {}'.format(len(self.idx_to_word), len(self.word_to_idx)))