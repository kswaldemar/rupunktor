import numpy as np
from pymorphy2 import MorphAnalyzer

from rupunktor.converter import PUNKT_TAGS


class PosTagger:
    def __init__(self):
        self.morph = MorphAnalyzer()
        self.pos_tag_to_idx = {t: i for i, t in enumerate(sorted(self.morph.TagClass.PARTS_OF_SPEECH))}
        self.pos_tag_to_idx[None] = self.pos_tag_to_idx['NOUN']  # Treat all unknowns as NOUN
        self.case_to_idx = {t: i for i, t in enumerate(sorted(self.morph.TagClass.CASES))}
        self.case_to_idx[None] = self.case_to_idx['nomn']
        self.number_to_idx = {t: i for i, t in enumerate(sorted(self.morph.TagClass.NUMBERS))}
        self.number_to_idx[None] = len(self.number_to_idx)

        self.dec_len = [len(self.pos_tag_to_idx), len(self.case_to_idx), len(self.number_to_idx)]

    @property
    def vocab_size(self):
        return self.flat_index(np.array(self.dec_len) - 1) + 1

    def flat_index(self, vec):
        return vec[0] * self.dec_len[1] * self.dec_len[2] + vec[1] * self.dec_len[2] + vec[2]

    def encode_word(self, word):
        tag = self.morph.parse(word)[0].tag
        return [
            self.pos_tag_to_idx[tag.POS],
            self.case_to_idx[tag.case],
            self.number_to_idx[tag.number]
        ]

    def encode_sentence(self, sentence):
        tokens = sentence.lower().split()
        ret = []
        for tok in tokens:
            if tok in PUNKT_TAGS:
                continue

            ret.append(self.encode_word(tok))
        return ret
