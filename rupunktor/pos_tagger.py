import numpy as np
from pymorphy2 import MorphAnalyzer

from rupunktor.converter import PUNKT_TAGS


class PosTagger:
    def __init__(self):
        self.morph = MorphAnalyzer()
        self.pos_tag_to_idx = {
            tag: idx for idx, tag in enumerate('NOUN ADJF ADJS COMP VERB INFN PRTF PRTS GRND '
                                               'NUMR ADVB NPRO PRED PREP CONJ PRCL INTJ'.split())
        }
        self.tag_space_size = len(self.pos_tag_to_idx)

    def encode_word(self, word):
        p = self.morph.parse(word)[0]
        return self.pos_tag_to_idx.get(p.tag.POS, self.tag_space_size)

    def encode_sentence(self, sentence):
        tokens = sentence.lower().split()
        ret = []
        for tok in tokens:
            if tok in PUNKT_TAGS:
                continue

            ret.append(self.encode_word(tok))
        return ret
