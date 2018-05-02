import numpy as np
import os

from keras.models import load_model

from rupunktor.pos_tagger import PosTagger
from rupunktor.converter import Converter, PUNKT_TAGS, Tag, EOS_TAGS, CRAP_TOKENS
from rupunktor.utils import pickle_load


class InteractWrapper:
    def __init__(self, bundle_dir):
        corpus_file = os.path.join(bundle_dir, 'corpus.pkl')
        model_file = os.path.join(bundle_dir, 'model.hdf5')

        self.corpus = pickle_load(corpus_file)
        self.model = load_model(model_file)
        self.converter = Converter()
        self.pos_tagger = None
        if len(self.model.input_layers) > 1:
            self.pos_tagger = PosTagger()

    def cleanup_text(self, text):
        return ''.join(t for t in text if t not in CRAP_TOKENS)

    def process_text_block(self, tokens, show_confidence=False, show_unknown=False):
        # Replace with tags
        st = self.converter.process_tokens(tokens) + [Tag.EOS]
        # Encode to index vector
        st = ' '.join(st)
        x1, _ = self.corpus.encode_sentence(st)
        x1 = x1.reshape(1, -1)
        if self.pos_tagger:
            x2 = self.pos_tagger.encode_sentence(st)
            x2 = np.asarray([self.pos_tagger.flat_index(v) for v in x2])
            x2 = x2.reshape(1, -1)
            y_pred = self.model.predict([x1, x2])
        else:
            y_pred = self.model.predict(x1)

        y_pred = y_pred[0, :-1, :]
        y_sm = np.argmax(y_pred, -1)

        # Strip punctuation marks
        tokens = [t for t in tokens if t not in self.converter.punkt_to_tag]

        # Reverse pass through corpus
        if show_unknown:
            tokens = [self.corpus.idx_to_word[i] for i in x1.ravel()]

        out = []
        need_capitalize = True
        for w, pidx, pv in zip(tokens, y_sm, y_pred):
            out.append(w.capitalize() if need_capitalize else w)
            tag = PUNKT_TAGS[pidx]
            if tag != Tag.SPACE:
                out.append(self.converter.readable_tag(tag))
                if show_confidence:
                    out.append('(p:{:.2f})'.format(pv[pidx]))
            out.append(' ')
            need_capitalize = tag in EOS_TAGS
        if len(tokens) > len(y_sm):
            out.append(tokens[-1])

        return ''.join(out).strip()

    def process_text(self, text, show_confidence=False, show_unknown=False, lowercase=False):
        if lowercase:
            text = text.lower()

        text = self.cleanup_text(text)
        tokens = self.converter.tokenize_sentence(text)

        return self.process_text_block(tokens, show_confidence, show_unknown)

        # print('Found {} tokens'.format(len(tokens)))
        # window_size = 50
        # ret = []
        # for i in range(0, len(tokens), window_size):
        #     ret.append(
        #         self.process_text_block(tokens[i * window_size: (i + 1) * window_size], show_confidence, show_unknown)
        #     )
        # if len(tokens) % window_size > 0:
        #     ret.append(
        #         self.process_text_block(tokens[len(tokens) // window_size * window_size:],
        #                                 show_confidence,
        #                                 show_unknown)
        #     )
        # return ''.join(ret)
