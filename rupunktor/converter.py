import numpy as np
import re

from nltk import word_tokenize


class Tag:
    SPACE = '<space>'
    COMMA = '<comma>'
    PERIOD = '<period>'

    NUM = '<num>'
    UNK = '<unknown>'
    ENG = '<eng>'
    EOS = '</s>'


PUNKT_TAGS = [Tag.SPACE, Tag.COMMA, Tag.PERIOD]
EOS_TAGS = [Tag.PERIOD]
CRAP_TOKENS = {'%$#@&^*`~'}


def is_tag(token):
    return token in Tag.__dict__.values()


class Converter:
    def __init__(self):
        self.punkt_to_tag = {',': Tag.COMMA, '.': Tag.PERIOD}
        self.tag_to_punkt = {p: d for d, p in self.punkt_to_tag.items()}
        self.digits_checker = re.compile('.*\d')
        self.eng_checker = re.compile('.*[a-zA-Z]')

    def readable_tag(self, tag):
        return self.tag_to_punkt[tag]

    def convert_sentence(self, sentence):
        tokens = word_tokenize(sentence)
        converted = []
        for tok in tokens:
            t = tok.lower()
            if t in CRAP_TOKENS:
                continue
            elif t in self.punkt_to_tag:
                converted.append(self.punkt_to_tag[t])
            elif self.digits_checker.match(t):
                converted.append(Tag.NUM)
            elif self.eng_checker.match(t):
                converted.append(Tag.ENG)
            else:
                converted.append(tok)
        return converted

    def convert_file(self, file, outfile, log_every=100000):
        """
        Convert all sentences in file to normalized format with tags
        Resulted file will have one sentence per line
        """
        ready_line = []
        converted_cnt = 0
        for i, line in enumerate(file, 1):
            converted_line = self.convert_sentence(line)
            for tok in converted_line:
                ready_line.append(tok)
                if tok in EOS_TAGS:
                    print(' '.join(ready_line), file=outfile)
                    converted_cnt += 1
                    ready_line = []
            if i % log_every == 0:
                print('--- Processed {} lines, found {} sentences so far'.format(i, converted_cnt))
        if ready_line:
            print('Warning: last line has no end of sentence tag in the end. Will add Tag.PERIOD manually')
            ready_line.append(Tag.PERIOD)
            print(' '.join(ready_line), file=outfile)

    @staticmethod
    def strip_punkt_tags(converted_sentence):
        if isinstance(converted_sentence, str):
            converted_sentence = converted_sentence.split()
        return ' '.join(t for t in converted_sentence if t not in PUNKT_TAGS)


def convert_to_fixed_len_sentences(preprocessed_sentences, outfile, seq_len=50, norm_info=(0, 1000), log_every=100000):
    acc_toks = []
    eos_found = False
    skip_until_eos = False
    min_len, max_len = norm_info
    skip_cnt = 0
    expected_punkt = False
    words_cnt = 0
    for i, s in enumerate(preprocessed_sentences, 1):
        if i % log_every == 0:
            print('--- Processed {} sentences, skipped {}'.format(i, skip_cnt))

        s = s.lower().split()
        if not (min_len <= len(s) <= max_len):
            skip_cnt += 1
            continue

        for tok in s:
            if tok in EOS_TAGS:
                eos_found = True

            if skip_until_eos:
                # If found eos, skip one more time, thus have proper sentence start
                skip_until_eos = not eos_found
                eos_found = False
                continue

            if tok in PUNKT_TAGS:
                if expected_punkt:
                    acc_toks.append(tok)
                    expected_punkt = False
            else:
                acc_toks.append(tok)
                expected_punkt = True
                words_cnt += 1

            if words_cnt == seq_len + int(expected_punkt) - 1:
                if eos_found:
                    if expected_punkt:
                        print(' '.join(acc_toks[:-1] + [Tag.EOS]), file=outfile)
                    else:
                        print(' '.join(acc_toks + [Tag.EOS]), file=outfile)

                # Next sentence should not start from middle
                skip_until_eos = True
                if acc_toks[-1] in EOS_TAGS:
                    # Already reached
                    skip_until_eos = False

                if expected_punkt and not skip_until_eos:
                    # Last token was word
                    acc_toks = acc_toks[-1:]
                else:
                    acc_toks.clear()
                eos_found = False
                words_cnt = len(acc_toks)


def apply_encoder(encoder, sentences, log_every=100000):
    data = []
    for i, s in enumerate(sentences, 1):
        data.append(encoder.encode_sentence(s))
        if i % log_every == 0:
            print('--- Processed {} lines'.format(i))
    return np.asarray(data)
