import argparse

import numpy as np
import os

from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras.utils import to_categorical

import rupunktor.model_zoo as models

# N_HIDDEN = 128
# VOCABULARY_SIZE = 50002
# TAGS_SPACE_SIZE = 18


def _create_model():
    return models.cut_emb_bgru(hidden_units=128, words_vocabulary_size=50002)


DATA_DIR = './runews_data/'

MODELS_DIR = './models/'
CHECKPOINTS_DIR = './weights/'
COMPILE_OPTS = dict(optimizer='adagrad', loss='categorical_crossentropy', metrics=['categorical_accuracy'])


def main(args):
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)

    if not os.path.exists(CHECKPOINTS_DIR):
        os.makedirs(CHECKPOINTS_DIR)

    data_file = os.path.join(args.data_dir, args.data_file)
    tags_file = os.path.join(args.data_dir, 'morph_tags.npy')

    xy_all = np.load(data_file)
    x1_all = xy_all[:, 0, :]
    y_all = to_categorical(xy_all[:, 1, :])
    # Drop last y mark, because we cannot predict signs after </s> tag
    y_all = y_all[:, :-1, :]

    x2_all = None
    if os.path.isfile(tags_file) and args.use_tags:
        x2_all = np.load(tags_file)

    if args.no_embedding:
        # Reshape data for rnn input
        x1_all = x1_all.reshape(x1_all.shape[0], x1_all.shape[1], 1)

    if args.model:
        print('Load model from file {}'.format(args.model))
        model = load_model(args.model)
        model.name = os.path.splitext(args.model)[0]
    else:
        model = _create_model()
        model.compile(**COMPILE_OPTS)

    if args.weight:
        print('Use weights from file {}'.format(args.weights))
        model.load_weights(args.weights)

    save_fname = os.path.join(MODELS_DIR, model.name + '.hdf5')

    print('Will save trained model to {}'.format(save_fname))

    checkpoint = ModelCheckpoint(
        filepath=CHECKPOINTS_DIR + model.name + '.w.{epoch:02d}-{val_categorical_accuracy:.5f}.hdf5',
        monitor='val_categorical_accuracy', save_weights_only=True, period=1, mode='max',
    )
    opts = dict(batch_size=128, epochs=40, verbose=1, validation_split=0.2, callbacks=[checkpoint])
    if x2_all:
        model.fit([x1_all, x2_all], y_all, **opts)
    else:
        model.fit(x1_all, y_all, **opts)
    model.save(save_fname)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', metavar='DIR', default='./runews_data')
    parser.add_argument('--model', metavar='PATH', help='Load pretrained model from file specified')
    parser.add_argument('--weight', metavar='PATH', help='Load pretrained model weights')
    parser.add_argument('--no_embedding', action='store_true')
    parser.add_argument('--use_tags', action='store_true', help='Use morphological tags file morph_tags.npy')
    parser.add_argument('--data_file', metavar='NAME', default='data.npy')
    args = parser.parse_args()
    main(args)
