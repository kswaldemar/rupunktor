from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Bidirectional, GRU, Embedding, Dropout, Lambda
from keras.layers import Input, concatenate

NUM_CLASSES = 3  # Predict [space, comma, period]


def blstm(hidden_units):
    m = Sequential()
    m.add(Bidirectional(LSTM(hidden_units, return_sequences=True), input_shape=(None, 1)))
    m.add(Dense(NUM_CLASSES, activation='softmax'))
    m.summary()
    m.name = 'blstm_{}'.format(hidden_units)
    return m


def bgru(hidden_units):
    m = Sequential()
    m.add(Bidirectional(GRU(hidden_units, return_sequences=True), input_shape=(None, 1)))
    m.add(Dense(NUM_CLASSES, activation='softmax'))
    m.summary()
    m.name = 'bgru_{}'.format(hidden_units)
    return m


def emb_bgru(hidden_units, words_vocabulary_size):
    m = Sequential()
    m.add(Embedding(words_vocabulary_size, hidden_units))
    m.add(Bidirectional(GRU(hidden_units, return_sequences=True)))
    m.add(Dense(NUM_CLASSES, activation='softmax'))
    m.summary()
    m.name = 'emb_bgru_{}'.format(hidden_units)
    return m


def cut_emb_bgru(hidden_units, words_vocabulary_size):
    m = Sequential()
    m.add(Embedding(words_vocabulary_size, hidden_units))
    m.add(Bidirectional(GRU(hidden_units, return_sequences=True)))
    m.add(Dense(NUM_CLASSES, activation='softmax'))
    m.add(Lambda(lambda x: x[:, :-1, :]))
    m.summary()
    m.name = 'cut_emb_bgru_{}'.format(hidden_units)
    return m


def drop_emb_bgru(hidden_units, vocabulary_size):
    m = Sequential()
    m.add(Embedding(vocabulary_size, hidden_units))
    m.add(Bidirectional(LSTM(hidden_units, return_sequences=True)))
    m.add(Dropout(0.4))
    m.add(Dense(NUM_CLASSES, activation='softmax'))
    m.summary()
    m.name = 'drop_emb_bgru_{}'.format(hidden_units)
    return m


def uni_lstm(hidden_units):
    m = Sequential()
    m.add(LSTM(hidden_units, return_sequences=True, input_shape=(None, 1)))
    m.add(Dense(NUM_CLASSES, activation='softmax'))
    m.summary()
    m.name = 'lstm_{}'.format(hidden_units)
    return m


def augmented_gru(hidden_units, words_vocabulary_size, tags_vocabulary_size):
    input1 = Input(shape=(None,), name='word_input')
    input2 = Input(shape=(None,), name='pos_tags_input')

    # Distribute evenly
    emb1_out_units_cnt = hidden_units // 2
    emb2_out_units_cnt = hidden_units - emb1_out_units_cnt

    emb1 = Embedding(words_vocabulary_size, emb1_out_units_cnt, name='word_index_embedding')(input1)
    emb2 = Embedding(tags_vocabulary_size, emb2_out_units_cnt, name='pos_tags_embedding')(input2)

    x = concatenate([emb1, emb2])

    rnn = Bidirectional(GRU(hidden_units, name='gru_layer', return_sequences=True))(x)
    dense = Dense(NUM_CLASSES, activation='softmax', name='output_tags')(rnn)

    m = Model(inputs=[input1, input2], outputs=[dense])
    m.summary()
    m.name = 'augmented_gru_{}'.format(hidden_units)
    return m
