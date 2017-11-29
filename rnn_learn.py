### load data file with structure

import math as math
import numpy as np
import pickle


DATA_DIR ="words_file"
SEQ_LENGTH = 50
createDtb = True #load and save new data to pickle


def load_data(DATA_DIR, SEQ_LENGTH):

    data=load_pickle(DATA_DIR)

    # data = open(data_dir, 'r').read()
    words = list(set(data))
    VOCAB_SIZE = len(words)

    print('Data length: {} characters'.format(len(data)))
    print('Vocabulary size: {} characters'.format(VOCAB_SIZE))
    print("words = {}".format(words))

    ix_to_char = {ix: char for ix, char in enumerate(words)}
    char_to_ix = {char: ix for ix, char in enumerate(words)}

    X = np.zeros((math.ceil(len(data) / SEQ_LENGTH), SEQ_LENGTH, VOCAB_SIZE))
    y = np.zeros((math.ceil(len(data) / SEQ_LENGTH), SEQ_LENGTH, VOCAB_SIZE))

    print("for[0:" + str(math.floor(len(data) / SEQ_LENGTH)) + "]")

    for i in range(0, math.floor(len(data) / SEQ_LENGTH)):
        X_sequence = data[i * SEQ_LENGTH:(i + 1) * SEQ_LENGTH]
        X_sequence_ix = [char_to_ix[value] for value in X_sequence] #return ix value for each char in sequence
        input_sequence = np.zeros((SEQ_LENGTH, VOCAB_SIZE))

        for j in range(SEQ_LENGTH):
            input_sequence[j][X_sequence_ix[j]] = 1.
            X[i] = input_sequence

        if (i % 100) == 0:
            print(i, end=", ")

        y_sequence = data[i * SEQ_LENGTH + 1:(i + 1) * SEQ_LENGTH + 1]
        y_sequence_ix = [char_to_ix[value] for value in y_sequence]
        target_sequence = np.zeros((SEQ_LENGTH, VOCAB_SIZE))

        for j in range(SEQ_LENGTH):
            target_sequence[j][y_sequence_ix[j]] = 1.
            y[i] = target_sequence
    return X, y, VOCAB_SIZE, ix_to_char


def save_pickle(name, data):
    data_dir = 'data/' + name + '.pickle'

    with open(data_dir, 'wb') as f:
        # Pickle the 'data' dictionary using the highest protocol available.
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)


def load_pickle(name):
    data_dir = 'data/' + name + '.pickle'
    with open(data_dir, 'rb') as f:
        data = pickle.load(f)
    return data



def create_save_dtb():
    X, y, VOCAB_SIZE, ix_to_char = load_data(DATA_DIR, SEQ_LENGTH)

    save_pickle('X', X)
    save_pickle('y', y)
    save_pickle('VOCAB_SIZE', VOCAB_SIZE)
    save_pickle('ix_to_char', ix_to_char)

    # return X, y, VOCAB_SIZE, ix_to_char


def load_dtb():
    X = load_pickle('X')
    y = load_pickle('y')
    VOCAB_SIZE = load_pickle('VOCAB_SIZE')
    ix_to_char = load_pickle('ix_to_char')

    return X, y, VOCAB_SIZE, ix_to_char

if(createDtb):
    create_save_dtb()

X, y, VOCAB_SIZE, ix_to_char = load_dtb() #everytime load the deta from pickle

print (X)