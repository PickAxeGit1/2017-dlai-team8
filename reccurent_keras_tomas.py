import math as math
# from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import time
import csv
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM, SimpleRNN
from keras.layers.wrappers import TimeDistributed


# import argparse
# from RNN_utils import *


# Parsing arguments for Network definition

DATA_DIR = 'resources/LOTR.txt'
BATCH_SIZE = 50
LAYER_NUM = 2
SEQ_LENGTH = 50
HIDDEN_DIM = 500
GENERATE_LENGTH = 500
NB_EPOCH = 20 # standard:20
MODE = "train"
WEIGHTS = ""
# VOCAB_SIZE = None
createDtb = True #load and save new data to pickle
loadModel=False #load existing keras model and dont train it again

# from __future__ import print_function
# import numpy as np


# method for generating text
def generate_text(model, length, vocab_size, ix_to_char):
    # starting with random character
    ix = [np.random.randint(vocab_size)]
    y_char = [ix_to_char[ix[-1]]]
    X = np.zeros((1, length, vocab_size))
    for i in range(length):
        # appending the last predicted character to sequence
        X[0, i, :][ix[-1]] = 1
        print(ix_to_char[ix[-1]], end="")
        ix = np.argmax(model.predict(X[:, :i + 1, :])[0], 1)
        y_char.append(ix_to_char[ix[-1]])
    return ('').join(y_char)


# method for preparing the training data
def load_data(data_dir, seq_length):
    data = open(data_dir, 'r').read()
    chars = list(set(data))
    VOCAB_SIZE = len(chars)

    print('Data length: {} characters'.format(len(data)))
    print('Vocabulary size: {} characters'.format(VOCAB_SIZE))
    print("chars = {}".format(chars))

    ix_to_char = {ix: char for ix, char in enumerate(chars)}
    char_to_ix = {char: ix for ix, char in enumerate(chars)}

    X = np.zeros((math.ceil(len(data) / seq_length), seq_length, VOCAB_SIZE))
    y = np.zeros((math.ceil(len(data) / seq_length), seq_length, VOCAB_SIZE))

    print("for[0:" + str(math.floor(len(data) / seq_length)) + "]")

    for i in range(0, math.floor(len(data) / seq_length)):
        X_sequence = data[i * seq_length:(i + 1) * seq_length]
        X_sequence_ix = [char_to_ix[value] for value in X_sequence] #return ix value for each char in sequence
        input_sequence = np.zeros((seq_length, VOCAB_SIZE))

        for j in range(seq_length):
            input_sequence[j][X_sequence_ix[j]] = 1.
            X[i] = input_sequence

        if (i % 100) == 0:
            print(i, end=", ")

        y_sequence = data[i * seq_length + 1:(i + 1) * seq_length + 1]
        y_sequence_ix = [char_to_ix[value] for value in y_sequence]
        target_sequence = np.zeros((seq_length, VOCAB_SIZE))

        for j in range(seq_length):
            target_sequence[j][y_sequence_ix[j]] = 1.
            y[i] = target_sequence
    return X, y, VOCAB_SIZE, ix_to_char


# method for saving data
def save_pickle(name, data):
    import pickle
    data_dir = 'data/' + name + '.pickle'

    with open(data_dir, 'wb') as f:
        # Pickle the 'data' dictionary using the highest protocol available.
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)


# method for loading data
def load_pickle(name):
    import pickle
    data_dir = 'data/' + name + '.pickle'

    with open(data_dir, 'rb') as f:
        # The protocol version used is detected automatically, so we do not
        # have to specify it.
        data = pickle.load(f)
    return data


# Creating training data
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


# print("X_load= ")
# print(X_load[0:10], end=", ")
# print("y_load= ")
# print(y_load[0:10], end=", ")


# imports



# Creating and compiling the Network
print("\n------Creating and compiling the Network------")

model = Sequential()
model.add(LSTM(HIDDEN_DIM, input_shape=(None, VOCAB_SIZE), return_sequences=True))
for i in range(LAYER_NUM - 1):
    model.add(LSTM(HIDDEN_DIM, return_sequences=True))
model.add(TimeDistributed(Dense(VOCAB_SIZE)))
model.add(Activation('softmax'))
model.compile(loss="categorical_crossentropy", optimizer="rmsprop")

# ix_to_char = load_pickle('ix_to_char')
# VOCAB_SIZE = load_pickle('VOCAB_SIZE')

# Generate some sample before training to know how bad it is!
print("\n------Generate some sample before training to know how bad it is!------")
generate_text(model, GENERATE_LENGTH, VOCAB_SIZE, ix_to_char)

if not WEIGHTS == '':
    model.load_weights(WEIGHTS)
    nb_epoch = int(WEIGHTS[WEIGHTS.rfind('_') + 1:WEIGHTS.find('.')])
else:
    nb_epoch = 0

# X = load_pickle('X')
# y = load_pickle('y')
# ix_to_char = load_pickle('ix_to_char')
# VOCAB_SIZE = load_pickle('VOCAB_SIZE')


# Training if there is no trained weights specified
print("\n\n------Training if there is no trained weights specified------")
if MODE == 'train' or WEIGHTS == '':
    while True:
        print('\n\nEpoch: {}\n'.format(nb_epoch))
        model.fit(X, y, batch_size=BATCH_SIZE, verbose=1, epochs=1)
        nb_epoch += 1

        generate_text(model, GENERATE_LENGTH, VOCAB_SIZE, ix_to_char)

        # save every 10th epoch
        if nb_epoch % 1 == 0:
            model.save_weights('checkpoint_layer_{}_hidden_{}_epoch_{}.hdf5'.format(LAYER_NUM, HIDDEN_DIM, nb_epoch))

# Else, loading the trained weights and performing generation only
elif WEIGHTS == '':
    # Loading the trained weights
    model.load_weights(WEIGHTS)
    generate_text(model, GENERATE_LENGTH, VOCAB_SIZE, ix_to_char)
    print('\n\n')
else:
    print('\n\nNothing to do!')
