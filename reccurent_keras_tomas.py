import math as math
# from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import time
import csv
import pickle
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM, SimpleRNN
from keras.layers.wrappers import TimeDistributed


# import argparse
# from RNN_utils import *


# Parsing arguments for Network definition

# DATA_DIR = 'resources/DelverMagic_lowercase.txt'
DATA_DIR = 'resources/DelverMagic_lowercase.txt'
SEQ_LENGTH = 50
BATCH_SIZE = 50
LAYER_NUM = 2
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

    data="ty-nine they began to call him we//-preserved, but unchanged would have been nearer the mark. There were some that shook their heads and thought this was too much of a good thing, it seemed unfair that anyone should possess (apparently) perpetual youth as well as (reputedly) inexhaustible wealth. 'It will have to be paid for,' they said. 'It isn't natural, and trouble will come of it!' But so far trouble had not come, and as Mr. Baggins was generous with his money, most people were willing to forgive him his oddities and his good fortune. He remained on visiting terms with his relatives (except, of course, the Sackville-Bagginses), and he had many devoted admirers among the hobbits of poor and unimportant families. But he had no close friends, until some of his younger cousins began to grow up. The eldest of these, and Bilbo's favourite, was young Frodo Baggins. When Bilbo was ninety-nine, he adopted Frodo as his heir, and brought him to live at Bag End, and the hopes of the Sackville-Bagginses were finally dashed. Bilbo and Frodo happened to have the same birthday, September 22nd. 'You had better come and live here, Frodo my lad,' said Bilbo one day, 'and then we can celebrate our birthday-parties comfortably together.' At that time Frodo was still in his tweens, as the hobbits called the irresponsible twenties between childhood and coming of age at thirty-three. Twelve more years passed. Each year the Bagginses had given very lively combined birthday-parties at Bag End, but now it was understood that something quite exceptional was being planned for that autumn. Bilbo was going to be eleventy-one, 111, a rather curious number and a very respectable age for a hobbit (the Old Took himself had only reached 130), and Frodo was going to be thirty-three, 33) an important number: the date of his 'coming of age'. Tongues began to wag in Flobbiton and By water, and rumour of the coming event travelled all over the Shire. The history and character of Mr. Bilbo Baggins became once again the chief topic of conversation, and the older folk suddenly found their reminiscences in welcome demand. No one had a more attentive audience than old Ham Gamgee, commonly known as the Gaffer. He held forth at The Ivy Bush , a small inn on the Bywater road, and he spoke with some authority, for he had tended the garden at Bag End for forty years, and had helped old Holman in the same job before that. Now that he was himself growing old and stiff in the joints, the job was mainly carried on by his youngest son, Sam Gamgee. Both father and son were on very friendly terms with Bilbo and Frodo. They lived on the Hill itself, in Number 3 Bagshot Row just below Bag End. ’A very nice well-spoken gentlehobbit is Mr. Bilbo, as I've always said,' the Gaffer declared. With perfect truth: for Bilbo was very polite to him, calling him 'Master Hamfast', and consulting him constantly upon the growing of vegetables - in the matter of 'roots', especially potatoes, the Gaffer was recognized as the leading authority by all in the neighbourhood (including himself). 'But what about this Frodo that lives with him?' asked Old Noakes of By water. 'Baggins is his name, but he's more than half a Brandybuck, they say. It beats me why any Baggins of Hobbiton should go looking for a wife away there in Buckland, where folks are so queer.' 'And no wonder they're queer,' put in Daddy Twofoot (the Gaffer's next-door neighbour), 'if they live on the wrong side of the Brandywine River, and right agin the Old Forest. That's a dark bad place, if half the tales be true.' 'You're right, Dad!' said the Gaffer. 'Not that the Brandybucks of Buck-land live in the Old Forest, but they're a queer breed, seemingly. They fool about with boats on that big river - and that isn’t natural. Small wonder that trouble came of it, I say. But be that as it may, Mr. Frodo is as nice a young hobbit as you could wish to meet. Very much like Mr. Bilbo, and in more than looks. After all his father was a Baggins. A decent respectable hobbit was Mr. Drogo Baggins, there was never much to tell of him, till he was drownded.' 'Drownded?' said several voices. They had heard this and other darker rumours before, of course, but hobbits have a passion for family history, and they were ready to hear it again. 'Well, so they say,' said the Gaffer. 'You see: Mr. Drogo, he married poor Miss Primula Brandybuck. She was our Mr. Bilbo's first cousin on the mother's side (her mother being the youngest of the Old Took's daughters), and Mr. Drogo was his second cousin. So Mr. Frodo is his first and second cousin, once removed either way, as the saying is, if you follow me. And Mr. Drogo was staying at Brandy Flail with his father-in-law, old Master Gorbadoc, as he often did after his marriage (him being partial to his vittles, and old Gorbadoc keeping a mighty generous table), and he went out boating on the Brandywine River, and he and his wife were drownded, and poor Mr. Frodo only a child and all. ' 'I've heard they went on the water after dinner in the moonlight,' said Old Noakes, 'and it was Drogo's weight as sunk the boat.' 'And / heard she pushed him in, and he pulled her in after him,' said Sandy man, the Hobbiton miller. 'You shouldn't listen to all you hear, Sandyman,' said the Gaffer, who did not much like the miller. 'There isn't no call to go talking of pushing and pulling. Boats are quite tricky enough for those that sit still without looking further for the cause of trouble. Anyway: there was this Mr. Frodo left an orphan and stranded, as you might say, among those queer Bucklanders, being brought up anyhow in Brandy Hall. A regular warren, by all accounts. Old Master Gorbadoc never had fewer than a couple of hundred relations in the place. Mr. Bilbo never did a kinder deed than when he brought the lad back to live among decent folk. 'But I reckon it was a nasty shock for those Sackville-Bagginses. They thought they were going to get Bag End, that time when he went off and was thought to be dead. And then he comes back and orders them off, and he goes on living and living, and never looking a day older, bless him! And suddenly he produces an heir, and has all the papers made out proper. The Sackville-Bagginses won’t never see the inside of Bag End now, or it is to be hoped not.' 'There's a tidy bit of money tucked away up there, I hear tell,' said a stranger, a visitor on business from Michel Delving in the Westfarthing. 'All the top of your hill is full of tunnels packed with chests of gold and silver, and jools, by what I've heard. ' 'Then you've heard more than I can speak to,' answered the Gaffer. I know nothing about jools. Mr. Bilbo is free with his money, and there seems no lack of it, but I know of no tunnel -making. I saw Mr. Bilbo when he came back, a matter of sixty years ago, when I was a lad. I'd not long come prentice to old Holman (him being my dad's cousin), but he had me up at Bag End helping him to keep folks from trampling and trapessing all over the garden while the sale was on. And in the middle of it all Mr. Bilbo comes up the Hill with a pony and some mighty big bags and a couple of chests. I don't doubt they were mostly full of treasure he had picked up in foreign parts, where there be mountains of gold, they say, but there wasn't enough to fill tunnels. But my lad Sam will know more about that. He's in and out of Bag End. Cr"
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


print("X_load= ")
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
