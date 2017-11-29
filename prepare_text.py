import re
import collections
from collections import Counter
import pickle
import csv
import numpy as np
import operator
from operator import itemgetter

# text=("This account of the 'end of the? 'Third' Age is of drawn mainly Third Age from the Red Book of Westmarch. That most important source for the history of the War of the Ring was so called because it was long preserved at Undertowers, the home of the Fairbairns, Wardens of the Westmarch. 1 It was in origin Bilbo's private diary, which he took with him to Rivendell. Frodo brought it back to the Shire, together with many loose leaves of notes, and during S.R. 1420-1 he nearly filled its pages with his account of the War. But annexed to it and preserved with it, probably m a single red case,")
# text=("Taking in folk in off-hand like Frodo and eating extra food, and all that, said Hob. Frodo 'What's the matter with the place? Frodo?' said Merry. 'Frodo has it been a bad year, or what?")
data_dir = 'resources/LOTR.txt'
text = open(data_dir, 'r').read()
text=text[0:1000]
wordRegEx = re.compile("(?:[a-zA-Z]+)|(?:[,;:.!'´])")
words = wordRegEx.findall(text)
words_org = words  # will be used in the end for saving


def main(): # convert all words after end of sentence to lowercase.
    a = [[0, 0, 0, 0]]  # [value, idx, freq, final idx]
    words_file = [[0, 0]]
    outText = [[0, 0]]

    for xx in range(len(words) - 1):
        a.append([0, 0, 0, 0])
        words_file.append([0, 0])
        outText.append([0, 0])

    next_lowercase = True
    for i in range(len(words)):
        current_word = words[i]
        if next_lowercase:  # new sentence
            words[i] = current_word.lower()

        if current_word in {".", "?", "!", "'"}:
            next_lowercase = True
        else:
            next_lowercase = False

        a[i][0] = str(words[i])
        a[i][1] = i
        words_file[i][0] = str(words[i])
        words_file[i][1] = i




    # Counter function - it is not possible to use normal Counter()
    a = sorted(a, key=lambda v: (v[0].upper(), v[0].islower()))
    xx = 0
    while True:
        first = a[-xx][0] #go from the end, take first word
        second = a[-xx - 1][0] #take second word
        if first == second: #if the same, take first one, second delete
            a[-xx][2] += 1 #every equal means frequency + 1
            words_file[a[-xx - 1][1]][1] = a[-xx][1] #save idx of deleted word to words_file
            del a[-xx - 1] # deleting
        else:
            a[-xx][2] += 1 # everytime, when there is no more equal words, give me freq. num for myself
            xx += 1
        if xx == len(a):
            break

    k = 0
    c = 0
    while k < len(a):  # length of array is flexible - decreasing
        word = str(a[-k][0])  # array has flexible length, that is why counting is from the end
        if word[:1].isupper():  # if first char is uppercase
            lower_current = a[-k][0].lower()  # test: change word to lowercase
            next = a[-k - 1][0]
            if next == lower_current:  # if value next to first one is the same, we have a pair of words
                if a[-k][2] >= a[-k - 1][2]:  # if uppercase is more often than lowercase


                    a[-k][2] += a[-k - 1][2]  # merge frequencies
                    # print(k-1, a[-k-1])
                    # print(k, a[-k])

                    for rr in range(len(words_file)):  # replace lowercase in word_file by uppercase
                        if str(words_file[rr][0]) == a[-k - 1][0]:
                            words_file[rr][0] = str(a[-k][0])
                            words_file[rr][1] = a[-k][1]
                    del a[-k - 1]  # delete lowercase
                    k = k + 1  # to compensate delete function
                    c = c + 1
            elif a[-k + 1][0] == lower_current:  # the same but for case uppercase word is on other side
                if a[-k][2] >= a[-k + 1][2]:  # if uppercase is more often than lowercase

                    a[-k][2] += a[-k - 1][2]
                    # print(k - 1, a[-k + 1])
                    # print(k, a[-k])

                    for rr in range(len(words_file)):
                        if str(words_file[rr][0]) == a[-k + 1][0]:  # if you will find lowercase
                            words_file[rr][0] = str(a[-k+1][0])  # replace it by uppercase
                            words_file[rr][1] = a[-k][1]
                    del a[-k + 1]  # delete lowercase
                    k = k + 1  # to compensate delete function
                    c = c + 1

        k += 1

    sorted_by_freq = sorted(a, key=itemgetter(2), reverse=True) #sort reverse by frequency of words

    for kk in range(len(sorted_by_freq)): #fill matrix by new idx
        a[kk][3] = kk

    for w in range(len(words_file)):
        idx1 = words_file[w][1]
        for x in range(len(sorted_by_freq)):
            if sorted_by_freq[x][1] == idx1:
                outText[w][0] = words_file[w][0]
                outText[w][1] = a[x][3]

    print("words_file lenght:", len(words_file))
    print("Final count of unique words:", len(sorted_by_freq))

    save_pickle('outText', outText)  # file with all splited words without changin of

    with open('prepared_text_output.txt', 'w', newline='\n') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(outText)

        # print(counter_file)
        # print(words)
        # print(words_file)
        # print(sorted_by_freq[2][0])


def save_pickle(name, data):
    import pickle
    data_dir = 'data/' + name + '.pickle'

    with open(data_dir, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__": main()
