import re
import collections
from collections import Counter
import pickle

# text=("This account of the 'end of the? ?Third Age is drawn mainly Third Age from the Red Book of Westmarch. That most important source for the history of the War of the Ring was so called because it was long preserved at Undertowers, the home of the Fairbairns, Wardens of the Westmarch. 1 It was in origin Bilbo's private diary, which he took with him to Rivendell. Frodo brought it back to the Shire, together with many loose leaves of notes, and during S.R. 1420-1 he nearly filled its pages with his account of the War. But annexed to it and preserved with it, probably m a single red case,")
#
data_dir = 'resources/LOTR.txt'
text = open(data_dir, 'r').read()
text=text[0:1000000]
wordRegEx = re.compile("(?:[a-zA-Z']+)|(?:[,;:\\.!?])")
words = wordRegEx.findall(text)
words_org=words
next_lowercase=True

for i in range(len(words)):
    current_word=words[i]
    if next_lowercase: #new sentence
        words[i]=current_word.lower()

    if current_word in {".", "?", "!"}:
        next_lowercase=True
    else:
        next_lowercase=False

data=words

chars = list(set(words_org))
data_counter = Counter(data)
data_counter = sorted(data_counter.items(), key=lambda v: v[0].upper())

data_counter = [list(row) for row in data_counter]

# print(chars)
# print(data_counter)

for k in range (len(data_counter)):
    if str(data_counter[-k][0]).istitle(): #if current word is title - !!!here is the problem - list index out of range
        lower_current=data_counter[-k][0].lower()
        if data_counter[-k-1][0]==lower_current:
            if data_counter[-k][1]>=data_counter[-k-1][1]: #if uppercase is more often

                data_counter[-k][1]+=1
                # data_counter[-k-1][1]=0
                del data_counter[-k-1]
                print(data_counter[-k][0])

words_file=words_org
counter_file=data_counter

print(data_counter)

pkl_file1=open('words_file.pickle','wb')
pickle.dump(words_file,pkl_file1)

pkl_file2=open('counter_file.pickle','wb')
pickle.dump(counter_file,pkl_file2)


