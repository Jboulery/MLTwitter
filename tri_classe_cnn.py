from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
from keras.layers import LSTM, Conv1D, Flatten, Dropout, Dense
from keras import utils
import numpy as np
from nltk.stem.snowball import FrenchStemmer
stemmer = FrenchStemmer()
from pprint import pprint


####################### define documents and labels extracted from corpus ####################


import os

ann_txt_directory = "C:/Users/Admin/Desktop/Dossier_Etudes/Option_info/projet_option/training_set/canephore/canephore-corpus/existing_ann_with_text"
os.chdir(ann_txt_directory)

annotations = list()
docs = list()
for file in os.listdir(ann_txt_directory):
    filename = os.fsdecode(file)
    a = str(filename)
    if a[-4:] == ".ann":
        f = open(file,'r')
        p = f.readlines()
        phrases = [[x] for x in p]
        words = list()
        for phrase in phrases:
            word = phrase[0].split()
            words += word
        if 'Subjectiveme_positive' in words:
            if 'Negates' in words:
                annotations.append('negative')
            else:
                annotations.append('positive')
        elif 'Subjectiveme_negative' in words and 'Subjectiveme_positive' not in words:
            if 'Negates' in words:
                annotations.append('positive')
            else:
                annotations.append('negative')
        else:
            annotations.append('neutral')
    else:
        f = open(file,'r',encoding='utf-8')
        tweet = ''
        interim = f.readlines()
        interim2 = [x.split() for x in interim]
        for x in interim2:
            for i in range(len(x)):
                tweet += ' '+x[i]
        docs.append(tweet)

labels = list()
for x in annotations:
    if x == 'positive':
        labels.append(2)
    elif x == 'negative':
        labels.append(0)
    else:
        labels.append(1)

####################### start embedding and modelling #############################

ndocs = []
#stemming
for i in range(len(docs)):
    ligne = docs[i].split()
    st_ligne = [stemmer.stem(x) for x in ligne]
    ndoc_i = ' '.join(st_ligne)
    ndocs.append(ndoc_i)

# integer encode the documents
vocab_size = 5000
encoded_docs = [one_hot(d, vocab_size) for d in ndocs]
#print(encoded_docs)
# pad documents to a max length of 30 words
max_length = 30
padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
#print(padded_docs)

labels = utils.to_categorical(labels, 3)

# define the model
model = Sequential()
model.add(Embedding(vocab_size, 32, input_length=max_length))

model.add(Conv1D(64, 3, border_mode='same'))
model.add(Conv1D(32, 3, border_mode='same'))
model.add(Conv1D(16, 3, border_mode='same'))

model.add(Flatten())

model.add(Dropout(0.2))
model.add(Dense(180,activation='sigmoid'))
model.add(Dropout(0.2))
model.add(Dense(3,activation='sigmoid'))

# compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
# summarize the model
print(model.summary())
# fit the model
model.fit(padded_docs, labels, epochs=50, verbose=0)
# evaluate the model
loss, accuracy = model.evaluate(padded_docs, labels, verbose=0)
print('Accuracy: %f' % (accuracy*100))

def adapt_size(l,s=30):
    if len(l) < s:
        nl = list()
        m = len(l)
        for x in l:
            nl.append(x)
        for i in range(s - m):
            nl.append(0)
    else:
        nl = l
    return np.asarray(nl).reshape(1,30)
    
def predire(c):
    new_k = one_hot(c,vocab_size)
    new_kk = adapt_size(new_k)
    return model.predict(new_kk)



