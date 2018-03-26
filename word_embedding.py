import pandas as pd 
import numpy as np
import string

import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')


import gensim
from gensim.models.word2vec import Word2Vec # the word2vec model gensim class
from gensim.models.doc2vec import TaggedDocument


from nltk.tokenize import TweetTokenizer # a tweet tokenizer from nltk.
tokenizer = TweetTokenizer()

from sklearn.model_selection import train_test_split


#load and extract data
def extract():
    data = pd.read_csv('./TheVoiceTweets.csv')
    data.drop(['extended_tweet.full_text'], axis=1, inplace=True)
    data = data[data.sentiment.isnull() == False]
    data['sentiment'] = data['sentiment'].map(str)
    data = data[data['text'].isnull() == False]
    data.reset_index(inplace=True)
    print ('dataset loaded with shape', data.shape)    
    return data

data = extract()
print(data.head(5))

string_punctuation =  ".@#,_;http"

def clear_punctuation(s):
    clear_string = ""
    for symbol in s:
        if symbol not in string.punctuation:
            clear_string += symbol
    return clear_string

#tokenize tweets
def tokenize(tweet):
    #try:
        tweet = str(tweet.encode('utf-8').lower())
        tweet = clear_punctuation(tweet)
        tokens = tokenizer.tokenize(tweet)
        return tokens
    #except:
        return 'NC'

    
print(tokenize("#NTX #TheVoice La saison 7 dans un instant sur"))

#Remove lines with NC
def postprocess(data, n=999):
    data = data.head(n)
    data['tokens'] = data['text'].map(tokenize)  
    data = data[data.tokens != 'NC']
    data.reset_index(inplace=True)
    data.drop('index', inplace=True, axis=1)
    return data

data = postprocess(data)

#build the model
x_train, x_test, y_train, y_test = train_test_split(np.array(data.tokens),
                                                    np.array(data.sentiment), test_size=0.2)
													
#labelize tokens 

def labelizeTweets(tweets, label_type):
    labelized = []
    for i,v in enumerate(tweets):
        label = '%s_%s'%(label_type,i)
        labelized.append(TaggedDocument(v, [label]))
    return labelized

x_train = labelizeTweets(x_train, 'TRAIN')
x_test = labelizeTweets(x_test, 'TEST')

tweet_w2v = Word2Vec(size=10, min_count=1)
tweet_w2v.build_vocab([x.words for x in x_train])
tweet_w2v.train([x.words for x in x_train],total_examples=tweet_w2v.corpus_count, epochs=tweet_w2v.epochs)

#test
tweet_w2v.wv['france']
tweet_w2v.wv.most_similar('france')