############# SVM for prediction ########################

import sys
import nltk
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split, cross_validate
from sklearn import metrics
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import os


os.chdir('D:/Master_Centrale/S2/TwitterML')

class SVMClf():

    def __init__(self):
        # The training data folder must be passed as first argument
        data_folder = 'data/tweets_neg_pos/'
        dataset = load_files(data_folder, shuffle=False, encoding='ISO-8859-1')
        print("n_samples: %d" % len(dataset.data))

        # Split the dataset in training and test set:
        docs_train, docs_test, y_train, y_test = train_test_split(
            dataset.data, dataset.target, test_size=0.25, random_state=12)

        #min_df is the minimal frequency of a word
        stopwords_set = set(stopwords.words('french'))
        stopwords_set.remove('pas')
        stemmer = SnowballStemmer('french', ignore_stopwords=True)
        kfolds = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)

        class StemmedCountVectorizer(CountVectorizer):
            def build_analyzer(self):
                analyzer = super(StemmedCountVectorizer, self).build_analyzer()
                return lambda doc: ([stemmer.stem(w) for w in analyzer(doc)])

        stemmed_count_vect = StemmedCountVectorizer(stop_words=stopwords_set, ngram_range=(1,3))
        
        svm_clf = Pipeline([
            ('vect', stemmed_count_vect),
            ('tfidf', TfidfTransformer(use_idf=False)),
            ('clf', SVC(C=10, gamma=0.1, kernel='linear')),
            ])

        svm_clf.fit(docs_train, y_train)

        scoring = ['precision', 'recall']

        cv_results = cross_validate(
                        svm_clf, dataset.data, dataset.target, 
                        cv=kfolds, scoring=scoring, verbose=1
                    )

        print('CLF svm.SVC() Results')
        print('Precision sur les folds : ')
        print(cv_results['test_precision'])
        print('Recall sur les folds : ')
        print(cv_results['test_recall'])
        print('-'*60)
        print()
        y_true, y_pred = y_test, svm_clf.predict(docs_test)
        print(metrics.classification_report(y_true, y_pred))
        cm = metrics.confusion_matrix(y_true, y_pred)
        print(cm)
        self.clf = svm_clf
        self.target_classes = dataset.target_names

            #               precision    recall  f1-score   support

            #           0       0.70      0.71      0.71       182
            #           1       0.86      0.85      0.86       378

            # avg / total       0.81      0.81      0.81       560

    def predict(self, text):
        return (self.target_classes[int(self.clf.predict([text])[0])])

svm = SVMClf()
print(svm.predict('Coucou ce fut un moment magique !'))


################ BEGIN SCORING ########################

import pandas as pd 
import numpy as np



#load and extract data
def extract():
    data = pd.read_csv('./twitter.Prime24Mars.csv')
    data['text'] = data['extended_tweet.full_text'].fillna(data['text'])
    data.drop(['extended_tweet.full_text'], axis=1, inplace=True)
    data = data[data['text'].isnull() == False]
    data.reset_index(inplace=True)
    data = data.drop(['index'],axis=1)

    print ('dataset loaded with shape', data.shape)    
    return data

data = extract()

# sentiment analysis and add 'sentiment' column

sentiment = []
for tweet in data['text']:
        y_pred = predict_tweets('%s' % (tweet))

        if (y_pred == 0):
               sentiment.append('positive')
        elif(y_pred == 1):
               sentiment.append('negative')

data['sentiment'] = sentiment

candidats = [["Ecco","ecco","eco","piano"],
             ["Xam","xam","hurricane","huricane"],
             ["Liv","liv","del","estal"],
             ["Gabriel","gabriel","grabiell","voiture"],
             ["rebecca","rébécca","rebeca","Rebecca"],
             ["Gulaan","gulan","gulaan","gulann","gullan","gullaan","polynésien","polynesie","polynésienne","Gulan"],
             ["drea","dréa","dury","duri","dreaduri","dreadury","colombienne","colombie","Drea"],
             ["kriil","kril","krill","kriill","kezako"],
             ["yvette","dantier","ivette","yvett","yvete","maurice"],
             ["meryem","meriem","Meriem","mariam","miriam"],
             ["Assia","assya","acia","assia"],
             ["Capucine","capucine","capussine","nantes"],
             ["leho","leo"],
             ["Norig","norig"]]

#loop over candidates 

cand_score = {}
len_tweet = {}

for cand in candidats:
        for i in range(len(cand)):
                tweet = data[data['text'].str.contains(cand[i])]
                pos_tweets = tweet.loc[tweet['sentiment'] == 'positive']
                neg_tweets = tweet.loc[tweet['sentiment'] == 'negative']
                
                
                pos_score = 0
                neg_score = 0
                
                for j in range(len(pos_tweets)):
                    pos_score += 1

                for j in range(len(neg_tweets)):
                    neg_score += -1
                
                tweet_len = len(tweet)
                len_tweet [cand[i]] = tweet_len

                value = pos_score + neg_score
                cand_score[cand[i]] = value



new_dict = {}

for i in range(len(candidats)):
    new_score = 0
    length = 0
    for key,value in cand_score.items():
            if (key in candidats[i]):
                new_score += value
    for key1,value1 in len_tweet.items():
            if (key1 in candidats[i]):
                length += value1
    
    new_dict[candidats[i][0]] = round(new_score/length if length != 0 else 0,3)

print(new_dict)
            

            
################ END SCORING ########################

"""test results
{'Assia': -1.0, 'Norig': -0.273, 'meryem': -0.333, 'Ecco': 0.36, 'Xam': -0.098, 'rebecca': -1.0, 'Capucine': 0.047, 'Liv': -0.18, 'yvette': 0.286, 
'kriil': -0.333, 'Gulaan': 0.239, 'Gabriel': 0.191, 'leho': 0.25, 'drea': 0.15}
"""