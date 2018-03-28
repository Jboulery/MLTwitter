############## imports #####################
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
from sklearn.feature_selection import chi2
from pylab import barh,plot,yticks,show,grid,xlabel,figure
import pandas as pd
from sklearn.model_selection import train_test_split
from nltk.stem import snowball
from nltk.corpus import stopwords
import nltk
from gensim.models import Word2Vec
import itertools
import emoji_list as moji
from sklearn.datasets import load_files
import os
from string import printable
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.model_selection import train_test_split, cross_validate
from sklearn import metrics
from nltk.stem.snowball import SnowballStemmer
from sklearn.naive_bayes import MultinomialNB
from textblob import TextBlob
from textblob_fr import PatternTagger, PatternAnalyzer
import pandas as pd 




############## Classifieur des tweets basé sur les emojis ###############

os.chdir('C:/Users/Admin/Desktop/Dossier_Etudes/Option_info/projet_option')
data = pd.read_excel('TheVoiceTweets.xlsx')

data['text'] = data['extended_tweet.full_text'].fillna(data['text'])
data = data.drop(columns=['extended_tweet.full_text']) # fusion de la colonne text avec la colonne full_text

train, test = train_test_split(data,test_size = 0) # séparation des données en ensemble de train / ensemble de test (90%, 10%)

train = train.dropna() # on se débarrasse des lignes qui n'ont pas d'annotation
train = train[train.sentiment != 'zero'] # l'apprentissage ne se fait que sur des tweets positifs ou négatifs

tweets = []

emojis = set(moji.all_emoji)
for char in printable: # on retire les caractères de la liste d'emojis
    emojis.discard(char)
               
def clean_tweet(tweet): # Entrée : un tweet de type string. Sortie : une liste de couples : (liste des emojis du tweet, sentiment du tweet) 
    cleaned = []
    tweet.strip(printable) # retirer tous les caractères du tweet
    
    for word in tweet.split(): # pour chaque mot du tweet
        for char in word:
            if char not in cleaned and char in emojis: # si c'est la première fois que l'emoji apparait dans le tweet
                cleaned.append(char)
                
    return cleaned

#Constitution de la liste de tweets nettoyés, qui servira pour l'apprentissage
for index, row in train.iterrows(): # pour chaque tweet
    cleaned = clean_tweet(row.text)
    if cleaned:
        tweets.append((cleaned,row.sentiment))

def get_words_in_tweets(tweets):
    all = []
    for (words, sentiment) in tweets:
        all.extend(words)
    return all

def get_word_features(wordlist):
    wordlist = nltk.FreqDist(wordlist)
    features = wordlist.keys()
    return features
w_features = get_word_features(get_words_in_tweets(tweets))

def extract_features(document):
    document_words = set(document)
    features = {}
    for word in w_features:
        features['containts(%s)' % word] = (word in document_words)
    return features

training_set = nltk.classify.apply_features(extract_features,tweets)
classifier = nltk.NaiveBayesClassifier.train(training_set)

def classify_tweet(tweet): # Entrée : un tweet de type string. Sortie : "positive" ou "negative"
    for word in tweet.split():
        for char in word:
            if char in emojis: # Dès que l'on identifie un emoji dans le tweet
                return classifier.classify(extract_features(tweet)) # Retourne la classification du tweet ('positive' ou 'negative') s'il y a au moins 1 emoji dans le tweet
    return False # Retourne False s'il n'y a aucun emoji dans le tweet


################################## Identification des tweets neutres ######################

class NeutralClf():

    def __init__(self):

        def BayesClf():
            data_folder = 'data/tweets/'
            dataset = load_files(data_folder, shuffle=False, encoding="ISO-8859-1")
            print("n_samples: %d" % len(dataset.data))

            # Split the dataset in training and test set:
            docs_train, docs_test, y_train, y_test = train_test_split(
                dataset.data, dataset.target, test_size=0.25, random_state=12)
            kfolds = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)

            stopwords_set = set(stopwords.words('french'))
            stopwords_set.remove('pas')
            stemmer = SnowballStemmer('french', ignore_stopwords=True)

            class StemmedCountVectorizer(CountVectorizer):
                def build_analyzer(self):
                    analyzer = super(StemmedCountVectorizer, self).build_analyzer()
                    return lambda doc: ([stemmer.stem(w) for w in analyzer(doc)])

            stemmed_count_vect = StemmedCountVectorizer(stop_words=stopwords_set, ngram_range=(1 ,1))

            bayes_clf = Pipeline([
                ('vect', stemmed_count_vect),
                ('tfidf', TfidfTransformer(use_idf=False)),
                ('clf', MultinomialNB(fit_prior=False, alpha=0.1)),
                ])

            bayes_clf.fit(docs_train, y_train)

            scoring = ['precision', 'recall']

            cv_results = cross_validate(
                            bayes_clf, dataset.data, dataset.target, 
                            cv=kfolds, scoring=scoring, verbose=1
                        )

            print('CLF Bayes Results')
            print('Precision sur les folds : ')
            print(cv_results['test_precision'])
            print('Recall sur les folds : ')
            print(cv_results['test_recall'])
            print('-'*60)
            print()
            y_true, y_pred = y_test, bayes_clf.predict(docs_test)
            print(metrics.classification_report(y_true, y_pred))
            print()
            cm = metrics.confusion_matrix(y_true, y_pred)
            print(cm)
            
            return bayes_clf


        def SVMClf():
            data_folder = 'data/tweets/'
            dataset = load_files(data_folder, shuffle=False, encoding="ISO-8859-1")
            print("n_samples: %d" % len(dataset.data))

            docs_train, docs_test, y_train, y_test = train_test_split(
                dataset.data, dataset.target, test_size=0.25, random_state=12)

            stopwords_set = set(stopwords.words('french'))
            stopwords_set.remove('pas')
            stemmer = SnowballStemmer('french', ignore_stopwords=True)
            kfolds = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)

            class StemmedCountVectorizer(CountVectorizer):
                def build_analyzer(self):
                    analyzer = super(StemmedCountVectorizer, self).build_analyzer()
                    return lambda doc: ([stemmer.stem(w) for w in analyzer(doc)])

            stemmed_count_vect = StemmedCountVectorizer(stop_words=stopwords_set, ngram_range=(1,2))

            svm_clf = Pipeline([
                ('vect', stemmed_count_vect),
                ('tfidf', TfidfTransformer(use_idf=False)),
                ('clf', SVC(C=10, gamma=0.1, kernel='rbf')),
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
            print()
            cm = metrics.confusion_matrix(y_true, y_pred)
            print(cm)
            
            return svm_clf

        print('#'*60)
        print('#' + ' '*20 + 'Entrainement Bayes' + ' '*20 + '#')
        print('#'*60)

        self.bayes_clf = BayesClf()

        print('#'*60)
        print('#' + ' '*21 + 'Entrainement SVM' + ' '*21 + '#')
        print('#'*60)

        self.svm_clf = SVMClf()


    def is_neutral(self, text):
        btext = TextBlob(text, 
                        pos_tagger=PatternTagger(), 
                        analyzer=PatternAnalyzer())
        if btext.sentiment[0] == 0:
            if (self.bayes_clf.predict([text])[0] == 0) or (self.svm_clf.predict([text])[0] == 0):
                return True
        else:
            if (self.bayes_clf.predict([text])[0] == 0) and (self.svm_clf.predict([text])[0] == 0):
                return True
            return False


neutral_clf = NeutralClf()

def neutral_detection(BASE_DIR):
    count = 0
    neutral_count = 0
    for root, dirs, files in os.walk(BASE_DIR):
        for file in files:
            count += 1
            path = os.path.join(root, file)
            with open(path, 'r') as f:
                text = f.read()
                if neutral_clf.is_neutral(text):
                    neutral_count += 1
                        
    return str('{:0.2f}% neutres détectés'.format(neutral_count / count * 100))

print('Autres : ', neutral_detection('data/tweets/others'))
print('Neutres : ', neutral_detection('data/tweets/neutral'))

################################## Les 3 classifieurs : CNN, Bayes, SVM ###################

os.chdir('C:/Users/Admin/Desktop/Dossier_Etudes/Option_info/projet_option/MLTwitter')
dataset_folder = 'data/tweets_neg_pos'
dataset = load_files(dataset_folder, shuffle=False, encoding='ISO-8859-1')
docs_train, docs_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=0.25, random_state=12)

nndocs = docs_train
nndocs_test = docs_test
labels = y_train
labels_test = y_test

# cnn
vocab_size = 30000
encoded_docs = [one_hot(d, vocab_size) for d in nndocs]
# pad documents to a max length of 30 words
max_length = 30
padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')

#pour les test
encoded_tests = [one_hot(d, vocab_size) for d in nndocs_test]
padded_tests = pad_sequences(encoded_tests, maxlen=max_length, padding='post')


labels = utils.to_categorical(labels, 2)
labels_test = utils.to_categorical(labels_test, 2)

# define the model
model = Sequential()
model.add(Embedding(vocab_size, 30, input_length=max_length))#,weights=[embedding_matrix], trainable=False))

model.add(Conv1D(64, 3, border_mode='same'))
model.add(Conv1D(32, 3, border_mode='same'))
model.add(Conv1D(16, 3, border_mode='same'))

model.add(Flatten())

model.add(Dropout(0.2))
model.add(Dense(180,activation='sigmoid'))
model.add(Dropout(0.2))
model.add(Dense(2,activation='sigmoid'))

# compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
# summarize the model
print(model.summary())
# fit the model 

model.fit(padded_docs, labels, validation_data=(padded_tests,labels_test), epochs=50, verbose=0) 
# evaluate the model
loss, accuracy = model.evaluate(padded_tests, labels_test, verbose=0)
print('Accuracy on test: %f' % (accuracy*100))

    
def predire(c_list):
    out = []
    if type(c_list) == str:
        new_k = one_hot(c_list,vocab_size)
        new_kk = pad_sequences([new_k],maxlen=max_length,padding='post')
        out.append(model.predict(new_kk))
    else:
        for x in c_list:
            new_k = one_hot(x,vocab_size)
            new_kk = pad_sequences([new_k],maxlen=max_length,padding='post')
            out.append(model.predict(new_kk))
    return out

y_pred = model.predict(padded_tests)
y_pred = [1 if x[0] <= x[1] else 0 for x in y_pred]
labels_t = [1 if x[0] <= x[1] else 0 for x in labels_test]

def Confusion(y_pred,labels_t):
    results = []
    for i in range(len(y_pred)):
        x = y_pred[i] - labels_t[i]
        if x == 0: 
            if y_pred[i] == 1: #TP
                results.append(3)
            else: #TN
                results.append(0)
        elif x == -1: #FN
            results.append(2)
        elif x == 1: #FP
            results.append(1)
    
    TP = sum([1 if x == 3 else 0 for x in results])
    TN = sum([1 if x == 0 else 0 for x in results])
    FP = sum([1 if x == 1 else 0 for x in results])
    FN = sum([1 if x == 2 else 0 for x in results])
    conf_mat = [[TP,FN],[FP,TN]]
    confusion = np.asarray(conf_mat)
    return confusion

def plot_most_significant(labels):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(nndocs)
    chi2score = chi2(X,labels)[0]
    figure(figsize=(6,6))
    wscores = list(zip(vectorizer.get_feature_names(),chi2score))
    wchi2 = sorted(wscores,key=lambda x:x[1]) 
    topchi2 = list(zip(*wchi2[-10:]))
    x = [i for i in range(len(topchi2[1]))]
    label = topchi2[0]
    barh(x,topchi2[1],align='center',alpha=.2,color='g')
    plot(topchi2[1],x,'-o',markersize=2,alpha=.8,color='g')
    yticks(x,label)
    xlabel('$\chi^2$')
    show()

# svm
y_svm_pred, labels_svm = list(), list()

class SVMClf():
    def __init__(self):
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

# bayes
y_bayes_pred, labels_bayes = list(), list()

class BayesClf():

    def __init__(self):
        kfolds = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)

        #min_df is the minimal frequency of a word
        stopwords_set = set(stopwords.words('french'))
        stopwords_set.remove('pas')
        stemmer = SnowballStemmer('french', ignore_stopwords=True)

        class StemmedCountVectorizer(CountVectorizer):
            def build_analyzer(self):
                analyzer = super(StemmedCountVectorizer, self).build_analyzer()
                return lambda doc: ([stemmer.stem(w) for w in analyzer(doc)])

        stemmed_count_vect = StemmedCountVectorizer(stop_words=stopwords_set, ngram_range=(1 ,1))

        ##########################################################
        #                 Classifier Definition                  #
        ##########################################################

        #1
        #                precision   recall  f1-score   support

        #           0       0.69      0.79      0.73       182
        #           1       0.89      0.83      0.86       378

        # avg / total       0.82      0.81      0.82       560

        # Confusion Matrix
        # [[144  38]
        #  [ 66 312]]

        bayes_clf = Pipeline([
            ('vect', stemmed_count_vect),
            ('tfidf', TfidfTransformer(use_idf=False)),
            ('clf', MultinomialNB(fit_prior=False, alpha=0.1)),
            ])

        bayes_clf.fit(docs_train, y_train)

        scoring = ['precision', 'recall']

        cv_results = cross_validate(
                        bayes_clf, dataset.data, dataset.target, 
                        cv=kfolds, scoring=scoring, verbose=1
                    )

        print('CLF Bayes Results')
        print('Precision sur les folds : ')
        print(cv_results['test_precision'])
        print('Recall sur les folds : ')
        print(cv_results['test_recall'])
        print('-'*60)
        print()
        y_true, y_pred = y_test, bayes_clf.predict(docs_test)
        print(metrics.classification_report(y_true, y_pred))
        print()
        cm = metrics.confusion_matrix(y_true, y_pred)
        print(cm)
        self.clf = bayes_clf
        self.target_classes = dataset.target_names

    def predict(self, text):
        return self.target_classes[int(self.clf.predict([text])[0])]
        
bayes = BayesClf()
svm = SVMClf()
y_svm_pred = [svm.predict(x) for x in list(docs_test)]
y_bayes_pred = [bayes.predict(x) for x in list(docs_test)]

y_svm_pred = [1 if x == 'pos' else 0 for x in y_svm_pred]
y_bayes_pred = [1 if x == 'pos' else 0 for x in y_bayes_pred]

predictions = []
for i in range(len(y_pred)):
    x,y,z = y_pred[i],y_svm_pred[i],y_bayes_pred[i]
    if x + y + z >= 2:
        predictions.append(1)
    elif x + y + z <= 1 :
        predictions.append(0)
        
# # # Confusion(predictions,y_test)
# # # array([[352,  26],
# # #        [ 82, 100]])

############# predict a new tweet or set of tweets ############

def predict_tweets(L):
    if type(L) == str:
        flag = True
        if classify_tweet(L) == 'negative':
            flag= False
            return 0
        if flag == True:
            x,y,z = int, int, int
            if predire(L)[0][0][0] > predire(L)[0][0][1]:
                x = 0
            else:
                x = 1
            if bayes.predict(L) == 'pos':
                y = 1
            else:
                y = 0
            if svm.predict(L) == 'pos':
                z = 1
            else:
                z = 0
            if x + y + z >= 2:
                return 1
            elif x + y + z <= 1 :
                return 0
    if type(L) == list:
        results = []
        for k in L:
            flag2 = True
            if classify_tweet(k) == 'negative':
                flag2 = False
                results.append(0)
            if flag2 == True:
                x,y,z = int, int, int
                if predire(k)[0][0][0] > predire(k)[0][0][1]:
                    x = 0
                else:
                    x = 1
                if bayes.predict(k) == 'pos':
                    y = 1
                else:
                    y = 0
                if svm.predict(k) == 'pos':
                    z = 1
                else:
                    z = 0
                if x + y + z >= 2:
                    results.append(1)
                elif x + y + z <= 1 :
                    results.append(0)
                
        return results

############## popularité ######################

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
        ypred = predict_tweets('%s' % (tweet))

        if (ypred == 0):
               sentiment.append('positive')
        elif(ypred == 1):
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

"""test results SVM [PRIME 24 Mars]
{'Assia': -1.0, 'Norig': -0.273, 'meryem': -0.333, 'Ecco': 0.36, 'Xam': -0.098, 'rebecca': -1.0, 'Capucine': 0.047, 'Liv': -0.18, 'yvette': 0.286, 
'kriil': -0.333, 'Gulaan': 0.239, 'Gabriel': 0.191, 'leho': 0.25, 'drea': 0.15}
"""