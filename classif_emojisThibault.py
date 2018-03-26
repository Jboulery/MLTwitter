import pandas as pd
from string import printable
from sklearn.model_selection import train_test_split
import nltk
import emoji_list as moji

data = pd.read_excel('TheVoiceTweets.xlsx')

data['text'] = data['extended_tweet.full_text'].fillna(data['text'])
data = data.drop(columns=['extended_tweet.full_text']) # fusion de la colonne text avec la colonne full_text

train, test = train_test_split(data,test_size = 0.1) # séparation des données en ensemble de train / ensemble de test (90%, 10%)

train = train.dropna() # on se débarrasse des lignes qui n'ont pas d'annotation
train = train[train.sentiment != 'zero'] # l'apprentissage ne se fait que sur des tweets positifs ou négatifs

train_pos = train[ train['sentiment'] == 'positive']
train_pos = train_pos['text']
train_neg = train[ train['sentiment'] == 'negative']
train_neg = train_neg['text']

tweets = []

emojis = set(moji.all_emoji)
for char in printable: # on retire les caractères de la liste d'emojis
    emojis.discard(char)
               
def clean_tweet(tweet): # Entrée : un tweet de type string. Sortie : une liste de couples : (liste des emojis du tweet, sentiment du tweet) 
    cleaned = []
    emojis_used = []
    
    for word in tweet.split(): # pour chaque mot du tweet
        word = word.strip(printable) #retirer tous les caractères du mot (on ne garde que les emojis)

        while (len(word) > 0) and (word[0] in emojis): # si un emoji se trouve au début du mot
            if word[0] not in emojis_used:
                emojis_used.append(word[0])
                cleaned.append(word[0])
            
            word = word[1:]
        
        while (len(word) > 0) and (word[-1] in emojis): # si un emoji se trouve à la fin du mot
            if word[-1] not in emojis_used:
                emojis_used.append(word[-1])
                cleaned.append(word[-1])
            
            word = word[:-1]
        
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
        if (word[0] in emojis) or (word[-1] in emojis):
            return classifier.classify(extract_features(tweet)) # Retourne la classification du tweet ('positive' ou 'negative') s'il y a au moins 1 emoji dans le tweet
    return False # Retourne False s'il n'y a aucun emoji dans le tweet


# ------------------------------- Test du classifieur ---------------------------------
test_pos = test[ test['sentiment'] == 'positive']
test_pos = test_pos['text']
test_neg = test[ test['sentiment'] == 'negative']
test_neg = test_neg['text']

neg_cnt = 0
pos_cnt = 0
total_neg = 0
total_pos = 0

# Test sur tweets annotés négativement
for tweet in test_neg:
    classification = classify_tweet(tweet)
    if classification: # si le tweet a au moins un emoji
        total_neg += 1
        if classification == 'negative': 
            neg_cnt += 1

# Test sur les tweets annotés positivement
for tweet in test_pos:
    classification = classify_tweet(tweet)
    if classification:
        total_pos += 1
        if classification == 'positive': 
            pos_cnt += 1

print()
print('[negative]: %s/%s '  % (neg_cnt, total_neg))
print('[positive]: %s/%s '  % (pos_cnt, total_pos))
print()
# ---------------------------- Fin de test du classifieur ------------------------------