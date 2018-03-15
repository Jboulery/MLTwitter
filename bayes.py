import pandas as pd
from sklearn.model_selection import train_test_split
from nltk.stem import snowball
from nltk.corpus import stopwords
import nltk
import emoji_list as moji

stemmer=snowball.FrenchStemmer()

data = pd.read_excel('TheVoiceTweets.xlsx')

data['text'] = data['extended_tweet.full_text'].fillna(data['text'])
data = data.drop(columns=['extended_tweet.full_text'])

train, test = train_test_split(data,test_size = 0.1)

train = train.dropna() # getting rid of the not-annotated rows
train = train[train.sentiment != 'zero']

train_pos = train[ train['sentiment'] == 'positive']
train_pos = train_pos['text']
train_neg = train[ train['sentiment'] == 'negative']
train_neg = train_neg['text']

tweets = []


stopwords_set = set(stopwords.words('french'))
stopwords_set.add('les')
stopwords_set.remove('pas')

emojis = moji.all_emoji
names=["nikos", "aliagas", "zazie", "zazi", "obispo", "pascal", "mika", "florent", "pagny", "matt", "pokora", "mpokora", "m.pokora", "drea", "dury", "édouard", "edouard", "solene", "solène", "solen", "solenne", "solenn", "ecco", "aurélien", "aurel", "aurelien", "rebecca", "kriill", "krill", "rihanna", "katy", "perry", "david", "bowie"]

def clean_tweet(tweet):
    cleaned = []
    emojis_used = []
    suitePas = False
    
    for word in tweet.split(): # for each word in the tweet
        
        if ('http' in word) or ('#' in word) or ('@' in word):
           continue
        
        word = word.lower()
        
        while "'" in word:
            word = word.split("'")[1] # "j'aime" ==> "aime"
        
        while "’" in word:
            word = word.split("’")[1] # "j’aime" ==> "aime"
            
        word = word.strip('!.,"?*0123456789') # "16ans???" ==> "ans"
        
        word = word.replace('\\n','') # "\\ngénial" ==> "génial"
        
        while (len(word) > 0) and (word[0] in emojis): # ":o:):)super:D" ==> ":o", ":)", "super:D"
            
            if word[0] not in emojis_used:
                emojis_used.append(word[0])
                cleaned.append(word[0])
            
            word = word[1:]
        
        while (len(word) > 0) and (word[-1] in emojis): # "super:D" ==> ":D", "super"
            
            if word[-1] not in emojis_used:
                emojis_used.append(word[-1])
                cleaned.append(word[-1])
            
            word = word[:-1]
        
        if (len(word) > 2) and (word not in stopwords_set) and (word not in names):
            if suitePas: # "pas", "aimé" ==> "pas aim"
                cleaned.append("pas " + stemmer.stem(word))
                suitePas = False
                
            elif word == "pas":
                suitePas = True
                
                if cleaned: # "aim", "pas" ==> "aim pas"
                    cleaned.append(cleaned.pop() + " " + word)
                
            else: # "aime" ==> "aim"
                cleaned.append(stemmer.stem(word))
        
    return cleaned

for index, row in train.iterrows(): # for each tweet
    cleaned = clean_tweet(row.text)
    if cleaned:
        tweets.append((cleaned,row.sentiment))
   
print(tweets)

test_pos = test[ test['sentiment'] == 'positive']
test_pos = test_pos['text']
test_neg = test[ test['sentiment'] == 'negative']
test_neg = test_neg['text']

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

neg_cnt = 0
pos_cnt = 0
for obj in test_neg: 
    res =  classifier.classify(extract_features(obj.split()))
    if(res == 'negative'): 
        neg_cnt = neg_cnt + 1
for obj in test_pos: 
    res =  classifier.classify(extract_features(obj.split()))
    if(res == 'positive'): 
        pos_cnt = pos_cnt + 1

print()
print('[negative]: %s/%s '  % (neg_cnt, len(test_neg)))
print('[positive]: %s/%s '  % (pos_cnt, len(test_pos)))
print()

def classify_tweet(tweet):
    print(tweet, "[", classifier.classify(extract_features(clean_tweet(tweet))), "]")
    return 0

classify_tweet("C'est nul")
classify_tweet("C'est pas nul")
classify_tweet("C'est bien")
classify_tweet("C'est pas bien")
classify_tweet("Ce soir on regarde The Voice")
classify_tweet("Je n'ai pas d'avis")
classify_tweet("J'ai aimé")
classify_tweet("Je n'ai pas aimé")
classify_tweet("Je suis fan")
classify_tweet("Je ne suis pas fan")