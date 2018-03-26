import pandas as pd
from string import printable
from sklearn.model_selection import train_test_split
import nltk
import emoji_list as moji

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

emojis = set(moji.all_emoji)
emojis.discard('#')

for char in printable:
    emojis.discard(char)
               
def clean_tweet(tweet):
    cleaned = []
    emojis_used = []
    
    for word in tweet.split(): # for each word in the tweet
        if ('http' in word) or ('#' in word) or ('@' in word):
           continue
       
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
        
    return cleaned

for index, row in train.iterrows(): # for each tweet
    cleaned = clean_tweet(row.text)
    if cleaned:
        tweets.append((cleaned,row.sentiment))

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
total_neg = 0
total_pos = 0
print("\n----- Negative tweets with emojis -----\n")
for obj in test_neg:
    for word in obj.split():
        if (word[0] in emojis) or (word[-1] in emojis):
            print(obj)
            total_neg += 1
            res =  classifier.classify(extract_features(obj))
            if(res == 'negative'): 
                neg_cnt = neg_cnt + 1
            break
        else:
            continue

for obj in test_pos:
    for word in obj.split():
        if (word[0] in emojis) or (word[-1] in emojis):
            total_pos += 1
            res =  classifier.classify(extract_features(obj))
            if(res == 'positive'): 
                pos_cnt = pos_cnt + 1
            break
        else:
            continue

print()
print('[negative]: %s/%s '  % (neg_cnt, total_neg))
print('[positive]: %s/%s '  % (pos_cnt, total_pos))
print()

def classify_tweet(tweet):
    for word in tweet.split():
        if (word[0] in emojis) or (word[-1] in emojis):
            print(tweet, "[", classifier.classify(extract_features(clean_tweet(tweet))), "]")
            break
    return 0