import json
import nltk
from nltk.tokenize import wordpunct_tokenize

file = "dump1801280155.json"
tweet_data = []
tweet_content = []
list_token = []

tweets_file = open(file, 'r', encoding="utf8")

for line in tweets_file :
    try :
        tweet = json.loads (line)
        tweet_data.append(tweet)
    except:
        continue

#liste de contenu des tweets [tweet1, tweet2....]

for tweet in tweet_data :
    try :
        content = tweet["text"]
        tweet_content.append(content)
    except:
        continue

#tokenization (par espaces et par ponctuation, retourne une liste de liste 
for text in tweet_content :
    try :
        s = wordpunct_tokenize(text)
        list_token.append(s)
    except:
        continue

    
    

