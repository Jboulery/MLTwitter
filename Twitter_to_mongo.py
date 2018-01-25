
# coding: utf-8

# In[ ]:

from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
import json
from pymongo import MongoClient

#Twitter credentials

access_token = "$access_token"
access_token_secret = "$access_token_secret"
consumer_key = "$consumer_key"
consumer_secret = "$consumer_secret"

#Connect the database
client = MongoClient('localhost', 27017)
db = client.tweets
collection = db.testCollection

#This is a basic listener that inserts tweets in a mongo database.
class StdOutListener(StreamListener):

    def on_data(self, data):
        clean_data = json.loads(data)
        collection.insert_one(clean_data)
        print('inserted')
        return True

    def on_error(self, status):
        if status == 420: #Disconnect the stream
            print(status)
            return False


if __name__ == '__main__':

    #Twitter authentication
    l = StdOutListener()
    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    stream = Stream(auth, l)

    #Twitter streams
    stream.filter(track=['#LRDS'])

