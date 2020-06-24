import sys
import os
import re, string, random

import pandas as pd
import nltk

# replace apostrophe/short words in python
from contractions import contractions_dict

from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer



def pre_processing_tweet(tweet):

    # remove special characters
    tweet = re.sub("[^a-zA-Z#]", ' ', tweet)

    # convert all @username to empty value
    tweet = re.sub('@[^\s]+', '', tweet)

    # convert "#topic" to just "topic"
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet)

    # remove punctuations
    tweet = re.sub(r'[^(a-zA-Z)\s]', '', tweet)

    # correct all multiple white spaces to a single white space
    tweet = re.sub('[\s]+', ' ', tweet)

    # tokenize
    tweet = word_tokenize(tweet)

    return tweet


def clean_twitter_data(tweet_df):

    tweet_data_frame = pd.DataFrame(columns=['tweet_id', 'screen_id', 'tweet_message',
                                                     'tweet_source', 'retweet_count', 'likes_count',
                                                     'tweet_date'])
    for row in tweet_df.itertuples(index=False):
        text = row.tweet_message

        text = pre_processing_tweet(text)

        # sentense length greater than 5 words
        if len(text) <= 5:
            tweet_df.drop(tweet_df[tweet_df['tweet_id'] == row.tweet_id ].index , inplace=True)
 
    return tweet_df;

if __name__ == "__main__":

    text = "@KimbleCharting 😃 sss"
    tweet = pre_processing_tweet(text)
    print(len(tweet))
