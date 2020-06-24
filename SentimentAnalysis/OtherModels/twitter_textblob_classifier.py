import sys
import os
import pandas as pd

# data cleaning and load methods
import pre_processing_data as pre_processing
import data_load as load

# sentiment analysis methods
from textblob import TextBlob

# database
sys.path.append(os.path.abspath('../../Database'))
import database_log



def textblob_sentiment(tweet):

    print(tweet)

    # clean tweet
    tweet = pre_processing.pre_processing_textblob_tweet(tweet)
    analyser = TextBlob(tweet)

    sentiment_score = analyser.sentiment

    sentiment = ''

    # sentiment type positive (1), neutral (2), negative (0)
    if analyser.sentiment.polarity > 0:
        sentiment = '1'
    elif analyser.sentiment.polarity == 0:
        sentiment = '2'
    else:
        sentiment = '0'

    print(tweet)

    return sentiment_score, sentiment


def get_textblob_sentiment(tweets_train):

    for row in tweets_train.itertuples(index=False):

        tweet = row.tweet

        sentiment_score, sentiment = textblob_sentiment(tweet)

        print("original sentiment - [{}] ; sentiment score - [{}] ; sentiment - [{}]"
              .format(row["label"], sentiment_score, sentiment))


if __name__ == "__main__":

    print("***** Load data set *****")
    df_tweets_train_data, df_tweets_test_data = load.load_data()

    print("***** TextBlob sentiment analyser *****")
    get_textblob_sentiment(df_tweets_train_data)
