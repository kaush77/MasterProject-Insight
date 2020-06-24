import sys
import os
import pandas as pd

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

import cleaning_data as pre_processing
import data_load as load

# database
sys.path.append(os.path.abspath('../../Database'))
import database_log


def vader_sentiment_analyzer_scores(tweet):

    analyser = SentimentIntensityAnalyzer() 

    x_pos = analyser.polarity_scores(tweet)["pos"]
    x_neu = analyser.polarity_scores(tweet)["neu"]
    x_neg = analyser.polarity_scores(tweet)["neg"]
    x_comp = analyser.polarity_scores(tweet)["compound"]

    sentiment = ""
    # sentiment type positive (1), neutral (2), negative (0)
    if x_comp >= 0.05:
        sentiment = "1"
    elif (x_comp > -0.05) and (x_comp < 0.05):
        sentiment = "2"
    elif x_comp <= -0.05:
        sentiment = "0"

    return x_pos, x_neu, x_neg, x_comp, sentiment


def get_vader_sentiment(tweets_train):

    for row in tweets_train.itertuples(index=False):
        tweet = row.tweet

        x_pos, x_neu, x_neg, x_comp, sentiment = vader_sentiment_analyzer_scores(tweet)

        print("original sentiment - [{}] ; positive - [{}] ; neutral - [{}] ; negative - [{}] ;"
              " compound - [{}] ; sentiment - [{}]"
              .format(row["label"], x_pos, x_neu, x_neg, x_comp, sentiment))


if __name__ == "__main__":

    print("***** Load data set *****")
    df_tweets_train_data, df_tweets_test_data = load.load_data()

    print("***** Vader sentiment analyser *****")
    get_vader_sentiment(df_tweets_train_data)
