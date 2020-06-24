# ML Packages
from sklearn.base import TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split

# pre processing package
import pre_processing_data as pre_processing
import data_load as load
from pre_processing_data import SpacyPreProcessingModel
import twitter_spacy_countVectorizer_model
import twitter_spacy_tfidfVectorizer_model

# Ensemble
from nltk.classify import ClassifierI
from statistics import mode

import pickle
import spacy
import string
import calendar
import time
from spacy.lang.en import English

import en_core_web_sm
from spacy.lang.en.stop_words import STOP_WORDS
nlp = en_core_web_sm.load()

# To build a list of stop words for filtering
stopwords = list(STOP_WORDS)

punctuations = string.punctuation
parser = English()

import twitter_nltk_classifier
import pre_processing_data as pre_processing
import twitter_word2vec_classifier_model

import sys
import os
import pandas as pd
import Database.twitter as sql_execute
import Database.database as sql_database_execute
from nltk.tokenize import TweetTokenizer
tokenizer = TweetTokenizer()

# database
# sys.path.append(os.path.abspath('../../Database'))
sys.path.append(os.path.abspath('../../'))
import Database.database_log as database_log

sys.path.append(os.path.abspath('../../'))
import app_config

import warnings
warnings.filterwarnings('ignore')

# Creating a Spacy Parser

class predictors(TransformerMixin):

    def transform(self, X, **transform_params):
        return [clean_text(text) for text in X]

    def fit(self, X, y, **fit_params):
        return self

    def get_params(self, deep=True):
        return {}

# words tokenizer, remove pronoun, punctuations
def tokenizer(tweet):

    tweet = SpacyPreProcessingModel.clean_text(tweet)
    tweet_tokens = nlp(tweet)

    # Lemma that are NOUN, VERB, ADV
    tweet_tokens = [word.lemma_.lower().strip() if word.pos_ in("NOUN","VERB","ADV") else word.lower_ for word in tweet_tokens]

    # Stop words and Punctuation In List Comprehension
    tweet_tokens = [word for word in tweet_tokens if word not in stopwords and word not in punctuations]

    return tweet_tokens

# Basic function to clean the text
def clean_text(text):
    return text.strip().lower()

# function to load models given file path
def load_model(file_path):

    classifier_f = open(file_path, "rb")
    classifier = pickle.load(classifier_f)
    classifier_f.close()
    return classifier

# end spacy prediction
def predict_models_sentiment():

    # read tweet from database
    twitter_data_list = sql_execute.read_twitter_data()
    twitter_df = pd.DataFrame(twitter_data_list,columns=['id', 'tweet_id', 'screen_id', 'tweet_message', 'tweet_date'])

    # ---------twitter nltk load-------------")
    word_features = twitter_nltk_classifier.load_save_dataset('word_features.pickle')
    ensemble_nltk_clf = twitter_nltk_classifier.get_ensemble_models()

    # --------twitter word2vec load-------------
    model, w2vmodel, tfidf = twitter_word2vec_classifier_model.load_prediction_model_parameters()

    sentiment_data_list = []

    for row in twitter_df.itertuples(index=False):

        tweet = row.tweet_message

        try:
            # ---------nltk-------------")
            classify, nltk_confidence = twitter_nltk_classifier.sentiment_analyzer(tweet, ensemble_nltk_clf)

            nltk_classify = 2
            if classify == "Positive":
                nltk_classify = 1
            elif classify == "Negative":
                nltk_classify = 0
            else:
                nltk_classify = 2

            # --------twitter word2vec-------------
            word2vec_classify = twitter_word2vec_classifier_model.predict(model, w2vmodel, tfidf, tweet)
            # word2vec_classify = 0

            # predict using CountVectorizer
            classify_cVector, confidence_cVector = twitter_spacy_countVectorizer_model.sentiment_analyzer(tweet)
            # classify_cVector = 2
            # confidence_cVector =1

            # predict using TfidfVectorizer
            classify_tfidf, confidence_tfidf = twitter_spacy_tfidfVectorizer_model.sentiment_analyzer(tweet)
            # classify_tfidf =2
            # confidence_tfidf=1

            sentiment_data_list.append({'id': row.id, 'tweet_id':row.tweet_id, 'screen_id': row.screen_id,
                               'tweet_message': row.tweet_message, 'tweet_date': row.tweet_date,
                               'nltk_classify': nltk_classify, 'nltk_confidence': nltk_confidence,
                               'word2vec_classify':word2vec_classify,
                               'count_vectorizer_classify':int(classify_cVector),'count_vectorizer_confidence':confidence_cVector,
                               'tfidf_vectorizer_classify':int(classify_tfidf), 'tfidf_vectorizer_confidence':confidence_tfidf})

        except Exception as error:
            database_log.error_log("run_twitter_sentiment_analyzer - predict_models_sentiment", error)
            print("run_twitter_sentiment_analyzer - predict_models_sentiment - {}".format(error))


    tweet_sentiment_data = pd.DataFrame(columns=['id', 'tweet_id', 'screen_id','tweet_message','tweet_date','nltk_classify','nltk_confidence',
                                                 'word2vec_classify', 'count_vectorizer_classify', 'count_vectorizer_confidence','tfidf_vectorizer_classify'
                                                 'tfidf_vectorizer_confidence'])
    if len(sentiment_data_list) > 0:
        tweet_sentiment_data = tweet_sentiment_data.append(sentiment_data_list)

    return tweet_sentiment_data

# predict sentiment
def twitter_run_init():
    twitter_data_sentiment_df = predict_models_sentiment()

    if twitter_data_sentiment_df.empty is not True:
        sql_execute.bulk_insert_twitter_sentiment(twitter_data_sentiment_df)


if __name__ == "__main__":

    # predict sentiment
    # try:

    print("\n Running twitter sentiment prediction model....\n")

    while True:
        # Wait for value in seconds
        scheduled_task_sleeping =  app_config.twitter_sentiment_calculation_task_sleeping

        # try:
        twitter_run_init()
        print("Last run was successful, wating to run in next {} seconds.".format(scheduled_task_sleeping))
        time.sleep(scheduled_task_sleeping)

        # except Exception as error:
        #     database_log.error_log("twitter__sentiment_analyzer - main", error)

    # except Exception as error:
    #     database_log.error_log("twitter__sentiment_analyzer - main", error)
