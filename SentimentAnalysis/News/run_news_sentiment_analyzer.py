# ML Packages
from sklearn.base import TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize

# Ensemble
from nltk.classify import ClassifierI
from statistics import mode

# pre processing package
import data_load as load
import news_spacy_countVectorizer_model
import news_spacy_tfidfVectorizer_model 

import sys
import os
import pandas as pd
import calendar
import time
import pickle
import spacy
import string
import numpy as np
from spacy.lang.en import English
import en_core_web_sm

from spacy.lang.en.stop_words import STOP_WORDS
nlp = en_core_web_sm.load()

sys.path.append(os.path.abspath('../../'))
import app_config

# To build a list of stop words for filtering
stopwords = list(STOP_WORDS)

punctuations = string.punctuation
parser = English()

import nltk_classifier
import spacy_classifier_model
import word2vec_classifier_model

import Database.database as sql_database_execute
import Database.news as sql_execute

# database
sys.path.append(os.path.abspath('../../Database'))
import Database.database_log as database_log

# start spacy prediction
# Creating a Spacy Parser
class predictors(TransformerMixin):

    def transform(self, X, **transform_params):
        return [clean_text(text) for text in X]

    def fit(self, X, y, **fit_params):
        return self

    def get_params(self, deep=True):
        return {}

# words tokenizer, remove pronoun, punctuations
def tokenizer(text):
    text = news_spacy_countVectorizer_model.pre_processing_text(text)
    text_tokens = word_tokenize(text)
    return text_tokens

# Basic function to clean the text
def clean_text(text):
    return text.strip().lower()

# function to load models given file path
def load_model(file_path):

    classifier_f = open(file_path, "rb")
    classifier = pickle.load(classifier_f)
    classifier_f.close()
    return classifier

# predict header sentiment
def predict_models_header_sentiment(text_data_list):

    text_data_df = pd.DataFrame(text_data_list,columns=['id', 'header', 'sub_header'])

    # delete record with empty or "NaN" value
    text_data_df.drop(text_data_df[text_data_df['header'] == "NaN"].index , inplace=True)
    text_data_df.drop(text_data_df[text_data_df['header'] == ""].index , inplace=True)

    # added sentiment analyzer columns to store sentiment value
    text_data_df["sentiment_for"] = ""
    text_data_df["nltk_classify"] = ""
    text_data_df["nltk_confidence"] = ""
    # text_data_df["word2vec_classify"] = -1
    text_data_df["count_vectorizer_classify"] = ""
    text_data_df["count_vectorizer_confidence"] = ""
    text_data_df["tfidf_vectorizer_classify"] = ""
    text_data_df["tfidf_vectorizer_confidence"] = ""

    # ---------news nltk load-------------")
    word_features = nltk_classifier.load_save_dataset('word_features.pickle')
    ensemble_clf = nltk_classifier.get_ensemble_models()

    # --------news word2vec load-------------
    # model, w2vmodel, tfidf = word2vec_classifier_model.load_prediction_model_parameters()

    sentiment_data_list = []

    for row in text_data_df.itertuples(index=False):
        text = row.header

        if len(text) > 0 and text != "NaN":

            try:
                # ---------nltk-------------")
                classify, nltk_confidence = nltk_classifier.sentiment_analyzer(text, ensemble_clf, word_features)

                nltk_classify = 2
                if classify == "pos":
                    nltk_classify = 1
                elif classify == "neg":
                    nltk_classify = 0
                else:
                    nltk_classify = 2


                # --------news word2vec-------------
                # word2vec_classify = word2vec_classifier_model.predict(model, w2vmodel, tfidf, text)
                #
                # text_data_df['word2vec_classify'] = word2vec_classify

                # --------spacy------------------------
                classify_cVector, confidence_cVector = news_spacy_countVectorizer_model.sentiment_analyzer(text)
                 
                # predict using TfidfVectorizer
                classify_tfidf, confidence_tfidf = news_spacy_tfidfVectorizer_model.sentiment_analyzer(text)                 
                
                sentiment_data_list.append({'id': row.id, 'header':row.header, 'sub_header': row.sub_header,
                                   'sentiment_for': "header", 'nltk_classify': nltk_classify, 'nltk_confidence': nltk_confidence,
                                   'count_vectorizer_classify':int(classify_cVector),'count_vectorizer_confidence':confidence_cVector,
                                   'tfidf_vectorizer_classify':int(classify_tfidf), 'tfidf_vectorizer_confidence':confidence_tfidf})

            except Exception as error:
                print(error)
                database_log.error_log("run_news_sentiment_analyzer - predict_models_header_sentiment", error)

    news_sentiment_data = pd.DataFrame(columns=['id', 'header', 'sub_header','sentiment_for','nltk_classify','nltk_confidence',
                                                 'count_vectorizer_classify', 'count_vectorizer_confidence','tfidf_vectorizer_classify'
                                                 'tfidf_vectorizer_confidence'])

    if len(sentiment_data_list) > 0:
        news_sentiment_data = news_sentiment_data.append(sentiment_data_list)

    return news_sentiment_data

# predict sub header sentiment
def predict_models_subheader_sentiment(text_data_list):

    text_data_df = pd.DataFrame(text_data_list,columns=['id', 'header', 'sub_header'])

    # delete record with empty or "NaN" value
    text_data_df.drop(text_data_df[text_data_df['sub_header'] == "NaN"].index , inplace=True)
    text_data_df.drop(text_data_df[text_data_df['sub_header'] == ""].index , inplace=True)

    # ---------news nltk load-------------")
    word_features = nltk_classifier.load_save_dataset('word_features.pickle')
    ensemble_clf = nltk_classifier.get_ensemble_models()

    # --------news word2vec load-------------
    # model, w2vmodel, tfidf = word2vec_classifier_model.load_prediction_model_parameters()

    sentiment_data_list=[]

    for row in text_data_df.itertuples(index=False):
        text = row.sub_header

        if len(text) > 0 and text != "NaN":

            try:
                # ---------nltk-------------")
                classify, nltk_confidence = nltk_classifier.sentiment_analyzer(text, ensemble_clf, word_features)

                nltk_classify = 0
                if classify == "pos":
                    nltk_classify = 1
                elif classify == "neg":
                    nltk_classify = 0
                else:
                    nltk_classify = 2

                # --------news word2vec-------------
                # word2vec_classify = word2vec_classifier_model.predict(model, w2vmodel, tfidf, text)
                # text_data_df['word2vec_classify'] = int(word2vec_classify)

                # --------spacy------------------------
                classify_cVector, confidence_cVector = news_spacy_countVectorizer_model.sentiment_analyzer(text) 

                # predict using TfidfVectorizer
                classify_tfidf, confidence_tfidf = news_spacy_tfidfVectorizer_model.sentiment_analyzer(text)

                sentiment_data_list.append({'id': row.id, 'header':row.header, 'sub_header': row.sub_header,
                                   'sentiment_for': "sub_header", 'nltk_classify': nltk_classify, 'nltk_confidence': nltk_confidence,
                                   'count_vectorizer_classify':int(classify_cVector),'count_vectorizer_confidence':confidence_cVector,
                                   'tfidf_vectorizer_classify':int(classify_tfidf), 'tfidf_vectorizer_confidence':confidence_tfidf})

            except Exception as error:
                database_log.error_log("run_news_sentiment_analyzer - predict_models_subheader_sentiment", error)

    news_sentiment_data = pd.DataFrame(columns=['id', 'header', 'sub_header','sentiment_for','nltk_classify','nltk_confidence',
                                                 'count_vectorizer_classify', 'count_vectorizer_confidence','tfidf_vectorizer_classify'
                                                 'tfidf_vectorizer_confidence'])

    if len(sentiment_data_list) > 0:
        news_sentiment_data = news_sentiment_data.append(sentiment_data_list)

    return news_sentiment_data

# predict sentiment
def news_run_init():
    # read text from database
    text_data_list = sql_execute.read_news_data()

    print("News sub-header sentiment prediction... ")
    # sub header sentiment
    news_subheader_sentiment_df = predict_models_subheader_sentiment(text_data_list)

    if news_subheader_sentiment_df.empty is not True:
        sql_execute.bulk_insert_news_sentiment(news_subheader_sentiment_df, False)

    print("News header sentiment prediction...")
    # header sentiment
    news_header_sentiment_df = predict_models_header_sentiment(text_data_list)

    if news_header_sentiment_df.empty is not True:
        sql_execute.bulk_insert_news_sentiment(news_header_sentiment_df, True)


if __name__ == "__main__":

    # predict sentiment
    # try:

    print("\nRunning news headlines sentiment prediction model....\n")

    while True:

        # Wait for value in seconds
        scheduled_task_sleeping =  app_config.news_sentiment_calculation_task_sleeping

        # try:
        news_run_init()
        print("Last run was successful, wating to run in next {} seconds.".format(scheduled_task_sleeping))
        time.sleep(scheduled_task_sleeping)

        # except Exception as error:
        #     database_log.error_log("news_sentiment_analyzer - main", error)
        #     continue

    # except Exception as error:
    #     database_log.error_log("news_sentiment_analyzer - main", error)
