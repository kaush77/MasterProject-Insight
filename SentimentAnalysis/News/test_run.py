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

import pickle
import spacy
import string

# start spacy prediction

import numpy as np
import pickle
import spacy
import string
from spacy.lang.en import English

import en_core_web_sm
from spacy.lang.en.stop_words import STOP_WORDS
nlp = en_core_web_sm.load()

# To build a list of stop words for filtering
stopwords = list(STOP_WORDS)

punctuations = string.punctuation
parser = English()

# end spacy prediction

# import news_textblob_classifier as news_textblob
# import news_vader_classifier
import nltk_classifier
import spacy_classifier_model
import word2vec_classifier_model

import sys
import os
import pandas as pd
import news as sql_execute
import database as sql_database_execute

# database
sys.path.append(os.path.abspath('../../Database'))
import database_log

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

    text = pre_processing.pre_processing_text(text)
    text_tokens = parser(text)
    # Lemma that are not pronouns
    text_tokens = [word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in text_tokens]
    # Stop words and Punctuation In List Comprehension
    text_tokens = [word for word in text_tokens if word not in stopwords and word not in punctuations]

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

def get_spacy_model():

    model_path = '../TrainModel/News/'

    # load spacy train models CountVectorizer
    pipe_count_vectorizer = load_model(model_path +"CountVectorizer.pickle")

    # load spacy train models TfidfVectorizer
    pipe_tfidf_vectorizer = load_model(model_path +"TfidfVectorizer.pickle")

    return pipe_count_vectorizer, pipe_tfidf_vectorizer


def predict_sentiment(text):

    # predict using CountVectorizer
    count_vectorizer_prediction = pipe_count_vectorizer.predict(text)

    # predict using TfidfVectorizer
    tfidf_vectorizer_prediction = pipe_tfidf_vectorizer.predict(text)

    return count_vectorizer_prediction, tfidf_vectorizer_prediction

# end spacy prediction

# predict header sentiment
def predict_models_header_sentiment(text_data_list):

    text_data_df = pd.DataFrame(text_data_list,columns=['id', 'header', 'sub_header'])

    # added sentiment analyzer columns to store sentiment value
    text_data_df["sentiment_for"] = ""
    text_data_df["nltk_classify"] = -1
    text_data_df["nltk_confidence"] = 0.0
    text_data_df["word2vec_classify"] = -1
    text_data_df["count_vectorizer_classify"] = -1
    text_data_df["tfidf_vectorizer_classify"] = -1

    # ---------news nltk load-------------")
    word_features = nltk_classifier.load_save_dataset('word_features.pickle')
    ensemble_clf = nltk_classifier.get_ensemble_models(None)

    # --------news word2vec load-------------
    model, w2vmodel, tfidf = word2vec_classifier_model.load_prediction_model_parameters()

    # --------spacy model load-------------
    pipe_count_vectorizer, pipe_tfidf_vectorizer = get_spacy_model()

    for row in text_data_df.itertuples(index=False):
        text = row.header

        # ---------nltk-------------")
        classify, nltk_confidence = nltk_classifier.sentiment_analyzer(text, ensemble_clf, word_features)

        nltk_classify = 0
        if classify == "pos":
            nltk_classify = 1
        elif classify == "neg":
            nltk_classify = 0
        else:
            nltk_classify = 2

        text_data_df["sentiment_for"] = "header"
        text_data_df.at[index, 'nltk_classify'] = nltk_classify
        text_data_df.at[index, 'nltk_confidence'] = nltk_confidence

        # --------news word2vec-------------
        word2vec_classify = word2vec_classifier_model.predict(model, w2vmodel, tfidf, text)

        text_data_df.at[index, 'word2vec_classify'] = word2vec_classify

        # --------spacy------------------------
        text_list = []
        text_list.append(text)

        # predict using CountVectorizer
        count_vectorizer_classify = pipe_count_vectorizer.predict(text_list)
        # predict using TfidfVectorizer
        tfidf_vectorizer_classify = pipe_tfidf_vectorizer.predict(text_list)

        text_data_df.at[index, 'count_vectorizer_classify'] = int(count_vectorizer_classify[0])
        text_data_df.at[index, 'tfidf_vectorizer_classify'] = int(tfidf_vectorizer_classify[0])

    return text_data_df

# predict sub header sentiment
def predict_models_subheader_sentiment(text_data_list):

    text_data_df = pd.DataFrame(text_data_list,columns=['id', 'header', 'sub_header'])

    # added sentiment analyzer columns to store sentiment value
    text_data_df["sentiment_for"] = ""
    text_data_df["nltk_classify"] = -1
    text_data_df["nltk_confidence"] = 0.0
    text_data_df["word2vec_classify"] = -1
    text_data_df["count_vectorizer_classify"] = -1
    text_data_df["tfidf_vectorizer_classify"] = -1

    # ---------news nltk load-------------")
    word_features = nltk_classifier.load_save_dataset('word_features.pickle')
    ensemble_clf = nltk_classifier.get_ensemble_models(None)

    # --------news word2vec load-------------
    model, w2vmodel, tfidf = word2vec_classifier_model.load_prediction_model_parameters()

    # --------spacy model load-------------
    pipe_count_vectorizer, pipe_tfidf_vectorizer = get_spacy_model()

    for row in text_data_df.itertuples(index=False):
        text = row.sub_header

        if len(text) > 0:

            # ---------nltk-------------")
            classify, nltk_confidence = nltk_classifier.sentiment_analyzer(text, ensemble_clf, word_features)

            nltk_classify = 0
            if classify == "pos":
                nltk_classify = 1
            elif classify == "neg":
                nltk_classify = 0
            else:
                nltk_classify = 2

            text_data_df["sentiment_for"] = "sub_header"
            text_data_df.at[index, 'nltk_classify'] = int(nltk_classify)
            text_data_df.at[index, 'nltk_confidence'] = float(nltk_confidence)


            # --------news word2vec-------------
            word2vec_classify = word2vec_classifier_model.predict(model, w2vmodel, tfidf, text)

            text_data_df.at[index, 'word2vec_classify'] = int(word2vec_classify)

            # --------spacy------------------------
            text_list = []
            text_list.append(text)

            # predict using CountVectorizer
            count_vectorizer_classify = pipe_count_vectorizer.predict(text_list)
            # predict using TfidfVectorizer
            tfidf_vectorizer_classify = pipe_tfidf_vectorizer.predict(text_list)

            text_data_df.at[index, 'count_vectorizer_classify'] = int(count_vectorizer_classify[0])
            text_data_df.at[index, 'tfidf_vectorizer_classify'] = int(tfidf_vectorizer_classify[0])

    return text_data_df

if __name__ == "__main__":

    # tweet = "thanks for #lyft credit i can't use because they don't offer wheelchair vans in pdx.    " \
    #         "#disapointed #getthanked"
    text ="Intel surges 8% on an earnings beat and better-than-expected forecast"
    # text = "$MMM fell on hard times but could be set to rebound soon.  "
    # text = "$MMM is best #dividend #stock out there and down 40% in 2019 $XLI go go please "
    # text =" I am doing bad in stock market"

    tweet_list = ["thanks for #lyft credit i can't use cause they don't offer wheelchair vans in pdx. #disapointed #getthanked",
                  "I do enjoy my job",
                  "What a poor product!,I will have to get a new one",
                  "I feel amazing!",
                  "thanks for lyft credit i can not use because they do not offer wheelchair vans in pdx disapointed getthanked",
                "Intel surges 8% on an earnings beat and better-than-expected forecast"]

    print("\n--------nltk-------------")

    # print(pre_processing.common_pre_processing_steps(tweet))
    # print("***** loading word features *****")
    word_features = nltk_classifier.load_save_dataset('word_features.pickle')

    # print("***** save train models to a ensemble *****")
    ensemble_clf = nltk_classifier.get_ensemble_models(None)

    # print("***** predict tweets sentiment *****")
    for tokens in range(len(tweet_list)):
        classify, confidence = nltk_classifier.sentiment_analyzer(tweet_list[tokens], ensemble_clf, word_features)
        print("classify - {} , confidence - {}".format(classify, confidence))

    print("\n--------news_word2vec-------------")

    model, w2vmodel, tfidf = word2vec_classifier_model.load_prediction_model_parameters()

    for tokens in range(len(tweet_list)):
        prediction = word2vec_classifier_model.predict(model, w2vmodel, tfidf, tweet_list[tokens])
        print(" prediction - {}".format(prediction))

    # --------spacy model load-------------
    print("\n--------spacy-------------")
    pipe_count_vectorizer, pipe_tfidf_vectorizer = get_spacy_model()
    text_list = []

    for tokens in range(len(tweet_list)):
        text_list.append(text)

    # predict using CountVectorizer
    count_vectorizer_classify = pipe_count_vectorizer.predict(text_list)
    # predict using TfidfVectorizer
    tfidf_vectorizer_classify = pipe_tfidf_vectorizer.predict(text_list)
    print(" count_vectorizer_classify - {} tfidf_vectorizer_classify - {}".
                format(count_vectorizer_classify,tfidf_vectorizer_classify))
