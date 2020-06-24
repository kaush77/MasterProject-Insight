import sys
import os
import re

import random
import pandas as pd
import nltk

from sklearn.metrics import f1_score
from nltk.classify.scikitlearn import SklearnClassifier
import pickle
import twitter_nltk_training_model as ntm

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
stop_words = list(set(stopwords.words('english')))

# database
sys.path.append(os.path.abspath('../../Database'))
import database_log


def pre_processing(df_tweets_train):

    all_words = []
    documents = []

    #  j is adject, r is adverb, and v is verb allowed_word_types = ["J","R","V"]

    allowed_word_types = ["J"]

    # filter positive and negative tweets data
    df_tweets_train_pos = df_tweets_train.loc[df_tweets_train['label'] == 1]
    df_tweets_train_neg = df_tweets_train.loc[df_tweets_train['label'] == 0]

    # pre process positive tweets
    for row in df_tweets_train_pos.itertuples(index=False):

        tweet = row.tweet

        # create a list of tuples where the first element of each tuple is a review
        # the second element is the label

        documents.append((tweet, "pos"))

        # remove punctuations
        remove_punctuations = re.sub(r'[^(a-zA-Z)\s]', '', tweet)

        # tokenize
        tokenized = word_tokenize(remove_punctuations)

        # remove stopwords
        remove_stopped_words = [w for w in tokenized if not w in stop_words]

        # parts of speech tagging for each word
        parts_of_speech_tagging = nltk.pos_tag(remove_stopped_words)

        # make a list of  all adjectives identified by the allowed word types list above
        # lemmatize_sentence
        for w in parts_of_speech_tagging:
            if w[1][0] in allowed_word_types:
                all_words.append(w[0].lower())

    # pre process negative tweets
    for row in df_tweets_train_neg.itertuples(index=False):

        tweet = row.tweet

        # create a list of tuples where the first element of each tuple is a review
        # the second element is the label
        documents.append((tweet, "neg"))

        # remove punctuations
        remove_punctuations = re.sub(r'[^(a-zA-Z)\s]', '', tweet)

        # tokenize
        tokenized = word_tokenize(remove_punctuations)

        # remove stopwords
        remove_stopped_words = [w for w in tokenized if not w in stop_words]

        # parts of speech tagging for each word
        parts_of_speech_tagging = nltk.pos_tag(remove_stopped_words)

        # make a list of  all adjectives identified by the allowed word types list above
        for w in parts_of_speech_tagging:
            if w[1][0] in allowed_word_types:
                all_words.append(w[0].lower())

    return all_words, documents

# remove Hyperlinks, Twitter handles in replies (@),
def pre_processing_v1(df_tweets_train):

    all_words = []
    documents = []

    #  j is adject, r is adverb, and v is verb allowed_word_types = ["J","R","V"]

    allowed_word_types = ["J"]

    # filter positive and negative tweets data
    df_tweets_train_pos = df_tweets_train.loc[df_tweets_train['label'] == 1]
    df_tweets_train_neg = df_tweets_train.loc[df_tweets_train['label'] == 0]

    # pre process positive tweets
    for row in df_tweets_train_pos.itertuples(index=False):

        tweet = row.tweet

        # create a list of tuples where the first element of each tuple is a review
        # the second element is the label

        documents.append((tweet, "pos"))

        # remove punctuations
        remove_punctuations = re.sub(r'[^(a-zA-Z)\s]', '', tweet)

        # tokenize
        tokenized = word_tokenize(remove_punctuations)

        print(tokenized)

        # remove stopwords
        remove_stopped_words = [w for w in tokenized if not w in stop_words]

        # parts of speech tagging for each word
        parts_of_speech_tagging = nltk.pos_tag(remove_stopped_words)

        # make a list of  all adjectives identified by the allowed word types list above
        # lemmatize_sentence
        for w in parts_of_speech_tagging:
            if w[1][0] in allowed_word_types:
                all_words.append(w[0].lower())

    # pre process negative tweets
    for row in df_tweets_train_neg.itertuples(index=False):

        tweet = row.tweet

        # create a list of tuples where the first element of each tuple is a review
        # the second element is the label
        documents.append((tweet, "neg"))

        # remove punctuations
        remove_punctuations = re.sub(r'[^(a-zA-Z)\s]', '', tweet)

        # tokenize
        tokenized = word_tokenize(remove_punctuations)

        # remove stopwords
        remove_stopped_words = [w for w in tokenized if not w in stop_words]

        # parts of speech tagging for each word
        parts_of_speech_tagging = nltk.pos_tag(remove_stopped_words)

        # make a list of  all adjectives identified by the allowed word types list above
        for w in parts_of_speech_tagging:
            if w[1][0] in allowed_word_types:
                all_words.append(w[0].lower())

    return all_words, documents


# function to create a dictionary of features for each review in the list document.
# The keys are the words in word_features
# The values of each key are either true or false for wether that feature appears in the review or not
def find_features(document, word_features):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features


if __name__ == '__main__':

    df_tweets_train_data, df_tweets_test_data = ntm.load_data()

    pre_processing_v1(df_tweets_train_data)

# df_tweets_train_data, df_tweets_test_data = load_data()
# # print(df_tweets_train_data.head(3))
#
# all_words, documents = pre_processing(df_tweets_train_data)
#
#
# # creating a frequency distribution of each adjectives.
# BOW = nltk.FreqDist(all_words)
#
# # listing the 5000 most frequent words
# word_features = list(BOW.keys())[:5000]
#
# # Creating features for each review
# featuresets = [(find_features(rev), category) for (rev, category) in documents]
#
# # Shuffling the documents
# random.shuffle(featuresets)
# print(len(featuresets))
#
# training_set = featuresets[:25000]
# testing_set = featuresets[25000:]
# print('training_set :', len(training_set), '\ntesting_set :', len(testing_set))
#
# classifier = nltk.NaiveBayesClassifier.train(training_set)
# print("Classifier accuracy percent:",(nltk.classify.accuracy(classifier, testing_set))*100)
# classifier.show_most_informative_features(15)
#
# # getting predictions for the testing set by looping over each reviews featureset tuple
# # The first elemnt of the tuple is the feature set and the second element is the label
# ground_truth = [r[1] for r in testing_set]
#
# preds = [classifier.classify(r[0]) for r in testing_set]
#
#
# f1_score(ground_truth, preds, labels = ['neg', 'pos'], average = 'micro')
