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

# database
# sys.path.append(os.path.abspath('../../Database'))
# import Database.database_log as database_log


# What's -> What is
def remove_apostrophe(tweet, contractions_dict):
    contractions_pattern = re.compile('({})'.format('|'.join(contractions_dict.keys())),
                                      flags=re.IGNORECASE | re.DOTALL)

    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = contractions_dict.get(match) \
            if contractions_dict.get(match) \
            else contractions_dict.get(match.lower())
        expanded_contraction = expanded_contraction
        return expanded_contraction

    expanded_sentence = contractions_pattern.sub(expand_match, tweet)
    expanded_sentence = re.sub("'", "", expanded_sentence)
    return expanded_sentence


def pre_processing_spacy(tweet):

    # remove apostrophe
    tweet = remove_apostrophe(tweet, contractions_dict)

    # convert all @username to empty
    tweet = re.sub('@[^\s]+', '', tweet)

    # convert "#topic" to just "topic"
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet)

    tweet = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\ / \ / \S+)", " ", tweet).split())

    return tweet


# region textblob pre processing

def pre_processing_textblob_tweet(tweet):

    # remove apostrophe
    tweet = remove_apostrophe(tweet, contractions_dict)

    # remove special characters
    tweet = re.sub("[^a-zA-Z#]", ' ', tweet)

    # convert all @username to empty
    tweet = re.sub('@[^\s]+', '', tweet)

    # convert "#topic" to just "topic"
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet)

    tweet = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\ / \ / \S+)", " ", tweet).split())

    # correct all multiple white spaces to a single white space
    tweet = re.sub('[\s]+', ' ', tweet)

    return tweet

# endregion

# region common pre processing method


def pre_processing_tweet(tweet):

    # remove special characters
    tweet = re.sub("[^a-zA-Z#]", ' ', tweet)

    # remove apostrophe
    tweet = remove_apostrophe(tweet, contractions_dict)

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

    # remove stopwords
    tweet = [w for w in tweet if not w in stop_words]

    return tweet

def common_pre_processing(tweet):

    # remove punctuations
    tweet = re.sub(r'[^(a-zA-Z)\s]', '', tweet)

    # tokenize
    tweet = word_tokenize(tweet)

    # remove stopwords
    tweet = [w for w in tweet if not w in stop_words]

    return tweet

# endregion

# region ****** pre processing model training data set ******

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

        # tuples (tweets,lable)
        documents.append((tweet, "pos"))

        tweet = common_pre_processing(tweet)

        # parts of speech tagging for each word
        parts_of_speech_tagging = nltk.pos_tag(tweet)

        # make a list of  all adjectives identified by the allowed word types list above
        # lemmatize_sentence
        for w in parts_of_speech_tagging:
            if w[1][0] in allowed_word_types:
                all_words.append(w[0].lower())

    # pre process negative tweets
    for row in df_tweets_train_neg.itertuples(index=False):

        tweet = row.tweet

        # tuples (tweets,lable)
        documents.append((tweet, "neg"))

        tweet = common_pre_processing(tweet)

        # parts of speech tagging for each word
        parts_of_speech_tagging = nltk.pos_tag(tweet)

        # make a list of  all adjectives identified by the allowed word types list above
        for w in parts_of_speech_tagging:
            if w[1][0] in allowed_word_types:
                all_words.append(w[0].lower())

    return all_words, documents

# function to create a dictionary of features for each review in the list document.
# The keys are the words in word_features
# The values of each key are either true or false for weather that feature appears in the review or not
def find_features(tweet, word_features):

    words = pre_processing_tweet(tweet)

    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features

# endregion ****** pre processing model training data set ******

class NltkPreProcessingModel:

    def __init__(self):
        pass

    @classmethod
    def remove_apostrophe(cls,tweet, contractions_dict):
        # What's -> What is
        contractions_pattern = re.compile('({})'.format('|'.join(contractions_dict.keys())),
                                          flags=re.IGNORECASE | re.DOTALL)

        def expand_match(contraction):
            match = contraction.group(0)
            first_char = match[0]
            expanded_contraction = contractions_dict.get(match) \
                if contractions_dict.get(match) \
                else contractions_dict.get(match.lower())
            expanded_contraction = expanded_contraction
            return expanded_contraction

        expanded_sentence = contractions_pattern.sub(expand_match, tweet)
        expanded_sentence = re.sub("'", "", expanded_sentence)
        return expanded_sentence

    @classmethod
    def clean_text(cls,text_tokens, stop_words = ()):

        cleaned_tokens = []

        for token, tag in pos_tag(text_tokens):

            # remove hyperlinks
            token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|'\
                           '(?:%[0-9a-fA-F][0-9a-fA-F]))+','', token)

            # convert all @username to empty value
            token = re.sub("(@[A-Za-z0-9_]+)","", token)

            # correct all multiple white spaces to a single white space
            token = re.sub('[\s]+', ' ', token)

            token = remove_apostrophe(token, contractions_dict)

            if tag.startswith("NN"):
                pos = 'n'
            elif tag.startswith('VB'):
                pos = 'v'
            else:
                pos = 'a'

            lemmatizer = WordNetLemmatizer()
            token = lemmatizer.lemmatize(token, pos)

            if len(token) > 0 and token not in string.punctuation and token.lower() not in stop_words:
                cleaned_tokens.append(token.lower())
        return cleaned_tokens

    @classmethod
    def get_all_words(cls,cleaned_tokens_list):
        for tokens in cleaned_tokens_list:
            for token in tokens:
                yield token

    @classmethod
    def get_tweets_for_model(cls,cleaned_tokens_list):
        for tweet_tokens in cleaned_tokens_list:
            yield dict([token, True] for token in tweet_tokens)


class SpacyPreProcessingModel:

    def __init__(self):
        pass

    @classmethod
    def remove_apostrophe(cls,tweet, contractions_dict):
        # What's -> What is
        contractions_pattern = re.compile('({})'.format('|'.join(contractions_dict.keys())),
                                          flags=re.IGNORECASE | re.DOTALL)

        def expand_match(contraction):
            match = contraction.group(0)
            first_char = match[0]
            expanded_contraction = contractions_dict.get(match) \
                if contractions_dict.get(match) \
                else contractions_dict.get(match.lower())
            expanded_contraction = expanded_contraction
            return expanded_contraction

        expanded_sentence = contractions_pattern.sub(expand_match, tweet)
        expanded_sentence = re.sub("'", "", expanded_sentence)
        return expanded_sentence


    @classmethod
    def clean_text(cls,tweet):
        # remove apostrophe
        tweet = remove_apostrophe(tweet, contractions_dict)

        # convert all @username to empty
        tweet = re.sub('@[^\s]+', '', tweet)

        # convert "#topic" to just "topic"
        tweet = re.sub(r'#([^\s]+)', r'\1', tweet)

        # convert to single space
        tweet = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\ / \ / \S+)", " ", tweet).split())

        return tweet


if __name__ == "__main__":

    text = "We aren't driving to the zoo, it'll take too long."
    tweet = remove_apostrophe(text, contractions_dict)

    print(tweet)
