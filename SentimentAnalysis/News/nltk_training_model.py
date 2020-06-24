import sys
import os
import re

# Data Analysis
import pandas as pd

import nltk
import random

from contractions import contractions_dict

from wordcloud import WordCloud
from nltk.classify.scikitlearn import SklearnClassifier
import pickle
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC
from nltk.classify import ClassifierI
from statistics import mode
from sklearn.model_selection import train_test_split

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
stop_words = list(set(stopwords.words('english')))

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Lemmatize with POS Tag
from nltk.corpus import wordnet
from nltk.stem.wordnet import WordNetLemmatizer

import data_load as load

# database
sys.path.append(os.path.abspath('../../'))
import Database.database_log as database_log

# Ensemble
from nltk.classify import ClassifierI
from statistics import mode

#plot
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

# pd.set_option('display.width', 200)
# pd.set_option('display.max_columns', 10)


# save train model
def save_train_model(classifier, file_name):

    file_path = "../TrainModel/News/Nltk/" + file_name
    save_classifier = open(file_path, 'wb')
    pickle.dump(classifier, save_classifier)
    save_classifier.close()

# save testing model data set
def save_data(data_set, file_name):

    file_path = "../TrainModel/News//Nltk/" + file_name

    with open(file_path, 'wb') as f:
        pickle.dump(data_set, f)

def construct_feature_word_list(df_news_train):

    all_words = []
    documents = []

    #  j is adject, r is adverb, and v is verb allowed_word_types = ["J","R","V"]

    allowed_word_types = ["J"]

    # filter positive and negative text data
    df_news_train_pos = df_news_train.loc[df_news_train['label'] == 1]
    df_news_train_neg = df_news_train.loc[df_news_train['label'] == 0]

    for row in df_news_train_pos.itertuples(index=False):

        text = row.sub_header

        # tuple of text and label
        documents.append((text, "pos"))

        text = pre_processing_feature_text(text)

        # parts of speech tagging for each word
        parts_of_speech_tagging = nltk.pos_tag(text)

        # make a list of  all adjectives identified by the allowed word types list above
        for w in parts_of_speech_tagging:
            if w[1][0] in allowed_word_types:
                all_words.append(w[0].lower())


    for row in df_news_train_neg.itertuples(index=False):

        text = row.sub_header

        # tuple of text and label
        documents.append((text, "neg"))

        text = pre_processing_feature_text(text)

        # parts of speech tagging for each word
        parts_of_speech_tagging = nltk.pos_tag(text)

        # make a list of  all adjectives identified by the allowed word types list above
        for w in parts_of_speech_tagging:
            if w[1][0] in allowed_word_types:
                all_words.append(w[0].lower())

    return all_words, documents

def train_classifiers(training_set, testing_set):

    model_result_df = pd.DataFrame(columns=["Model","Training Set Accuracy","Test Set Accuracy"])
    # try:


    naivebayesclassifier_classifier = nltk.NaiveBayesClassifier.train(training_set)
    model_result = {
         'Model': 'NaiveBayesClassifier',
         'Training Set Accuracy': ((nltk.classify.accuracy(naivebayesclassifier_classifier, training_set)) * 100),
             'Test Set Accuracy': ((nltk.classify.accuracy(naivebayesclassifier_classifier, testing_set)) * 100)
        }

    model_result_df = model_result_df.append(model_result, ignore_index=True)


    # save model
    save_train_model(naivebayesclassifier_classifier, 'NB_clf.pickle')

    multinomialnb_classifier = SklearnClassifier(MultinomialNB())
    multinomialnb_classifier.train(training_set)
    model_result = {
         'Model': 'MultinomialNB',
         'Training Set Accuracy': ((nltk.classify.accuracy(naivebayesclassifier_classifier, training_set)) * 100),
             'Test Set Accuracy': ((nltk.classify.accuracy(naivebayesclassifier_classifier, testing_set)) * 100)
        }

    model_result_df = model_result_df.append(model_result, ignore_index=True)

    # save model
    save_train_model(multinomialnb_classifier, 'MNB_clf.pickle')

    bernoullinb_classifier = SklearnClassifier(BernoulliNB())
    bernoullinb_classifier.train(training_set)
    model_result = {
         'Model': 'BernoulliNB',
         'Training Set Accuracy': ((nltk.classify.accuracy(bernoullinb_classifier, training_set)) * 100),
             'Test Set Accuracy': ((nltk.classify.accuracy(bernoullinb_classifier, testing_set)) * 100)
        }

    model_result_df = model_result_df.append(model_result, ignore_index=True)

    # save model
    save_train_model(bernoullinb_classifier, 'BNB_clf.pickle')

    logisticregression_classifier = SklearnClassifier(LogisticRegression())
    logisticregression_classifier.train(training_set)
    model_result = {
         'Model': 'LogisticRegression',
         'Training Set Accuracy': ((nltk.classify.accuracy(logisticregression_classifier, training_set)) * 100),
             'Test Set Accuracy': ((nltk.classify.accuracy(logisticregression_classifier, testing_set)) * 100)
        }

    model_result_df = model_result_df.append(model_result, ignore_index=True)

    # save model
    save_train_model(logisticregression_classifier, 'LogReg_clf.pickle')

    sgdclassifier_classifier = SklearnClassifier(SGDClassifier())
    sgdclassifier_classifier.train(training_set)
    model_result = {
         'Model': 'SGDClassifier',
         'Training Set Accuracy': ((nltk.classify.accuracy(sgdclassifier_classifier, training_set)) * 100),
             'Test Set Accuracy': ((nltk.classify.accuracy(sgdclassifier_classifier, testing_set)) * 100)
        }

    model_result_df = model_result_df.append(model_result, ignore_index=True)

    # save model
    save_train_model(sgdclassifier_classifier, 'SGD_clf.pickle')

    svc_classifier = SklearnClassifier(SVC(kernel='linear',break_ties=False))
    svc_classifier.train(training_set)
    model_result = {
         'Model': 'SVC',
         'Training Set Accuracy': ((nltk.classify.accuracy(svc_classifier, training_set)) * 100),
             'Test Set Accuracy': ((nltk.classify.accuracy(svc_classifier, testing_set)) * 100)
        }

    model_result_df = model_result_df.append(model_result, ignore_index=True)

    # save model
    save_train_model(svc_classifier, 'SVC_clf.pickle')

    return model_result_df

def word_cloud_plot(word_features):

    text = ' '.join(word_features)
    wordcloud = WordCloud().generate(text)

    plt.figure(figsize = (15, 9))
    # Display the generated image:
    plt.imshow(wordcloud, interpolation= "bilinear")
    plt.axis("off")

    plt.title('Words cloud')
    plt.show()

def find_features(text, word_features):

    text = tokenize_text(text)

    features = {}
    for w in word_features:
        features[w] = (w in text)

    return features

# preprocessing data

def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)

def remove_apostrophe(text,contractions_dict):
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

    expanded_sentence = contractions_pattern.sub(expand_match, text)
    expanded_sentence = re.sub("'", "", expanded_sentence)
    return expanded_sentence

def pre_processing_text(text):
    # remove apostrophe
    text = remove_apostrophe(text, contractions_dict)

    # remove unwanted chracters
    unwanted = "!@#$;:!*%)(&^~-"
    text = ''.join( c for c in text if c not in unwanted )

    # remove tag
    text = re.sub('<[^<]+?>','', text)

    # correct all multiple white spaces to a single white space
    text = re.sub('[\s]+', ' ', text)

    #text to lower case
    text = text.lower()

    return text

def pre_processing_feature_text(text):

    # remove apostrophe
    text = remove_apostrophe(text, contractions_dict)

    # remove punctuations
    text = re.sub(r'[^(a-zA-Z)\s]', '', text)

    # correct all multiple white spaces to a single white space
    text = re.sub('[\s]+', ' ', text)

     # remove tag
    text = re.sub('<[^<]+?>','', text)

    # tokenize
    lemmatizer = WordNetLemmatizer()

    text = [lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in nltk.word_tokenize(text)]

    # remove stopwords
    text = [w for w in text if not w in stop_words]

    return text

def tokenize_text(text):
    words = pre_processing_text(text)
    # tokenize
    text = word_tokenize(words)
    # remove stopwords
    text = [w for w in text if not w in stop_words]
    return text

def start_process():

    # step to load data
    print("***** Load data set *****")
    df_news_train_data = load.load_data_v1()

    # step to pre processing
    print("***** Data pre processing *****")
    all_words, documents = construct_feature_word_list(df_news_train_data)

    # creating a frequency distribution of each adjectives.
    print("***** FreqDist data *****")
    bag_of_words = nltk.FreqDist(all_words)

    # listing the 5555 most frequent words
    print("***** Creating bag of words data *****")
    word_features = list(bag_of_words.keys())[:5555]

    # Word cloud plot
    # word_cloud_plot(word_features)

    # Creating features for each review
    print("***** Creating features for each review *****")
    feature_sets = [(find_features(rev, word_features), category) for (rev, category) in documents]

    # Shuffling the documents to create train test sets
    print("***** Shuffling the documents to create train test sets *****")
    random.shuffle(feature_sets)

    print("***** Create training and testing data set *****")
    training_set ,testing_set = train_test_split(feature_sets,test_size=0.2, random_state=42)
    print('Training Set {} ; Testing Set {}'.format(len(training_set),len(testing_set)))

    print("***** Save training test data set *****")
    save_data(testing_set, 'testing_set.pickle')

    print("***** Save training word features *****")
    save_data(word_features, 'word_features.pickle')

    return training_set ,testing_set

# Training And Test Set Accuracy Plot
def model_accuracy(model_result_df):
    fig = go.Figure(data=[
        go.Bar(name='Training Set Accuracy', x=model_result_df["Model"], y=model_result_df["Training Set Accuracy"]),
        go.Bar(name='Test Set Accuracy', x=model_result_df["Model"], y=model_result_df["Test Set Accuracy"])
    ])
    # Change the bar mode
    fig.update_layout(barmode='group',
                      xaxis_title='Model',yaxis_title='Accuracy',
                      legend=dict(x=.3, y=1.1),legend_orientation="h")
    fig.show()


if __name__ == '__main__':

    print("***** Started pre processing steps *****")
    training_set ,testing_set = start_process()

    print("***** Train  models *****")
    model_result_df = train_classifiers(training_set ,testing_set)
    print(model_result_df)

    # call method to train models
    print("***** Training And Test Set Accuracy Plot *****")
    model_accuracy(model_result_df)
