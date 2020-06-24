import sys
import os
import re

# Data Analysis
import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

import nltk
import random

from wordcloud import WordCloud
import pre_processing_data as pre_processing
import data_load as load
from pre_processing_data import NltkPreProcessingModel

# Model Selection and Validation

from nltk.classify.scikitlearn import SklearnClassifier
import pickle
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC
from nltk.classify import ClassifierI
from statistics import mode
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
stop_words = list(set(stopwords.words('english')))

from nltk.tokenize import TweetTokenizer
tknzr = TweetTokenizer()

# database
sys.path.append(os.path.abspath('../../Database'))
import database_log

from nltk import FreqDist
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

# Ensemble
from nltk.classify import ClassifierI
from statistics import mode

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 10)


# save train model
def save_train_model(classifier, file_name):

    file_path = "../TrainModel/Twitter/Nltk/" + file_name
    save_classifier = open(file_path, 'wb')
    pickle.dump(classifier, save_classifier)
    save_classifier.close()

# save testing model data set
def save_data(data_set, file_name):

    file_path = '../TrainModel/Twitter/Nltk/' + file_name

    with open(file_path, 'wb') as f:
        pickle.dump(data_set, f)

def tokenize_dataset(train_dataset):

    positive_tokens_list = []
    negative_tokens_list = []

    train_positive_df = train_dataset.loc[train_dataset['label'] == 1]
    train_negative_df = train_dataset.loc[train_dataset['label'] == 0]


    for row in train_positive_df.itertuples(index=False):
        positive_tokens_list.append(NltkPreProcessingModel.clean_text(tknzr.tokenize(row.tweet),stop_words))

    for row in train_negative_df.itertuples(index=False):
        negative_tokens_list.append(NltkPreProcessingModel.clean_text(tknzr.tokenize(row.tweet),stop_words))


    positive_tokens_for_model = NltkPreProcessingModel.get_tweets_for_model(positive_tokens_list)
    negative_tokens_for_model = NltkPreProcessingModel.get_tweets_for_model(negative_tokens_list)

    positive_dataset = [(text_dict, "Positive")
                             for text_dict in positive_tokens_for_model]

    negative_dataset = [(text_dict, "Negative")
                             for text_dict in negative_tokens_for_model]

    token_dataset = positive_dataset + negative_dataset

    # Frequency distribution and plot
    freq_dist_plot_data = positive_tokens_list + negative_tokens_list
    # FreqDistPlot(freq_dist_plot_data)

    return token_dataset

# Frequency distribution and plot
def FreqDistPlot(wordlist):

    all_pos_words = NltkPreProcessingModel.get_all_words(wordlist)

    freq_dist_pos = FreqDist(all_pos_words)

    # creating a frequency distribution of positive word.

    bag_of_words = nltk.FreqDist(freq_dist_pos)

    # listing most frequent words
    word_features = list(bag_of_words.keys())[:5555]

    text = ' '.join(word_features)
    wordcloud = WordCloud().generate(text)

    plt.figure(figsize = (15, 9))
    # Display the generated image:
    plt.imshow(wordcloud, interpolation= "bilinear")
    plt.axis("off")

    plt.title('Positive Words')
    plt.show()

def prepare_model_dataset(token_dataset):
    random.shuffle(token_dataset)
    train_data,test_data = train_test_split(token_dataset,test_size=0.3)
    print("train_data - {} ; test_data - {}".format(len(train_data),len(test_data)))
    return train_data, test_data

def load_training_dataset():
    train_data = load.load_data_v6()
    token_dataset = tokenize_dataset(train_data)
    train_data, test_data = prepare_model_dataset(token_dataset)
    return train_data, test_data


def train_classifiers(training_set, testing_set):

    try:

        model_result_df = pd.DataFrame(columns=["Model","Training Set Accuracy","Test Set Accuracy"])

        columns=["classifier","fpr","tpr"]
        roc_curve_data_df = pd.DataFrame(columns=columns)


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

    except Exception as error:
        print(error)


# Training And Test Set Accuracy Plot
def plot_accuracy(model_result_df):

    fig = go.Figure(data=[
        go.Bar(name='Training Set Accuracy', x=model_result_df["Model"], y=model_result_df["Training Set Accuracy"]),
        go.Bar(name='Test Set Accuracy', x=model_result_df["Model"], y=model_result_df["Test Set Accuracy"])
    ])
    # Change the bar mode
    fig.update_layout(barmode='group',
                  xaxis_title='Model',
                  yaxis_title='Accuracy',
                 legend=dict(x=.3, y=1.1),legend_orientation="h")

    fig.show()


if __name__ == '__main__':

    print("***** Started pre processing steps *****")
    train_data, test_data = load_training_dataset()

    print("***** Save training test data set *****")
    save_data(test_data, 'testing_set_data.pickle')

    print("***** Started training  models *****")
    model_result_df = train_classifiers(train_data, test_data)
    print(model_result_df)

    # Training And Test Set Accuracy Plot
    plot_accuracy(model_result_df)
