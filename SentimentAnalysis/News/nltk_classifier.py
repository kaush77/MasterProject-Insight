import sys
import os
import pandas as pd
import pickle

# import created packages
import data_load as load
import nltk_training_model

from sklearn.metrics import f1_score

# Ensemble
from nltk.classify import ClassifierI
from statistics import mode

# database
sys.path.append(os.path.abspath('../../'))
import Database.database_log as database_log

# function to load models given file path
def load_model(file_path):

    classifier_f = open(file_path, "rb")
    classifier = pickle.load(classifier_f)
    classifier_f.close()
    return classifier

# load testing data set
def load_save_dataset(file_name):

    file_path = '../TrainingData/News/' + file_name

    with open(file_path, 'rb') as f:
        dataset = pickle.load(f)

    return dataset

# Ensemble model class
class EnsembleClassifier(ClassifierI):

    def __init__(self, *classifiers):
        self._classifiers = classifiers

    # returns the classification based on majority of votes
    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)

    # measurement the degree of confidence in the classification
    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)

        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf

 # save model to ensemble
def get_ensemble_models():

    model_path = "../TrainModel/News/Nltk/"

    # "***** loading train models *****"
    # Naive Bayes Classifier
    NB_clf = load_model(model_path + 'NB_clf.pickle')

    # Multinomial Naive Bayes Classifier
    MNB_clf = load_model(model_path + 'MNB_clf.pickle')

    # Bernoulli  Naive Bayes Classifier
    BNB_clf = load_model(model_path + 'BNB_clf.pickle')

    # Logistic Regression Classifier
    LogReg_clf = load_model(model_path + 'LogReg_clf.pickle')

    # Stochastic Gradient Descent Classifier
    SGD_clf = load_model(model_path + 'SGD_clf.pickle')

    # SVC Classifier
    SVC_clf = load_model(model_path + 'SVC_clf.pickle')

    # Initializing the ensemble classifier
    # ensemble_clf = EnsembleClassifier(NB_clf, MNB_clf, BNB_clf, LogReg_clf, SGD_clf) # level -3
    # ensemble_clf = EnsembleClassifier(BNB_clf, LogReg_clf, SGD_clf) # level - 2
    ensemble_clf = EnsembleClassifier(BNB_clf, LogReg_clf, SVC_clf) # level - 1

    return ensemble_clf

# Live Sentiment Analysis
def sentiment_analyzer(text, ensemble_clf, word_features):

    features = nltk_training_model.find_features(text, word_features)
    return ensemble_clf.classify(features), ensemble_clf.confidence(features)


if __name__ == '__main__':

    print("***** loading word features *****")
    word_features = load_save_dataset('word_features.pickle')

    print("***** save train models to a ensemble *****")
    ensemble_clf = get_ensemble_models()

    text_list = ["Silver futures up 5.76% in afternoon trade",
    "Stock market live updates: Stocks jump after last week's sell-off, but coronavirus fears remain",
              "Intel surges 8% on an earnings beat and better-than-expected forecast",
              "I do enjoy my job",
                  "Apple stock rebounds almost 7% to head for strongest day since 2018",
                  "I feel bad",
              "Global Manufacturing Shrinks Most Since 2009 on Virus Pain" ,
              "Asia shares slip, stimulus talk offers support",
             "Japan’s Nikkei was still down 2.8%, but futures had come well off their lows. MSCI’s broadest index of Asia-Pacific shares outside Japan lost 0.3%."]

    for index in range(len(text_list)):
        classify,confidence = sentiment_analyzer(text_list[index], ensemble_clf, word_features)
        print("classify - {} ; confidence - {}".format(classify, confidence))
