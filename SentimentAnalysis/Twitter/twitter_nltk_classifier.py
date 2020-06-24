import sys
import os
import pandas as pd
import pickle

# import created packages
import data_load as load
import pre_processing_data as pre_processing
from pre_processing_data import NltkPreProcessingModel

from sklearn.metrics import f1_score

# Ensemble
from nltk.classify import ClassifierI
from statistics import mode

# database
sys.path.append(os.path.abspath('../../Database'))
import database_log

from nltk.tokenize import TweetTokenizer
tknzr = TweetTokenizer()

pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 10)


# Ensemble model class
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

# function to load models given file path
def load_model(file_path):

    classifier_f = open(file_path, "rb")
    classifier = pickle.load(classifier_f)
    classifier_f.close()
    return classifier

# load save model to ensemble
def get_ensemble_models():

    model_path = '../TrainModel/Twitter/Nltk/'

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
    # ensemble_clf = EnsembleClassifier(BNB_clf, LogReg_clf, SGD_clf)
    # NB_clf, MNB_clf, BNB_clf, LogReg_clf, SGD_clf
    # BNB_clf, LogReg_clf, NB_clf
    ensemble_clf = EnsembleClassifier(BNB_clf, LogReg_clf, NB_clf)

    return ensemble_clf

# load testing data set
def load_save_dataset(file_name):

    file_path = '../TrainingData/TwitterData/' + file_name

    with open(file_path, 'rb') as f:
        dataset = pickle.load(f)

    return dataset

# Live Sentiment Analysis
def sentiment_analyzer(text, ensemble_clf):

    text_tokens = NltkPreProcessingModel.clean_text(tknzr.tokenize(text))
    tokens_dict = dict([token, True] for token in text_tokens)

    return ensemble_clf.classify(tokens_dict), ensemble_clf.confidence(tokens_dict)

def predict_sentiment(ensemble_clf, word_features):

    # load test data set
    training_set_data, testing_set_data = load.load_data()

    # Get test data set sentiment
    for index, row in testing_set_data.iterrows():

        tweet = row["tweet"]
        classify, confidence = sentiment_analyzer(tweet, ensemble_clf)
        print("classify - {} , confidence - {}".format(classify, confidence))

        break


if __name__ == '__main__':

    print("***** loading tweet data *****")
    testing_set_data = load_save_dataset('testing_set_data.pickle')
    # print(type(testing_set))

    print("***** loading word features *****")
    word_features = load_save_dataset('word_features.pickle')

    print("***** save train models to a ensemble *****")
    ensemble_clf = get_ensemble_models()

    print("***** predict tweets sentiment *****")
    predict_sentiment(ensemble_clf, word_features)
