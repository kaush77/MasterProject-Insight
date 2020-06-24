import re
import pandas as pd
import numpy as np
import pickle
import spacy
import string

# ML Packages
from sklearn.base import TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline,make_pipeline

from sklearn.svm import LinearSVC, SVC
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier

from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords

from sklearn.metrics import make_scorer, accuracy_score, f1_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix, roc_auc_score, recall_score, precision_score

# Ensemble
from nltk.classify import ClassifierI
from statistics import mode

# load training dataset
import data_load as load
from pre_processing_data import SpacyPreProcessingModel

# plot
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt

# replace apostrophe/short words in python
from contractions import contractions_dict

from spacy.lang.en import English
import en_core_web_sm
from spacy.lang.en.stop_words import STOP_WORDS
nlp = en_core_web_sm.load()

# To build a list of stop words for filtering
stopwords = list(STOP_WORDS)

# Creating a Spacy Parser
punctuations = string.punctuation
parser = English()


# save train model
def save_train_model(model, file_name):

    file_path = "../TrainModel/Twitter/Spacy_TfidfVectorizer/" + file_name
    save_classifier = open(file_path, 'wb')
    pickle.dump(model, save_classifier)
    save_classifier.close()


# loading and splitting training data set
def preparing_training_data():
    # load data set
    train_data = load.load_data_v6()

    x_tweet = train_data['tweet']
    y_labels = train_data['label']

    X_train, X_test, y_train, y_test = train_test_split(x_tweet, y_labels, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test


# Custom transformer using spaCy
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
    tweet_tokens = [word.lemma_.lower().strip() if word.pos_ != "-PRON-" else word.lower_ for word in tweet_tokens]

    # Stop words and Punctuation In List Comprehension
    tweet_tokens = [word for word in tweet_tokens if word not in stopwords and word not in punctuations]

    return tweet_tokens


# Basic function to clean the text
def clean_text(text):
    return text.strip().lower()

# Create the  pipeline to clean, tokenize, vectorize, and classify using "Tfidf Vectorizor"
def tfidf_vectorizer_pipeline(classifier):

    # tokenize words needs to be lemmatized and filtered for pronouns, stopwords and punctuations
    tf_vectorizer = TfidfVectorizer(tokenizer=tokenizer,min_df=5,max_df=0.95,
                                sublinear_tf = True,use_idf = True,ngram_range=(1, 1))

    pipe_tfidf_vectorizer = Pipeline([("cleaner", predictors()),
                                     ('vectorizer', tf_vectorizer),
                                     ('classifier', classifier)])
    return pipe_tfidf_vectorizer


# get plot roc curve data
def get_roc_curve(model, X, y):
    pred_proba = model.predict_proba(X)[:, 1]
    fpr, tpr, _ = roc_curve(y, pred_proba)
    return fpr, tpr


# training tfidf vectorizer model
def train_tfidf_vectorizor_model(X_train, X_test, y_train, y_test,classifier):
    # load tfidf vectorizer pipeline
    model = tfidf_vectorizer_pipeline(classifier)

    # fit data set
    model.fit(X_train, y_train)

    # predicting with a test data set
    tweet_prediction = model.predict(X_test)

    # get roc curve data
    fpr, tpr = get_roc_curve(model, X_test, y_test)

    # Accuracy
    X_test_y_test = model.score(X_test,y_test)
    X_test_tweet_prediction = model.score(X_test,tweet_prediction)
    X_train_y_train= model.score(X_train,y_train)

    return model,X_test_y_test,X_train_y_train,fpr, tpr


def process_model():
    model_result_df = pd.DataFrame(columns=["Model","Training Set Accuracy","Test Set Accuracy"])

     # load split training data set
    X_train, X_test, y_train, y_test = preparing_training_data()

    classifiers_list = {"SVC":SVC(kernel="linear",break_ties=False, C=0.1,probability=True),
                        "LogisticRegression":LogisticRegression(),
                        "MultinomialNB":MultinomialNB(), "BernoulliNB":BernoulliNB(),
                        "SGDClassifier":SGDClassifier(loss="log", penalty='l2', alpha=1e-3, max_iter= 5, random_state=42)}


    columns=["classifier","fpr","tpr"]
    roc_curve_data_df = pd.DataFrame(columns=columns)

    for classifier_key, classifier in classifiers_list.items():

        model,X_test_y_test,X_train_y_train,fpr, tpr = train_tfidf_vectorizor_model(
                                        X_train, X_test, y_train, y_test,classifier)

        model_result = {
         'Model': classifier_key,
         'Training Set Accuracy': ((X_train_y_train) * 100),
             'Test Set Accuracy': ((X_train_y_train) * 100)
        }

        model_result_df = model_result_df.append(model_result, ignore_index=True)

        roc_curve_df = pd.DataFrame()
        roc_curve_df["fpr"] = fpr
        roc_curve_df["tpr"] = tpr
        roc_curve_df["classifier"] = classifier_key

        roc_curve_data_df = roc_curve_data_df.append(roc_curve_df, ignore_index=True)

        # save spacy train models TfidfVectorizer
        save_train_model(model, classifier_key +".pickle")

        print("***** Trained {} model *****".format(classifier_key))
    return model_result_df,roc_curve_data_df


# plot result

def plot_individual_classifiers_roc(roc_curve_data_df):
    unique_classifiers_list = roc_curve_data_df.classifier.unique()

    fig = make_subplots(rows=2, cols=3, start_cell="top-left",vertical_spacing=0.15,horizontal_spacing=0.12)

    row_val = 1
    col_val = 1

    for classifier in unique_classifiers_list:
        plot_df = roc_curve_data_df.loc[roc_curve_data_df['classifier'] == classifier]

        if col_val > 3:
            row_val += 1
            col_val = 1

        fig.add_trace(go.Scatter(x=plot_df["fpr"], y=plot_df["tpr"], name=classifier),
                  row=row_val, col=col_val)

        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1],
                        mode='lines',
                        line=dict(color='black', width=2,dash='dash'),
                        showlegend=False),row=row_val, col=col_val)

        fig.update_xaxes(title_text="FPR", row=row_val, col=col_val)
        fig.update_yaxes(title_text="TPR", row=row_val, col=col_val)

        col_val +=1

    fig.update_layout(title_text="Roc curve",
                      legend_orientation = 'h',legend=dict(x=0.1, y=1.2))
    fig.show()


def plot_classifiers_roc(roc_curve_data_df):

    unique_classifiers_list = roc_curve_data_df.classifier.unique()

    fig = make_subplots(rows=1, cols=1, start_cell="top-left")

    for classifier in unique_classifiers_list:
        plot_df = roc_curve_data_df.loc[roc_curve_data_df['classifier'] == classifier]
        auc_area = auc(roc_curve_data_df.loc[roc_curve_data_df['classifier'] == classifier]["fpr"],
                           roc_curve_data_df.loc[roc_curve_data_df['classifier'] == classifier]["tpr"])

        fig.add_trace(go.Scatter(x=plot_df["fpr"], y=plot_df["tpr"], name=classifier + ' (area = %0.2f)' % auc_area),
                  row=1, col=1)

        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1],
                        mode='lines',line=dict(color='black', width=2,dash='dash'),
                        showlegend=False))

    fig.update_layout(title_text="Roc curve",legend_title='<b> Classifier </b>',
                      xaxis_title='FPR',yaxis_title='TPR')

    fig.show()


# Training And Test Set Accuracy Plot
def model_accuracy_plot(model_result_df):
    fig = go.Figure(data=[
        go.Bar(name='Training Set Accuracy', x=model_result_df["Model"], y=model_result_df["Training Set Accuracy"]),
        go.Bar(name='Test Set Accuracy', x=model_result_df["Model"], y=model_result_df["Test Set Accuracy"])
    ])

    fig.update_layout(barmode='group',
                  xaxis_title='Model',
                  yaxis_title='Accuracy',
                  legend=dict(x=.3, y=1.1),legend_orientation="h")
    fig.show()


# Predict Sentiment

# function to load models given file path
def load_model(file_path):

    classifier_f = open(file_path, "rb")
    classifier = pickle.load(classifier_f)
    classifier_f.close()
    return classifier

# Ensemble model class
class EnsembleClassifier(ClassifierI):

    def __init__(self, *classifiers):
        self._classifiers = classifiers

    # returns the classification based on majority of votes
    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.predict(features)
            v = v.tolist()
            votes.append(v[0])
        return mode(votes)

    # measurement the degree of confidence in the classification
    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.predict(features)
            v = v.tolist()
            votes.append(v[0])

        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf

# load save model to ensemble
def get_ensemble_models():

    model_path = '../TrainModel/Twitter/Spacy_TfidfVectorizer/'

    # "***** loading train models *****"
    # SVC Classifier
    SVC_clf = load_model(model_path + 'SVC.pickle')

    # Logistic Regression Classifier
    LogReg_clf = load_model(model_path + 'LogisticRegression.pickle')

    # Multinomial Naive Bayes Classifier
    MNB_clf = load_model(model_path + 'MultinomialNB.pickle')

    # Bernoulli  Naive Bayes Classifier
    BNB_clf = load_model(model_path + 'BernoulliNB.pickle')

    # Stochastic Gradient Descent Classifier
    SGD_clf = load_model(model_path + 'SGDClassifier.pickle')

    # Initializing the ensemble classifier
    ensemble_clf = EnsembleClassifier(SVC_clf,LogReg_clf,BNB_clf)


    return ensemble_clf

# Live Sentiment Analysis
def sentiment_analyzer(text):
    ensemble_clf = get_ensemble_models()

    text = SpacyPreProcessingModel.clean_text(text)
    return ensemble_clf.classify(text), ensemble_clf.confidence(text)


if __name__ == "__main__":

    # train model
    print("***** Started training  models *****")
    model_result_df, roc_curve_data_df = process_model()
    print(model_result_df)

    # save model result
    model_result_df.to_csv("../TrainModel/Twitter/Spacy_TfidfVectorizer/TfidfVectorizerModelResult.csv")

    # plot roc curve
    plot_individual_classifiers_roc(roc_curve_data_df)

    plot_classifiers_roc(roc_curve_data_df)

    # Training And Test Set Accuracy Plot
    model_accuracy_plot(model_result_df)

    tweet_list = ["thanks for #lyft credit i can't use cause they don't offer wheelchair vans in pdx. #disapointed #getthanked",
                  "I do enjoy my job",
                  "What a poor product!,I will have to get a new one",
                  "I feel amazing!",
                  "thanks for lyft credit i can not use because they do not offer wheelchair vans in pdx disapointed getthanked",
                "Intel surges 8% on an earnings beat and better-than-expected forecast"]

    ensemble_clf = get_ensemble_models()

    for tokens in range(len(tweet_list)):
        classify, confidence = sentiment_analyzer(tweet_list[tokens])
        print("classify - {} , confidence - {}".format(classify, confidence))
