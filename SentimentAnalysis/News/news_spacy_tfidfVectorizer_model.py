import nltk
import re
import pandas as pd
import numpy as np
import pickle
import spacy
import string
from spacy.lang.en import English

# ML Packages
from sklearn.base import TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC, SVC
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

# replace apostrophe/short words in python
from contractions import contractions_dict

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
stop_words = list(set(stopwords.words('english')))

# Ensemble
from nltk.classify import ClassifierI
from statistics import mode

from sklearn.metrics import make_scorer, accuracy_score, f1_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix, roc_auc_score, recall_score, precision_score

import data_load as load

import en_core_web_sm
from spacy.lang.en.stop_words import STOP_WORDS
nlp = en_core_web_sm.load()

# To build a list of stop words for filtering
stopwords = list(STOP_WORDS)

# Lemmatize with POS Tag
from nltk.corpus import wordnet
from nltk.stem.wordnet import WordNetLemmatizer

# Creating a Spacy Parser
punctuations = string.punctuation
parser = English()

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

import warnings
warnings.filterwarnings('ignore')


# loading and splitting training data set
def preparing_training_data():
    # load data set
    training_dataset = load.load_data_v1()

    x_text = training_dataset['sub_header']
    y_labels = training_dataset['label']

    X_train, X_test, y_train, y_test = train_test_split(x_text, y_labels, test_size=0.1, random_state=42)

    return X_train, X_test, y_train, y_test

# Prepare Training Dataset
# What's -> What is
def remove_apostrophe(text, contractions_dict):
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

    # remove special characters
    text = re.sub("[^a-zA-Z#]", ' ', text)

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

def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)

# Custom transformer using spaCy
class predictors(TransformerMixin):

    def transform(self, X, **transform_params):
        return [clean_text(text) for text in X]

    def fit(self, X, y, **fit_params):
        return self

    def get_params(self, deep=True):
        return {}

# Basic function to clean the text
def clean_text(text):
    return text.strip().lower()

def tokenizer(text):
    text = pre_processing_text(text)
    text_tokens = word_tokenize(text)
    return text_tokens

# Tfidf pipeline
# Create the  pipeline to clean, tokenize, vectorize, and classify using "Count Vectorizor"
def tfidf_vectorizer_pipeline(classifier):

     # tokenize words needs to be lemmatized and filtered for pronouns, stopwords and punctuations
    tf_vectorizer = TfidfVectorizer(tokenizer=tokenizer,min_df=5,max_df=0.95,
                                sublinear_tf = True,use_idf = True,ngram_range=(1, 1))

    pipe_tfidf_vectorizer = Pipeline([("cleaner", predictors()),
                                     ('vectorizer', tf_vectorizer),
                                     ('classifier', classifier)])
    return pipe_tfidf_vectorizer

# get roc curve data
def get_roc_curve(model, X, y):
    pred_proba = model.predict_proba(X)[:, 1]
    fpr, tpr, _ = roc_curve(y, pred_proba)

    return fpr, tpr

# training tfidf vectorizer model
def train_tfidf_vectorizor_model(X_train, X_test, y_train, y_test,classifier):

    # load tfidf vectorizer pipeline
    model = tfidf_vectorizer_pipeline(classifier)

    gs_cv_model = GridSearchCV(model,[{ }] ,scoring='accuracy',
                               cv=10,verbose=1,n_jobs=-1)
    
    # fit data set
    gs_cv_model.fit(X_train, y_train)    
    train_accuracy = gs_cv_model.best_score_

    clf = gs_cv_model.best_estimator_ 
    test_accuracy = clf.score(X_test, y_test)

    # get roc curve data
    fpr, tpr = get_roc_curve(gs_cv_model, X_test, y_test)  

    return model,train_accuracy,test_accuracy,fpr, tpr

# Save trained models
def save_train_model(model, file_name):

    file_path =  "../TrainModel/News/SpacyTfidf/" + file_name
    save_classifier = open(file_path, 'wb')
    pickle.dump(model, save_classifier)
    save_classifier.close()

# train model
def train_model():

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

        model,train_accuracy,test_accuracy,fpr, tpr = train_tfidf_vectorizor_model(
                                        X_train, X_test, y_train, y_test,classifier)

        model_result = {
         'Model': classifier_key,
         'Training Set Accuracy': ((train_accuracy) * 100),
             'Test Set Accuracy': ((test_accuracy) * 100)
        }

        model_result_df = model_result_df.append(model_result, ignore_index=True)

        roc_curve_df = pd.DataFrame()
        roc_curve_df["fpr"] = fpr
        roc_curve_df["tpr"] = tpr
        roc_curve_df["classifier"] = classifier_key

        roc_curve_data_df = roc_curve_data_df.append(roc_curve_df, ignore_index=True)

        # save spacy train models CountVectorizer
        save_train_model(model, classifier_key +".pickle")

    return model_result_df,roc_curve_data_df

# plots
def plot_roc_curve(roc_curve_data_df):
    unique_classifier_list = roc_curve_data_df.classifier.unique()

    fig = make_subplots(rows=2, cols=3, start_cell="top-left",vertical_spacing=0.15,horizontal_spacing=0.12)

    row_val = 1
    col_val = 1
    count = 1

    for classifier in unique_classifier_list:
        plot_df = roc_curve_data_df.loc[roc_curve_data_df['classifier'] == classifier]

        if col_val > 3:
            row_val += 1
            col_val = 1

        fig.add_trace(go.Scatter(x=plot_df["fpr"], y=plot_df["tpr"], name=classifier),
                  row=row_val, col=col_val)

        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1],
                        mode='lines',line=dict(color='black', width=2,dash='dash'),
                        showlegend=False),row=row_val, col=col_val)

        fig.update_xaxes(title_text="FPR", row=row_val, col=col_val)
        fig.update_yaxes(title_text="TPR", row=row_val, col=col_val)

        col_val +=1
        count +=1

    fig.update_layout(title_text="Roc curve",legend_orientation = 'h',legend=dict(x=0.1, y=1.2))
    fig.show()

def plot_one_roc_curve(roc_curve_data_df):
    unique_classifier_list = roc_curve_data_df.classifier.unique()

    fig = make_subplots(rows=1, cols=1, start_cell="top-left")

    for classifier in unique_classifier_list:
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
def plot_accuracy(model_result_df):

    fig = go.Figure(data=[
        go.Bar(name='Training Set Accuracy', x=model_result_df["Model"], y=model_result_df["Training Set Accuracy"]),
        go.Bar(name='Test Set Accuracy', x=model_result_df["Model"], y=model_result_df["Test Set Accuracy"])
    ])

    fig.update_layout(barmode='group',
                      xaxis_title='Model', yaxis_title='Accuracy',
                      legend=dict(x=.3, y=1.1),legend_orientation="h")

    fig.show()

# predict logic
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

    model_path = "../TrainModel/News/SpacyTfidf/"

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
    ensemble_clf = EnsembleClassifier(SVC_clf,MNB_clf,BNB_clf)


    return ensemble_clf

# Live Sentiment Analysis
def sentiment_analyzer(text):
    ensemble_clf = get_ensemble_models()
    text = pre_processing_text(text)
    return ensemble_clf.classify(text), ensemble_clf.confidence(text)


if __name__ == "__main__":

    print("***** Train  models *****")
    model_result_df,roc_curve_data_df = train_model()
    print(model_result_df)

    print("***** Plot roc curve *****")
    plot_roc_curve(roc_curve_data_df)
    plot_one_roc_curve(roc_curve_data_df)

    print("***** Training And Test Set Accuracy Plot *****")
    plot_accuracy(model_result_df)

    # Predict Sentiment
    text_list = ["Stock market live updates: Stocks jump after last week's sell-off, but coronavirus fears remain",
              "Intel surges 8% on an earnings beat and better-than-expected forecast",
              "I do enjoy my job",
                  "Apple stock rebounds almost 7% to head for strongest day since 2018",
                  "I feel bad",
              "Global Manufacturing Shrinks Most Since 2009 on Virus Pain"]

    # load model and add to ensemble
    ensemble_clf = get_ensemble_models()

    for index in range(len(text_list)):
        classify,confidence = sentiment_analyzer(text_list[index])
        print("classify - {}; confidence - {}".format(classify, confidence))
