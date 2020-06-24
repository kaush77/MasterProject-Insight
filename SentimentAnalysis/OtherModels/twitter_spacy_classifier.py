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
from pre_processing_data import SpacyPreProcessingModel

# Ensemble
from nltk.classify import ClassifierI
from statistics import mode

import pickle
import spacy
import string
from spacy.lang.en import English

import en_core_web_sm
from spacy.lang.en.stop_words import STOP_WORDS
nlp = en_core_web_sm.load()

# To build a list of stop words for filtering
stopwords = list(STOP_WORDS)

# Creating a Spacy Parser
punctuations = string.punctuation
parser = English()



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
def get_ensemble_models(model):

    model_path = ""

    if model == "CountVector":
        model_path = '../TrainModel/Twitter/Spacy_CountVectorizer/'
    elif model == "TfidfVector":
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
    ensemble_clf = EnsembleClassifier(SVC_clf, LogReg_clf, MNB_clf, BNB_clf, SGD_clf)

    return ensemble_clf


# Live Sentiment Analysis
def sentiment_analyzer(text, ensemble_clf):

    text = SpacyPreProcessingModel.clean_text(text[0])

    # text_tokens = nlp(text)
    #
    # # Lemma that are NOUN, VERB, ADV
    # text_tokens = [word.lemma_.lower().strip() if word.pos_ in("NOUN","VERB","ADV") else word.lower_ for word in text_tokens]
    #
    # # Stop words and Punctuation In List Comprehension
    # text_tokens = [word for word in text_tokens if word not in stopwords and word not in punctuations]

    print("classify - {}; confidence - {}".format(ensemble_clf.classify(text), ensemble_clf.confidence(text)))
    # return ensemble_clf.classify(text_tokens), ensemble_clf.confidence(text_tokens)


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

    tweet = pre_processing.pre_processing_spacy(tweet)
    tweet_tokens = parser(tweet)
    # Lemma that are not pronouns
    tweet_tokens = [word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in tweet_tokens]
    # Stop words and Punctuation In List Comprehension
    tweet_tokens = [word for word in tweet_tokens if word not in stopwords and word not in punctuations]

    return tweet_tokens

# Basic function to clean the text
def clean_text(text):
    return text.strip().lower()




def predict_sentiment(tweet):

    model_path = '../TrainModel/Twitter/'

    # load spacy train models CountVectorizer
    pipe_count_vectorizer = load_model(model_path +"CountVectorizer.pickle")

    # load spacy train models TfidfVectorizer
    pipe_tfidf_vectorizer = load_model(model_path +"TfidfVectorizer.pickle")

    # predict using CountVectorizer
    count_vectorizer_prediction = pipe_count_vectorizer.predict(tweet)

    # predict using TfidfVectorizer
    tfidf_vectorizer_prediction = pipe_tfidf_vectorizer.predict(tweet)

    return count_vectorizer_prediction, tfidf_vectorizer_prediction

# endregion predict sentiment


if __name__ == "__main__":

    tweet_list = ["thanks for #lyft credit i can't use cause they don't offer wheelchair vans in pdx. #disapointed #getthanked",
                  "I do enjoy my job",
                  "What a poor product!,I will have to get a new one",
                  "I feel amazing!",
              "thanks for lyft credit i can not use because they do not offer wheelchair vans in pdx disapointed getthanked"]
    #
    # count_vectorizer_prediction, tfidf_vectorizer_prediction= predict_sentiment(tweet_list)
    # print("CountVectorizer prediction - {} , tdif prediction - {}".format(count_vectorizer_prediction, tfidf_vectorizer_prediction))

    # model = "CountVector"
    model = "TfidfVector"

    ensemble_clf = get_ensemble_models(model)

    for index in range(len(tweet_list)):
        sentiment_analyzer(tweet_list[index], ensemble_clf)
