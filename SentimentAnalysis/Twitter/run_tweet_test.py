import twitter_nltk_classifier
import pre_processing_data as pre_processing
from sklearn.base import TransformerMixin

from pre_processing_data import SpacyPreProcessingModel
import twitter_spacy_countVectorizer_model
import twitter_spacy_tfidfVectorizer_model

# Ensemble
from nltk.classify import ClassifierI
from statistics import mode
import string

from spacy.lang.en import English
import en_core_web_sm
from spacy.lang.en.stop_words import STOP_WORDS
nlp = en_core_web_sm.load()

# To build a list of stop words for filtering
stopwords = list(STOP_WORDS)

punctuations = string.punctuation
parser = English()


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

# words tokenizer, remove pronoun, punctuations
def tokenizer(tweet):

    tweet = SpacyPreProcessingModel.clean_text(tweet)
    tweet_tokens = nlp(tweet)

    # Lemma that are NOUN, VERB, ADV
    tweet_tokens = [word.lemma_.lower().strip() if word.pos_ in("NOUN","VERB","ADV") else word.lower_ for word in tweet_tokens]

    # Stop words and Punctuation In List Comprehension
    tweet_tokens = [word for word in tweet_tokens if word not in stopwords and word not in punctuations]

    return tweet_tokens

# Basic function to clean the text
def clean_text(text):
    return text.strip().lower()

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

    model_path = '../TrainModel/Twitter/Spacy_CountVectorizer/'

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
    ensemble_clf = EnsembleClassifier(SVC_clf)


    return ensemble_clf

def load_model(file_path):

    classifier_f = open(file_path, "rb")
    classifier = pickle.load(classifier_f)
    classifier_f.close()
    return classifier


if __name__ == '__main__':

    # tweet = "thanks for #lyft credit i can't use because they don't offer wheelchair vans in pdx.    " \
    #         "#disapointed #getthanked"
    # tweet ="Intel surges 8% on an earnings beat and better-than-expected forecast"
    #
    # print("\n---------text blob------------")
    # # text blob
    # sentiment_score, sentiment_value = twitter_textblob.textblob_sentiment(tweet)
    # print("sentiment_score - [{}];  sentiment - [{}] ".format(sentiment_score, sentiment_value))
    #
    # print("\n--------vader-------------")
    # # vader
    # x_pos, x_neu, x_neg, x_comp, sentiment = twitter_vader_classifier.vader_sentiment_analyzer_scores(tweet)
    # print("positive - [{}] ; neutral - [{}] ; negative - [{}] ; compound - [{}] ; sentiment - [{}]"
    #       .format(x_pos, x_neu, x_neg, x_comp, sentiment))

    print("\n--------nltk-------------")

    tweet_list = [".@benbreitholtz coming up on #WDYM! https://t.co/FnJb7ypbX3",
    "I love this food",
    "France's CAC 40 index drops 2.7% to 3,036 http://t.co/fWANYWW #CAC #CAC40 #Trading",
    "Harnessing innovation for a healthier planet. A lot of people have been sharing technology-driven solutions for C… https://t.co/DlenQTII0B"]

    ensemble_clf = twitter_nltk_classifier.get_ensemble_models()
    for tokens in range(len(tweet_list)):
        classify, confidence = twitter_nltk_classifier.sentiment_analyzer(tweet_list[tokens], ensemble_clf)
        print("classify - {} , confidence - {}".format(classify, confidence))
    #
    # print("\n--------twitter_word2vec-------------")
    #
    # for tokens in range(len(tweet_list)):
    #     model, w2vmodel, tfidf = twitter_word2vec_classifier_model.load_prediction_model_parameters()
    #
    #     prediction = twitter_word2vec_classifier_model.predict(model, w2vmodel, tfidf, tweet_list[tokens])
    #     print("classify - {} ".format(prediction))

    print("\n--------spacy count vector-------------")

    for tokens in range(len(tweet_list)):
        classify, confidence = twitter_spacy_countVectorizer_model.sentiment_analyzer(tweet_list[tokens])
        print("classify - {} , confidence - {}".format(classify, confidence))

    print("\n--------spacy tfidf-------------")
    for tokens in range(len(tweet_list)):
        classify, confidence = twitter_spacy_tfidfVectorizer_model.sentiment_analyzer(tweet_list[tokens])
        print("classify - {} , confidence - {}".format(classify, confidence))
