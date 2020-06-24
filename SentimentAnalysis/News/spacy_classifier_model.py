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


# Custom transformer using spaCy
class predictors(TransformerMixin):

    def transform(self, X, **transform_params):
        return [clean_text(text) for text in X]

    def fit(self, X, y, **fit_params):
        return self

    def get_params(self, deep=True):
        return {}


# words tokenizer, remove pronoun, punctuations
def tokenizer(text):

    text = pre_processing.pre_processing_text(text)
    text_tokens = parser(text)

    # Lemma that are not pronouns
    text_tokens = [word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in text_tokens]
    # Stop words and Punctuation In List Comprehension
    text_tokens = [word for word in text_tokens if word not in stopwords and word not in punctuations]

    return text_tokens


# Basic function to clean the text
def clean_text(text):
    return text.strip().lower()

# loading and splitting training data set
def preparing_training_data():

    # load data set
    # training_set_data, texts_test = load.load_data_v1()

    training_set_data = load.load_data_v3()

    x_text = training_set_data['sub_header']
    y_labels = training_set_data['label']

    X_train, X_test, y_train, y_test = train_test_split(x_text, y_labels, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test


# classifier model
def classifier_model():

    classifier = LinearSVC()
    return classifier


# tokenize words needs to be lemmatized and filtered for pronouns, stopwords and punctuations
def vectorization():

    vectorizer = CountVectorizer(tokenizer=tokenizer, ngram_range=(1, 1))
    return vectorizer


# tokenize words needs to be lemmatized and filtered for pronouns, stopwords and punctuations
def tfidf_vectorizer():

    tf_vectorizer = TfidfVectorizer(tokenizer=tokenizer)
    return tf_vectorizer


# Create the  pipeline to clean, tokenize, vectorize, and classify using "Count Vectorizor"
def vectorization_pipeline():

    pipe_count_vectorizer = Pipeline([("cleaner", predictors()),
                                     ('vectorizer', vectorization()),
                                     ('classifier', classifier_model())])
    return pipe_count_vectorizer


# Create the  pipeline to clean, tokenize, vectorize, and classify using "tfidf"
def tfidf_vectorizer_pipeline():

    pipe_tfidf_vectorizer = Pipeline([("cleaner", predictors()),
                                     ('vectorizer', tfidf_vectorizer()),
                                     ('classifier', classifier_model())])
    return pipe_tfidf_vectorizer


# training count vectorizer model
def train_count_vectorizor_model():

    # load split training data set
    X_train, X_test, y_train, y_test = preparing_training_data()

    # load count vectorizer pipeline
    pipe_count_vectorizer = vectorization_pipeline()

    # fit data set
    pipe_count_vectorizer.fit(X_train, y_train)

    # predicting with a test data set
    text_prediction = pipe_count_vectorizer.predict(X_test)

    # Prediction Results 1 = Positive review ; 0 = Negative review
    # for (text, prediction) in zip(X_test, text_prediction):
    #     print(text, "Prediction=>", prediction)
    #     break

    # Accuracy
    print("CountVectorizer Accuracy")
    X_test_y_test = pipe_count_vectorizer.score(X_test,y_test)
    X_test_text_prediction = pipe_count_vectorizer.score(X_test,text_prediction)
    X_train_y_train= pipe_count_vectorizer.score(X_train,y_train)

    print("X_Y_test - [{:.5f}] ; X_test_prediction - [{:.5f}] ; X_Y_train -[{:.5f}]".
            format(X_test_y_test, X_test_text_prediction, X_train_y_train))

    return pipe_count_vectorizer


# training tfidf vectorizer model
def train_tfidf_vectorizor_model():

    # load split training data set
    X_train, X_test, y_train, y_test = preparing_training_data()

    # load count vectorizer pipeline
    pipe_tfidf_vectorizer = tfidf_vectorizer_pipeline()

    # fit data set
    pipe_tfidf_vectorizer.fit(X_train, y_train)

    # predicting with a test data set
    text_prediction = pipe_tfidf_vectorizer.predict(X_test)

    # Prediction Results 1 = Positive review ; 0 = Negative review
    # for (text, prediction) in zip(X_test, text_prediction):
    #     print(text, "Prediction=>", prediction)
    #     break

    # Accuracy
    print("Tfidf_Vectorizer Accuracy")
    X_test_y_test = pipe_tfidf_vectorizer.score(X_test,y_test)
    X_test_text_prediction = pipe_tfidf_vectorizer.score(X_test,text_prediction)
    X_train_y_train= pipe_tfidf_vectorizer.score(X_train,y_train)

    print("X_Y_test - [{:.5f}] ; X_test_prediction - [{:.5f}] ; X_Y_train -[{:.5f}]".
            format(X_test_y_test, X_test_text_prediction, X_train_y_train))

    return pipe_tfidf_vectorizer

# save train model
def save_train_model(model, file_name):

    file_path = "../TrainModel/News/" + file_name
    save_classifier = open(file_path, 'wb')
    pickle.dump(model, save_classifier)
    save_classifier.close()


# train model
def process_model():

    pipe_count_vectorizer = train_count_vectorizor_model()
    pipe_tfidf_vectorizer = train_tfidf_vectorizor_model()

    # save spacy train models CountVectorizer
    save_train_model(pipe_count_vectorizer, "CountVectorizer.pickle")

    # save spacy train models TfidfVectorizer
    save_train_model(pipe_tfidf_vectorizer, "TfidfVectorizer.pickle")

# endregion training twitter model using spacy


# function to load models given file path
def load_model(file_path):

    classifier_f = open(file_path, "rb")
    classifier = pickle.load(classifier_f)
    classifier_f.close()
    return classifier


def get_spacy_model():

    model_path = '../TrainModel/News/'

    # load spacy train models CountVectorizer
    pipe_count_vectorizer = load_model(model_path +"CountVectorizer.pickle")

    # load spacy train models TfidfVectorizer
    pipe_tfidf_vectorizer = load_model(model_path +"TfidfVectorizer.pickle")

    return pipe_count_vectorizer, pipe_tfidf_vectorizer


def predict_sentiment(text):

    # get saved spacy model
    pipe_count_vectorizer, pipe_tfidf_vectorizer = get_spacy_model()

    # predict using CountVectorizer
    count_vectorizer_prediction = pipe_count_vectorizer.predict(text)

    # predict using TfidfVectorizer
    tfidf_vectorizer_prediction = pipe_tfidf_vectorizer.predict(text)

    return count_vectorizer_prediction, tfidf_vectorizer_prediction

# endregion predict sentiment


if __name__ == "__main__":

    # train model
    process_model()

    tweet_list = ["thanks for #lyft credit i can't use cause they don't offer wheelchair vans in pdx. #disapointed #getthanked",
                  "I do enjoy my job",
                  "What a poor product!,I will have to get a new one",
                  "I feel amazing!",
              "thanks for lyft credit i can not use because they do not offer wheelchair vans in pdx disapointed getthanked"]

    # 0 negative , 1 positive
    count_vectorizer_prediction, tfidf_vectorizer_prediction= predict_sentiment(tweet_list)
    print("CountVectorizer prediction - {} , tdif prediction - {}".format(count_vectorizer_prediction, tfidf_vectorizer_prediction))
