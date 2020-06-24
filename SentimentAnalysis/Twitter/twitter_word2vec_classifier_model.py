import pandas as pd
import numpy as np
import pickle
import h5py
import sys
import os
import re

# word2vec model gensim class
import gensim
from gensim.models.word2vec import Word2Vec

from tqdm import tqdm
tqdm.pandas(desc="progress-bar")

# tokenizer from nltk
from nltk.tokenize import TweetTokenizer
from nltk import word_tokenize

from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras.callbacks import Callback
from keras.models import model_from_json
from keras.callbacks import EarlyStopping

import pre_processing_data as pre_processing
import data_load as load

LabeledSentence = gensim.models.doc2vec.LabeledSentence

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

tokenizer = TweetTokenizer()

# database
sys.path.append(os.path.abspath('../../Database'))
import database_log


def tokenize(tweet):

    try:
        tweet = tweet.lower()
        tokens = tokenizer.tokenize(tweet)
        tokens = list(filter(lambda t: not t.startswith('@'), tokens))
        tokens = list(filter(lambda t: not t.startswith('#'), tokens))
        tokens = list(filter(lambda t: not t.startswith('http'), tokens))
        return tokens
    except:
        return 'NC'


def pre_process(dataset):

    dataset['tokens'] = dataset['tweet'].progress_map(tokenize)
    dataset = dataset[dataset.tokens != 'NC']
    dataset.reset_index(inplace=True)
    dataset.drop('index', inplace=True, axis=1)

    return dataset

# labeled dataset data
def labelize_data(tweets, label_type):
    labelized = []
    for i,v in tqdm(enumerate(tweets)):
        label = '%s_%s'%(label_type,i)
        labelized.append(LabeledSentence(v, [label]))

    return labelized

# splitting for training and testing
def prepare_training_data(dataset):

    x_train, x_test, y_train, y_test = train_test_split(np.array(dataset.head(len(dataset)).tokens),
                                                    np.array(dataset.head(len(dataset)).label), test_size=0.2)

    return x_train, x_test, y_train, y_test

# builidng word2vec vocabulary and training
def builidng_word2vec_training(data_labellised,n_dim):

    tweet_w2v = Word2Vec(size=n_dim, min_count=10)
    tweet_w2v.build_vocab([x.words for x in tqdm(data_labellised)])
    tweet_w2v.train([x.words for x in tqdm(data_labellised)],total_examples=tweet_w2v.corpus_count, epochs=tweet_w2v.iter)

    return tweet_w2v

# F-IDF matrix of data
def tfidf_matrix(data_labellised):

    vectorizer = TfidfVectorizer(analyzer=lambda x: x, min_df=10)
    matrix = vectorizer.fit_transform([x.words for x in data_labellised])
    tfidf = dict(zip(vectorizer.get_feature_names(), vectorizer.idf_))
    print ('vocab size :', len(tfidf))

    return tfidf

# Build tweet vector to give input to FFNN
def build_word_vector(tokens, size, tweet_w2v, tfidf):

    vec = np.zeros(size).reshape((1, size))
    count = 0.
    for word in tokens:
        try:
            #combining w2v vectors with tfidf value of words in the tweet.
            vec += tweet_w2v[word].reshape((1, size)) * tfidf[word]
            count += 1.
        except KeyError: # handling the case where the token is not  in the corpus.
            continue
    if count != 0:
        vec /= count
    return vec

# training model
def Training(train_vecs_w2v, y_train):

    nb_epoch=10
    batch_size = 128
    model = Sequential()
    model.add(Dense(32, activation='relu', input_dim=200))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

    # model.fit(train_vecs_w2v, y_train, epochs=10, batch_size=10000, verbose=2)
    model_history = model.fit(train_vecs_w2v, y_train,shuffle=True,validation_split=0.2,
                        epochs=nb_epoch, batch_size=batch_size, verbose=2,
                        callbacks=[EarlyStopping(monitor='val_loss', patience=5)])

    return model_history,model

# save word vector data
def save_w2vmodel(tweet_w2v):

    w2vmodel_path = "../TrainModel/Twitter/word2vec/w2vmodel"
    tweet_w2v.save(w2vmodel_path)

# save the tfidf
def save_tfidf(tfidf):

    tfidf_path = "../TrainModel/Twitter/word2vec/tfidfdict.txt"

    with open(tfidf_path, "wb") as tfidf_file:
        pickle.dump(tfidf, tfidf_file)

# saving the train model
def save_train_model(model):

    model_path = "../TrainModel/Twitter/word2vec/model.json"
    model_weight = "../TrainModel/Twitter/word2vec/wmodel.h5"

    # serialize model to JSON
    model_json = model.to_json()
    with open(model_path, "w") as json_file:
        json_file.write(model_json)

    # serialize weights to HDF5
    model.save_weights(model_weight)

# loading save model
def load_prediction_model_parameters():

    path = "../TrainModel/Twitter/word2vec/"
    model_path = path + "model.json"
    model_weight_path = path + "wmodel.h5"
    w2vmodel_path = path + "w2vmodel"
    tfidf_path = path + "tfidfdict.txt"

    model = model_from_json(open(model_path).read())
    model.load_weights(model_weight_path)
    w2vmodel = gensim.models.Word2Vec.load(w2vmodel_path)

    with open(tfidf_path, "rb") as tfidf_file:
        tfidf = pickle.load(tfidf_file)

    return model, w2vmodel, tfidf

def process_start():

    n_dim = 200

    # load data set
    tweets_train, tweets_test = load.load_data_v2()

    print(tweets_train.shape)
    # clean and tokanize data set
    twitter_data = pre_process(tweets_train)

    # splitting for training and testing
    X_train, X_test, y_train, y_test = prepare_training_data(twitter_data)

    # labeled dataset data
    X_train = labelize_data(X_train, 'train')
    X_test = labelize_data(X_test, 'test')

    data_labellised= labelize_data(np.array(twitter_data.tokens), 'data')

    # builidng word2vec vocabulary and training
    tweet_w2v = builidng_word2vec_training(data_labellised, n_dim)

    # save word vector data
    save_w2vmodel(tweet_w2v)

    # F-IDF matrix of data
    tfidf = tfidf_matrix(data_labellised)

    # save the tfidf
    save_tfidf(tfidf)

    # Build tweet vector to give input to FFNN
    train_vecs_w2v = np.concatenate([build_word_vector(z, n_dim, tweet_w2v, tfidf) for z in tqdm(map(lambda x: x.words, X_train))])
    train_vecs_w2v = scale(train_vecs_w2v)

    test_vecs_w2v = np.concatenate([build_word_vector(z, n_dim, tweet_w2v, tfidf) for z in tqdm(map(lambda x: x.words, X_test))])
    test_vecs_w2v = scale(test_vecs_w2v)

    # training model
    model_history,model = Training(train_vecs_w2v, y_train)

    # Evaluating accuracy score
    print("------------------------------------------")
    score = model.evaluate(test_vecs_w2v, y_test, batch_size=128, verbose=2)
    print(model.metrics_names[0],": ",score[0],"\n",model.metrics_names[1],": ",score[1])
    print("------------------------------------------")

    # saving the train model
    print("Save train model")
    save_train_model(model)

    print("Model trained and save to a disk")

    return model_history,model

# predict sentiment
def predict(model, w2vmodel, tfidf, tweet):

    n_dim = 200
    tokens = tokenize(tweet)
    tweet_vecs_w2v = build_word_vector(tokens, n_dim, w2vmodel, tfidf)
    # build_word_vector(tokens, size, tweet_w2v, tfidf)
    return model.predict_classes(tweet_vecs_w2v).item()

# plot
def plot(model_history):

    hist = pd.DataFrame(model_history.history)
    hist['epoch'] = model_history.epoch

    # Model Accuracy Plot
    fig = make_subplots(rows=1, cols=1, start_cell="top-left")
    fig.add_trace(go.Scatter(x=hist["epoch"], y=hist["acc"], name="Training accuracy"))
    fig.add_trace(go.Scatter(x=hist["epoch"], y=hist["val_acc"], name="Validation accuracy"))
    fig.update_layout(title_text="Training and validation accuracy",
                      xaxis_title='Epochs',
                      yaxis_title='Accuracy',
                      legend=dict(x=.3, y=1.1),legend_orientation="h")
    fig.show()

    # Model Loss Plot
    fig = make_subplots(rows=1, cols=1, start_cell="top-left")
    fig.add_trace(go.Scatter(x=hist["epoch"], y=hist["loss"], name="Training loss"))
    fig.add_trace(go.Scatter(x=hist["epoch"], y=hist["val_loss"], name="Validation accuracy"))
    fig.update_layout(title_text="Training and validation model loss",
                      xaxis_title='Epochs',
                      yaxis_title='Loss',
                      legend=dict(x=.3, y=1.1),legend_orientation="h")
    fig.show()


if __name__ == "__main__":

    # process model
    model_history,model = process_start()

    # plot
    plot(model_history)

    # test train model
    # model, w2vmodel, tfidf = load_prediction_model_parameters()
    #
    # tweet = "thanks for #lyft credit i can't use because they don't offer wheelchair vans in pdx.  #disapointed #getthanked"
    #
    # prediction = predict(model, w2vmodel, tfidf, tweet)
    # print(prediction)
