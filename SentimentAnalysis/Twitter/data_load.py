import sys
import os

# Data Analysis
import pandas as pd

# database
# sys.path.append(os.path.abspath('../../Database'))
sys.path.append(os.path.abspath('../../'))
import Database.database_log as database_log


# load training data
def load_data():

    try:

        train_dataset_path = '../TrainingData/TwitterData/tweets_train.csv'
        test_dataset_path = '../TrainingData/TwitterData/tweets_test.csv'

        tweets_train = pd.read_csv(train_dataset_path)
        tweets_test = pd.read_csv(test_dataset_path)

        return tweets_train, tweets_test

    except Exception as error:
        database_log.error_log("data_load : load_data", error)

def load_data_v1():

    try:

        train_dataset_path = '../TrainingData/TwitterData/Sentiment_Analysis_Dataset.csv'
        tweets_train = pd.read_csv(train_dataset_path)

        return tweets_train

    except Exception as error:
        database_log.error_log("data_load : load_data_v1", error)

def clean_dataset_v2(dataset):

    dataset.columns=["label","ItemID","Date","Blank","SentimentSource","tweet"]
    dataset.drop(['ItemID','Date','Blank','SentimentSource'], axis=1, inplace=True)
    dataset = dataset[dataset.label.isnull() == False]
    dataset['label'] = dataset['label'].map( {4:1, 0:0}) #Converting 4 to 1
    dataset = dataset[dataset['tweet'].isnull() == False]
    dataset = dataset[dataset['label'].isnull() == False] # remove NaN row
    dataset.reset_index(inplace=True)
    dataset.drop('index', axis=1, inplace=True)
    # print ('dataset loaded with shape', dataset.shape  )

    return dataset

# load training data
def load_data_v2():

    # try:

    train_dataset_path = '../TrainingData/TwitterData/tweets.csv'
    test_dataset_path = '../TrainingData/TwitterData/tweetstest.csv'

    tweets_train = pd.read_csv(train_dataset_path, encoding='latin-1')
    tweets_test = pd.read_csv(test_dataset_path, encoding='latin-1')

    tweets_train = clean_dataset_v2(tweets_train)
    tweets_test = clean_dataset_v2(tweets_test)

    print(tweets_train.shape)
    return tweets_train, tweets_test

    # except Exception as error:
    # database_log.error_log("data_load : load_data_v2", error)

def clean_dataset_v3(dataset):

    try:

        dataset.columns=["label","ItemID","Date","Blank","SentimentSource","tweet"]
        dataset.drop(['ItemID','Date','Blank','SentimentSource'], axis=1, inplace=True)
        dataset = dataset[dataset.label.isnull() == False]
        dataset['label'] = dataset['label'].map( {4:1, 0:0}) #Converting 4 to 1
        dataset = dataset[dataset['tweet'].isnull() == False]
        dataset = dataset[dataset['label'].isnull() == False] # remove NaN row
        dataset.reset_index(inplace=True)
        dataset.drop('index', axis=1, inplace=True)
        # print ('dataset loaded with shape', dataset.shape  )

        return dataset

    except Exception as error:
        database_log.error_log("data_load : clean_dataset_v3", error)

# load training data
def load_data_v3():

    try:

        train_dataset_path = '../TrainingData/TwitterData/tweets.csv'
        test_dataset_path = '../TrainingData/TwitterData/tweetstest.csv'

        tweets_train = pd.read_csv(train_dataset_path, encoding='latin-1')
        tweets_test = pd.read_csv(test_dataset_path, encoding='latin-1')

        tweets_train = clean_dataset_v3(tweets_train)
        tweets_test = clean_dataset_v3(tweets_test)

        return tweets_train, tweets_test

    except Exception as error:
        database_log.error_log("data_load : load_data_v3", error)

# load training data
def load_data_v4():

    try:

        train_dataset_path = '../TrainingData/TwitterData/tweets.csv'
        test_dataset_path = '../TrainingData/TwitterData/tweetstest.csv'

        tweets_train = pd.read_csv(train_dataset_path, encoding='latin-1')
        tweets_test = pd.read_csv(test_dataset_path, encoding='latin-1')

        tweets_train = clean_dataset_v3(tweets_train)
        tweets_test = clean_dataset_v3(tweets_test)

        # merge and filter 50000 tweets
        df_tweets_train_pos = tweets_train.loc[tweets_train['label'] == 1].head(30000)
        df_tweets_train_neg = tweets_train.loc[tweets_train['label'] == 0].head(30000)

        # print(" train -{} ; {}".format(df_tweets_train_pos.shape,df_tweets_train_neg.shape))

        tweets_train = pd.concat([df_tweets_train_pos, df_tweets_train_neg])

        return tweets_train, tweets_test

    except Exception as error:
        database_log.error_log("data_load : load_data_v3", error)

def load_data_v5():

    try:

        twitter_training_data_path = '../TrainingData/TwitterData/twitter_training_data.csv'
        twitter_training_df = pd.read_csv(twitter_training_data_path)

        tweets_train, tweets_test = load_data_v4()
        train_data_v1 = pd.concat([twitter_training_df, tweets_train])
        # return twitter_training_df

        return tweets_train

    except Exception as error:
        database_log.error_log("data_load : load_data_v5", error)


def load_data_v6():

    try:

        tweets_train, tweets_test = load_data_v4()

        return tweets_train

    except Exception as error:
        database_log.error_log("data_load : load_data_v6", error)

if __name__ == "__main__":

    # twitter_training_df = load_data_v5()
    # print(twitter_training_df.shape)
    # load_data_v1()
    tweets_train, tweets_test = load_data_v2()
    print(tweets_train.shape)
    # tweets_train, tweets_test = load_data_v5()
    # tweets_train, tweets_test = load_data()
    #

    # df_tweets_train_pos = tweets_train.loc[tweets_train['label'] == 1]
    # df_tweets_train_neg = tweets_train.loc[tweets_train['label'] == 0]
    #
    # print(df_tweets_train_pos.shape)
    # print(tweets_test.shape)
    #
    # print(df_tweets_train_pos.head(2))
    # print(tweets_test.head(2))
