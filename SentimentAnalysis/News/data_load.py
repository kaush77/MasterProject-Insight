import sys
import os

# Data Analysis
import pandas as pd

# database
sys.path.append(os.path.abspath('../../'))
import Database.database_log as database_log


def clean_dataset(dataset):

    try:

        dataset.columns=["label","ItemID","Date","Blank","SentimentSource","sub_header"]
        dataset.drop(['ItemID','Date','Blank','SentimentSource'], axis=1, inplace=True)
        dataset = dataset[dataset.label.isnull() == False]
        dataset['label'] = dataset['label'].map( {4:1, 0:0}) #Converting 4 to 1
        dataset = dataset[dataset['sub_header'].isnull() == False]
        dataset = dataset[dataset['label'].isnull() == False]
        dataset.reset_index(inplace=True)
        dataset.drop('index', axis=1, inplace=True)
            # print ('dataset loaded with shape', dataset.shape  )

        return dataset

    except Exception as error:
        # database_log.error_log("News - data_load : clean_dataset", error)
        pass

# load training data
def load_data_v1():

    try:

        train_dataset_path = '../TrainingData/News/tweets.csv'
        test_dataset_path = '../TrainingData/News/tweetstest.csv'

        text_train = pd.read_csv(train_dataset_path, encoding='latin-1')
        text_test = pd.read_csv(test_dataset_path, encoding='latin-1')

        text_train = clean_dataset(text_train)
        text_test = clean_dataset(text_test)

        # merge and filter 50000 text
        df_text_train_pos = text_train.loc[text_train['label'] == 1].head(5000)
        df_text_train_neg = text_train.loc[text_train['label'] == 0].head(5000)

        text_train = pd.concat([df_text_train_pos, df_text_train_neg])

        train_news_path = '../TrainingData/News/train_news_dataset.csv'
        news_train = pd.read_csv(train_news_path, encoding='latin-1')
        news_train.drop(['datetime','ticker'], axis=1, inplace=True)

        train_data_v1 = pd.concat([text_train, news_train])

        # return text_train, text_test
        return train_data_v1

    except Exception as error:
        # database_log.error_log("News - data_load : load_data_v1", error)
        pass

def load_data_v2():

    train_news_path = '../TrainingData/News/train_news.csv'
    train_news_data = pd.read_csv(train_news_path, encoding='latin-1')

    train_news = train_news_data[['Polarity','Subjectivity','content','description']]

    train_news = train_news[train_news.content.isnull() == False]
    train_news = train_news[train_news.description.isnull() == False]
    train_news = train_news[train_news.Polarity.isnull() == False]


    train_news.reset_index(inplace=True)
    train_news.drop('index', axis=1, inplace=True)

    train_news.loc[train_news['Polarity'] >= 0.05, 'Polarity'] = 1
    train_news.loc[(train_news['Polarity'] > -0.05) & (train_news['Polarity'] < 0.05),'Polarity'] = 0
    train_news.loc[train_news['Polarity'] <= -0.05, 'Polarity'] = -1

    train_news.Polarity = train_news.Polarity.astype('int64')

    train_news = train_news.rename(columns = {"Polarity": "label",
                                    "content":"header",
                                    "description": "sub_header"})


    return train_news


def load_data_v3():

    train_news_path = '../TrainingData/News/train_news_v1.csv'
    train_news_data = pd.read_csv(train_news_path, encoding='latin-1')

    return train_news_data


if __name__ == "__main__":

     # text_train, text_test = load_data_v1()

     train_news = load_data_v1()
     print(" train -{} ".format(train_news.shape))

     print(train_news.head(5))
