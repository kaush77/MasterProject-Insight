from Database.connection import DatabaseConnection, CursorFromConnectionFromPool
import pandas as pd
import Database.database_log as database_log
import Database.connection as connection

# initialise database connection pool
DatabaseConnection.initialise()

def read_twitter_account():

    try:

        query = """ SELECT ta.id, ta.screen_id, CASE WHEN max(tweet_id) IS NULL THEN '1'
                    ELSE max(tweet_id) END tweet_id FROM twitter_account ta LEFT JOIN
                    twitter_data_dump td ON ta.screen_id = td.screen_id WHERE ta.is_active = true
                    GROUP BY ta.id ORDER BY ta.id """

        with CursorFromConnectionFromPool() as cursor:
            cursor.execute(query)
            twitter_account_list = cursor.fetchall()
            return twitter_account_list

    except Exception as error:
        database_log.error_log("read_twitter_account", error)


def bulk_insert_twitter_feeds(records):

    # try:

    sql_insert_query = """ INSERT INTO twitter_data_dump (tweet_id,screen_id,tweet_message,tweet_source,
                               retweet_count,likes_count,tweet_date) VALUES (%s,%s,%s,%s,%s,%s,%s) """

        # print(records)
    with CursorFromConnectionFromPool() as cursor:
        cursor.executemany(sql_insert_query, records)

    # except Exception as error:
    #     database_log.error_log("bulk_insert_twitter_feeds", error)


# sentiment processing

def read_twitter_data():

    try:

        query = """ SELECT tdump.id, tdump.tweet_id, tdump.screen_id, tdump.tweet_message, tdump.tweet_date FROM twitter_data_dump tdump
                    WHERE tdump.id NOT IN (select DISTINCT tweet_id from twitter_sentiment)
                    AND tweet_message!='NaN' AND tweet_message!='' """

        with CursorFromConnectionFromPool() as cursor:
            cursor.execute(query)

            twitter_data = cursor.fetchall()
            return twitter_data

    except Exception as error:
        database_log.error_log("read_twitter_data", error)


def bulk_insert_twitter_sentiment(twitter_data_sentiment_df):

    try:

        # convert data frame into a list
        twitter_data_list = [tuple(r) for r in twitter_data_sentiment_df[['id', 'tweet_id', 'screen_id',
                                                                'tweet_message', 'tweet_date']].values.tolist()]

        twitter_data_sentiment_list = [tuple(r) for r in twitter_data_sentiment_df[['id','nltk_classify','nltk_confidence',
                            'count_vectorizer_classify','count_vectorizer_confidence',
                            'tfidf_vectorizer_classify','tfidf_vectorizer_confidence','word2vec_classify']].values.tolist()]

        # sql_twitter_data_query = """ INSERT INTO twitter_data (id,tweet_id,screen_id,tweet_message,tweet_date)
        #                                     VALUES (%s,%s,%s,%s,%s) """

        sql_twitter_sentiment_query = """ INSERT INTO twitter_sentiment (tweet_id,nltk_classify,nltk_confidence,
                                          spacy_count_vectorizer_classify,spacy_count_vectorizer_confidence,
                                          spacy_tfidf_vectorizer_classify,spacy_tfidf_vectorizer_confidence,word2vec_classify)
                                          VALUES (%s,%s,%s,%s,%s,%s,%s,%s) """


        # with CursorFromConnectionFromPool() as cursor:
        #    cursor.executemany(sql_twitter_data_query, twitter_data_list)

        with CursorFromConnectionFromPool() as cursor:
            cursor.executemany(sql_twitter_sentiment_query, twitter_data_sentiment_list)

        # delete processd records
        # max_tweet_id = twitter_data_sentiment_df['id'].max()
        # max_tweet_id = int(max_tweet_id)
        # sql_delete_query = "DELETE FROM twitter_preprocessing_data WHERE id <= %s"
        #
        # with CursorFromConnectionFromPool() as cursor:
        #     cursor.execute(sql_delete_query, (max_tweet_id,))

    except Exception as error:
        database_log.error_log("bulk_insert_twitter_sentiment", error)
