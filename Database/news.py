from Database.connection import DatabaseConnection, CursorFromConnectionFromPool
import pandas as pd
import Database.database_log as database_log
import Database.connection as connection

def read_news_data():

    try:

        query = """ CREATE TEMPORARY TABLE temp_news_headlines
                    (
                          id INT,
                          header TEXT NULL,
                    	  sub_header TEXT NULL
                    );

                    INSERT INTO temp_news_headlines(id,header,sub_header)
                    SELECT ndump.id, ndump.header, ndump.sub_header FROM news_feeds_dump ndump
                    WHERE ndump.id NOT IN (SELECT DISTINCT news_id FROM news_feeds_sentiment)
                    AND (ndump.sub_header IS NOT NULL AND ndump.sub_header !='')
                    ORDER BY id;

                    delete from temp_news_headlines temp1 using temp_news_headlines temp2
					where temp1.id<temp2.id and temp1.sub_header=temp2.sub_header;

					delete from temp_news_headlines temp1 using temp_news_headlines temp2
					where temp1.id<temp2.id and temp1.header=temp2.header;

                    SELECT ndump.id, ndump.header, ndump.sub_header FROM temp_news_headlines ndump
                    WHERE ndump.header NOT IN (SELECT DISTINCT header FROM news_feeds_dump nfdump join news_feeds_sentiment nsentiment on nfdump.id=nsentiment.news_id)
                    ORDER BY id; """

        with CursorFromConnectionFromPool() as cursor:
            cursor.execute(query)

            news_data = cursor.fetchall()
            return news_data

    except Exception as error:
        database_log.error_log("read_news_data", error)

def bulk_insert_news_sentiment(news_data_sentiment_df, flag):

    # try:

    # convert data frame into a list
    news_data_sentiment_list = [tuple(r) for r in news_data_sentiment_df[['id','sentiment_for','nltk_classify','nltk_confidence',
                            'count_vectorizer_classify','count_vectorizer_confidence','tfidf_vectorizer_classify',
                            'tfidf_vectorizer_confidence']].values.tolist()]

    sql_news_sentiment_query = """ INSERT INTO news_feeds_sentiment (news_id,sentiment_for,nltk_classify,nltk_confidence,
                                            spacy_count_vectorizer_classify,spacy_count_vectorizer_confidence,
                                            spacy_tfidf_vectorizer_classify,spacy_tfidf_vectorizer_confidence)
                                            VALUES (%s,%s,%s,%s,%s,%s,%s,%s) """

    with CursorFromConnectionFromPool() as cursor:
        cursor.executemany(sql_news_sentiment_query, news_data_sentiment_list)

    # delete processd records
    # if flag:
    #     news_data_list = [tuple(r) for r in news_data_sentiment_df[['id', 'header', 'sub_header']].values.tolist()]
    #
    #     sql_news_data_query = """ INSERT INTO news_feeds (id,header,sub_header)
    #                                             VALUES (%s,%s,%s) """
    #
    #
    #     with CursorFromConnectionFromPool() as cursor:
    #        cursor.executemany(sql_news_data_query, news_data_list)
    #
    #     max_news_id = news_data_sentiment_df['id'].max()
    #     max_news_id = int(max_news_id)
    #     sql_delete_query = "DELETE FROM news_preprocessing_feed WHERE id <= %s"
    #
    #     with CursorFromConnectionFromPool() as cursor:
    #         cursor.execute(sql_delete_query, (max_news_id,))

    # except Exception as error:
    #     database_log.error_log("bulk_insert_news_sentiment", error)
