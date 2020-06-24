from Database.connection import DatabaseConnection, CursorFromConnectionFromPool
import pandas as pd
import Database.database_log as database_log

def read_indices():

    try:

        query = """ SELECT id, index_symbol, yahoo_symbol, start_date FROM indices WHERE is_active = true
                    ORDER BY id """

        with CursorFromConnectionFromPool() as cursor:
            cursor.execute(query)

            news_data = cursor.fetchall()
            return news_data

    except Exception as error:
        database_log.error_log("read_indices", error)


# daily data
def read_daily_indices():

    try:

        query = """ SELECT indices.id, indices.yahoo_symbol,indices.time_zone,CASE WHEN MAX(indices_data.entry_date) is null
                    THEN indices.start_date ELSE MAX(indices_data.entry_date) + INTERVAL '1 day' end AS last_date
                    FROM indices LEFT JOIN indices_data ON indices.id = indices_data.index_id
                    WHERE indices.is_active = true GROUP BY indices.id ORDER BY id """

        with CursorFromConnectionFromPool() as cursor:
            cursor.execute(query)

            news_data = cursor.fetchall()
            return news_data

    except Exception as error:
        database_log.error_log("read_daily_indices", error)


# historical data
def bulk_insert_indices_data(records):

    try:

        sql_insert_query = """ INSERT INTO indices_data (index_id,high,low,open,close,adj_close,entry_date)
                                                VALUES (%s,%s,%s,%s,%s,%s,%s) """

        with CursorFromConnectionFromPool() as cursor:
            cursor.executemany(sql_insert_query, records)

    except Exception as error:
        database_log.error_log("bulk_insert_twitter_feeds", error)
