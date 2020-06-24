from Database.connection import DatabaseConnection, CursorFromConnectionFromPool
import pandas as pd
import Database.database_log as database_log

# initialise database connection pool
DatabaseConnection.initialise()

def read_website_configuration(lookup_value):

    try:

        with CursorFromConnectionFromPool() as cursor:
            cursor.execute("SELECT website,website_category,website_link FROM website_configuration "
                           "WHERE is_active=true and website = %s", (lookup_value,))
            web_config_list = cursor.fetchall()
            return web_config_list

    except Exception as error:
        database_log.error_log("read_website_configuration", error)


def read_source_link(lookup_value):

    try:

        web_config_list = read_website_configuration(lookup_value)
        df_web_config = pd.DataFrame(web_config_list, columns=['website', 'website_category', 'website_link'])
        return df_web_config

    except Exception as error:
        database_log.error_log("read_source_link", error)


def read_configuration(lookup_value):

    try:

        with CursorFromConnectionFromPool() as cursor:
            cursor.execute("SELECT website,scheduled_task_sleeping,market_time_zone,market_start_time,market_end_time,"
                           "market_hours_delay,market_off_hours_delay,market_weekend_hours_delay,market_location "
                           "FROM configuration WHERE is_active=true and website = %s", (lookup_value,))
            config_list = cursor.fetchall()

            df_web_config = pd.DataFrame(config_list,
                                         columns=['website', 'scheduled_task_sleeping', 'market_time_zone',
                                                  'market_start_time', 'market_end_time', 'market_hours_delay',
                                                  'market_off_hours_delay', 'market_weekend_hours_delay',
                                                  'market_location'])
            return df_web_config

    except Exception as error:
        database_log.error_log("read_source_link", error)


def bulk_insert_news_feeds(records):

    # try:

    sql_insert_query = """ INSERT INTO news_feeds_dump (website,website_link,website_category,news_link,
                                header,sub_header,timestamp) VALUES (%s,%s,%s,%s,%s,%s,%s) """

        # print(records)
    with CursorFromConnectionFromPool() as cursor:
        cursor.executemany(sql_insert_query, records)

    # except Exception as error:
    #     database_log.error_log("read_source_link", error)
