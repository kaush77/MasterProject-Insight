import sys
import os
sys.path.append(os.path.abspath('../../'))
sys.path.append(os.path.abspath('../../Schedule'))
sys.path.append(os.path.abspath('../../Database'))
import schedule
from datetime import datetime, timedelta
import calendar
import time
import calendar_schedule as cs
import Database.database_log as database_log
import twitter as sql_execute
import database as sql_database_execute
import pandas as pd
import tweepy
import json
from langdetect import detect
import pre_processing

sys.path.append(os.path.abspath('../../'))
import app_config

CONSUMER_KEY = "9NfQodgml2Io3uUFslehvzZXd"
CONSUMER_SECRET = "cF1dN1SNK4X4VShvzBPQSb872opJeHLI2oQ8W8fGaBrHZ5KKsb"
ACCESS_TOKEN = "1143151270913552386-nTV6DXH8ri21Kdzbmjqbv167RfMS1V"
ACCESS_TOKEN_SECRET = "l2DraSWLyqr2YNq4aK2dbwmVEmP3SLC205pTodNVDU5SA"

# Authenticate to Twitter
auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)


def read_twitter_account():
    twitter_account_list = sql_execute.read_twitter_account()
    df_twitter_account = pd.DataFrame(twitter_account_list, columns=['id', 'screen_id', 'tweet_id'])
    return df_twitter_account


def init():

    # Create API object
    api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

    try:

        api.verify_credentials()
        print("Authentication OK")

    except Exception as error:
        database_log.error_log("twitter_scheduler - init - Error creating API", error)
        # auth_status = False

    return api


def read_user_timeline(api):

    try:

        # get twitter account
        df_twitter_account = read_twitter_account()

        for row in df_twitter_account.itertuples(index=False):
            # print(row["id"], row["screen_id"], row['tweet_id'])

            tweets = api.user_timeline(screen_name=row.screen_id, include_rts=False, since_id=int(row.tweet_id))

            tweet_list = []
            for tweet in tweets:
                tweet_id = tweet.id
                tweet_message = tweet.text
                tweet_source = tweet.source
                retweet_count = tweet.retweet_count
                likes_count = tweet.favorite_count
                tweet_date = tweet.created_at

                try:
                    text_lang = detect(tweet_message)

                    # check if text is in english
                    if text_lang == "en":
                        tweet_list.append({'tweet_id': tweet_id, 'screen_id':row.screen_id, 'tweet_message': tweet_message,
                                           'tweet_source': tweet_source, 'retweet_count': retweet_count,
                                           'likes_count': likes_count, 'tweet_date': tweet_date})

                except Exception as error:
                    database_log.error_log("twitter_scheduler - read_user_timeline - language error", error)

                    tweet_list.append({'tweet_id': tweet_id, 'screen_id':row.screen_id, 'tweet_message': tweet_message,
                                       'tweet_source': tweet_source, 'retweet_count': retweet_count,
                                       'likes_count': likes_count, 'tweet_date': tweet_date})

            if len(tweet_list) > 0:
                tweet_data_frame = pd.DataFrame(tweet_list, columns=['tweet_id', 'screen_id', 'tweet_message',
                                                                 'tweet_source', 'retweet_count', 'likes_count',
                                                                 'tweet_date'])

                clean_twitter_data_frame = pre_processing.clean_twitter_data(tweet_data_frame)

            if len(tweet_list) > 0 and clean_twitter_data_frame.empty is not True:
                tweet_data_list = [tuple(r) for r in clean_twitter_data_frame[['tweet_id', 'screen_id', 'tweet_message',
                                                                       'tweet_source', 'retweet_count', 'likes_count',
                                                                       'tweet_date']].values.tolist()]
                sql_execute.bulk_insert_twitter_feeds(tweet_data_list)
            else:
                database_log.error_log("twitter_scheduler - read_user_timeline", "no record found")
                print("No Record Found.")

    except Exception as error:
        database_log.error_log("twitter_scheduler - read_user_timeline", error)
        print("twitter_scheduler - read_user_timeline - {}".format(error))


if __name__ == '__main__':

    try:

        lookup_value = "twitter"  # search website name
        df_config = sql_database_execute.read_configuration(lookup_value)  # read config file setting

        row_id = df_config.index[0]
        scheduled_task_sleeping = df_config["scheduled_task_sleeping"][row_id]

        # initialize twitter api
        api = init()

        while True:
            scheduled_task_sleeping = app_config.twitter_data_scheduled_task_sleeping
            read_user_timeline(api)
            print("Last run was successful for Twitter API, next run in {} seconds.".format(scheduled_task_sleeping))
            time.sleep(scheduled_task_sleeping)

    except Exception as error:
        database_log.error_log("twitter_scheduler - main", error)
