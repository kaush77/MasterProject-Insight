import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from datetime import datetime
import re
import sys
import os
from langdetect import detect
sys.path.append(os.path.abspath('../../'))
import Database.database as sql_execute
import Database.database_log as database_log

pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 10)

# Read information
def read_information(df_web_config):

    try:

        # Creating an empty data frame with column names only
        df_news_data = pd.DataFrame(columns=['website', 'website_category', 'website_link', 'sub_header',
                                             'timestamp'])

        for row in df_web_config.itertuples(index=False):
            url_link = row.website_link

            # open with GET method
            resp = requests.get(url_link, headers={'User-Agent': 'Mozilla/5.0'})

            # http_response 200 means OK status
            if resp.status_code == 200:
                # parser
                soup = BeautifulSoup(resp.text, 'html.parser')

                website = 'business_standard'
                categories = row.website_category
                news_link = ''
                header = ''
                sub_header = ''
                timestamp = ''

                for level_1 in soup.findAll("div", {"class": "aticle-list"}):
                    for level_2 in level_1.findAll("li"):
                        for level_3 in level_2.findAll("a"):
                            news_link = level_3['href']
                            header = level_3.text.strip()

                        for level_3 in level_2.findAll("p"):
                            sub_header = level_3.text.strip()

                        if len(header) > 0:
                            text_lang = detect(header)

                            if text_lang == "en":
                                df_news_data = df_news_data.append({'website': website, 'website_link': url_link,
                                                    'website_category': categories,'news_link': news_link, 'header': header,
                                                    'sub_header': sub_header, 'timestamp': timestamp},ignore_index=True)

                sub_header = ''
                timestamp = ''

                for level_1 in soup.findAll("div", {"class": "coutent-panel bs-new-top-story-listing-block"}):
                    for level_2 in level_1.findAll("div"):
                        for level_3 in level_2.findAll("a"):
                            news_link = level_3['href']
                            header = level_3.text.strip()

                        if len(header) > 0:
                            text_lang = detect(header)

                        if text_lang == "en":
                            df_news_data = df_news_data.append({'website': website, 'website_link': url_link,
                                                'website_category': categories, 'news_link': news_link, 'header': header,
                                                 'sub_header': sub_header, 'timestamp': timestamp}, ignore_index=True)

            else:
                database_log.error_log("India - Loader - business_standard : read_information", resp.status_code)

        if df_news_data.empty is not True:
            df_news_feed_list = [tuple(r) for r in df_news_data[['website', 'website_link', 'website_category',
                                                                 'news_link', 'header', 'sub_header',
                                                                 'timestamp']].values.tolist()]
            sql_execute.bulk_insert_news_feeds(df_news_feed_list)
        else:
            database_log.error_log("India - Loader - business_standard : read_information", "no record found")

    except Exception as error:
        database_log.error_log("India - Loader - business_standard : read_information", error)


# method to start loading files
def start_load_process(lookup_value):
    df_web_config = sql_execute.read_source_link(lookup_value)
    read_information(df_web_config)


if __name__ == '__main__':
    start_load_process('business_standard')
