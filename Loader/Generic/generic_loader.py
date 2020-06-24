from urllib.request import urlopen
from bs4 import BeautifulSoup
from bs4.element import Tag as bs_tag
from bs4.element import NavigableString
from bs4.element import ResultSet
import re
import sys
import os
import json
import pandas as pd
import time

sys.path.append(os.path.abspath('../../'))
sys.path.append(os.path.abspath('../../Database'))
import Database.database as sql_execute

import app_config

# Global variables
json_file_path = "Json_Files/"
data_path = "Data/"

def read_data(url,file_name):

    html = urlopen(url)
    soup = BeautifulSoup(html, 'lxml')
    soup_body = soup.body
    print(url)

    column_names = ["website","web_url","article_url","header","sub_header","published_date"]
    news_df = pd.DataFrame()
    news_data_df = pd.DataFrame(columns=column_names)

        # read json file
    json_file = json_file_path + file_name +".json"
    json_data = json.loads(open(json_file).read())

    # collect webiste data
    news_df_temp = extract_values(json_data, soup_body, url,news_df)

    news_df["web_url"] = url
    news_df["website"] = file_name

        # add column to temp data frame in case of missing columns
    news_temp_df = dataframe_columns_validation(news_df,news_data_df)

        # append data to main dataframe
    news_data_df = news_data_df.append(news_temp_df, ignore_index=True)

    news_data_df = news_data_df.drop_duplicates(subset=["website","web_url","article_url","header","sub_header","published_date"],
                                                    keep='first')

    print(news_data_df.shape)

    # save data to a file
    news_data_df = news_data_df.replace('\n',' ', regex=True)
    file_path = data_path + file_name + ".csv"
    news_data_df.to_csv(file_path)

    # removing records from tables
    news_df.drop(news_df.index, inplace=True)

def dataframe_columns_validation(news_df, news_data_df):
    for col in news_df.columns:
        if col in news_data_df.columns:
            pass
        else:
            news_df[col] = ""

    return news_df


def extract_values(json_data, soup, url,news_df):
    news_temp_tf = pd.DataFrame()

    def extract(json_data):
        global news_temp_tf
        if isinstance(json_data, dict):
            for key,value in json_data.items():
                if isinstance(value,(dict,list)):
                    extract(value)
                else:
                    web_url = value

        elif isinstance(json_data,(dict,list)):
            if (json_data,(dict,list)):
                news_temp_tf = scrap_data(json_data, soup, url,news_df)

        return news_temp_tf

    news_data_df = extract(json_data)
    return news_data_df

def scrap_data(obj,soup,url,news_df):
    bs4_object = soup
    news_temp_df = pd.DataFrame()
    for item in obj:
        if isinstance(item,dict):
            if(item["data"] == "false"):
                bs4_object = read_tag_data(item,bs4_object,url,news_df)
            else:
                news_temp_df = read_tag_data(item,bs4_object,url,news_df)

    return news_temp_df

def read_tag_data(obj,bs4_object, url,news_df):
    record = []
    record_link = []
    news_temp_df = pd.DataFrame()
    if(obj["data"] == "false"):
        bs4_tag_list = []

        if obj["lookup_by"] == "class":

            if obj["class"] !="":
                for tag_list in bs4_object:
                    tag_instance = isinstance(tag_list,bs_tag)
                    if tag_instance:
                        for all_links_link in tag_list.find_all(obj["tag"],class_=re.compile("^"+obj["class"])):
                            bs4_tag_list.append(all_links_link)

                return bs4_tag_list

            else:
                for tag_list in bs4_object:
                    tag_instance = isinstance(tag_list,bs_tag)
                    if tag_instance:
                        for all_links_link in tag_list.find_all(obj["tag"]):
                            bs4_tag_list.append(all_links_link)

                return bs4_tag_list

        elif obj["lookup_by"] == "id":
            for tag_list in bs4_object:
                tag_instance = isinstance(tag_list,bs_tag)
                if tag_instance:
                    for all_links_link in tag_list.find_all(id=obj["id"]):
                        bs4_tag_list.append(all_links_link)

            # only get the first element in the list
            bs4_tag_first_item = bs4_tag_list[:1]

            return bs4_tag_first_item

    elif(obj["data"] == "true" and obj["tag_type"] == "anchor"):
        print(type(news_df))
        for tag_list in bs4_object:

            tag_instance = isinstance(tag_list,bs_tag)
            if tag_instance:
                record_temp,record_link_temp = get_anchor_tag_data(tag_list, obj)
                if len(record_temp) > 0:
                    record.extend(record_temp)
                    record_link.extend(record_link_temp)

        news_temp_df = add_column_to_dataframe(record,record_link,obj["data_field"],news_df)

        return news_temp_df

    elif(obj["data"] == "true" and obj["tag_type"] == "div"):
        for tag_list in bs4_object:
            tag_instance = isinstance(tag_list,bs_tag)
            if tag_instance:
                record_temp = get_div_tag_data(tag_list, obj)
                record.extend(record_temp)

        news_df = add_column_to_dataframe(record,record_link,obj["data_field"],news_df)

        return news_df

    elif(obj["data"] == "true" and (obj["tag_type"] == "span" or obj["tag_type"] == "p")):
        for tag_list in bs4_object:
            tag_instance = isinstance(tag_list,bs_tag)
            if tag_instance:
                for tag_data in tag_list.find_all(obj["tag"],class_=obj["class"]):
                    record.append(tag_data.text)

        news_temp_df = add_column_to_dataframe(record,record_link,obj["data_field"],news_df)

        return news_temp_df

    else:
        for tag_list in bs4_object:
            tag_instance = isinstance(tag_list,bs_tag)
            if tag_instance:
                for tag_data in tag_list.find_all(obj["tag"],class_=obj["class"]):
                    record.append(tag_data.text)

        add_column_to_dataframe(record,record_link,obj["data_field"])

def get_anchor_tag_data(tag_list, obj):

    record = []
    record_link = []

    if obj["type"] == "1":
        for header_data in tag_list.find_all(obj["tag"],class_=obj["class"]):
            if obj["data_field"] == "link":
                record_link.append(header_data.get("href"))
                record.append(header_data.getText())

    else:
        for header_data in tag_list.find_all(obj["tag"],class_=obj["class"]):
            if obj["data_field"] == "link":
                record_link.append(header_data.get("href"))
                record.append(header_data.getText())

    return record,record_link

def get_div_tag_data(tag_list, obj):

    record = []


    if obj["type"] == "1": # if multiple section exists with the same property
        tag_section = tag_list.find_all(obj["tag"],class_=re.compile("^"+obj["class"]))

        if len(tag_section) > 1:
            for tag_data in tag_list.find(obj["tag"],class_=re.compile("^"+obj["class"])):
                record.append(tag_data)
        else:
            for tag_data in tag_list.find_all(obj["tag"],class_=re.compile("^"+obj["class"])):
                record.append(tag_data.text)
    else:
        for tag_data in tag_list.find_all(obj["tag"],class_=re.compile("^"+obj["class"])):
            record.append(tag_data.text)

    return record

def add_column_to_dataframe(record,record_link,field_type,news_df):
    print("{} - {} - {}".format(len(record),len(record_link),field_type))
    if field_type == "link":
        if news_df is not None and "article_url" in news_df.columns:
            lhrow_index = len(news_df)
            larow_index = len(news_df)
            for row in record:
                lhrow_index +=1
                news_df.at[lhrow_index, 'header'] = row
            for row in record_link:
                larow_index +=1
                news_df.at[larow_index, 'article_url'] = row
        else:
            news_df["article_url"] = record_link
            news_df["header"] = record
    elif field_type == "published_date":
        if "published_date" in news_df.columns:
            news_df.reset_index(inplace = True)
            lrow_index = news_df["published_date"].last_valid_index()

            if lrow_index is None:
                lrow_index = len(news_df)
            for row in record:
                lrow_index +=1
                news_df.at[lrow_index, 'published_date'] = row
        else:
            news_df["published_date"] = record

    elif field_type == "header":
         if "header" in news_df.columns:
            news_df.reset_index(inplace = True)
            lrow_index = news_df["header"].last_valid_index()

            if lrow_index is None:
                lrow_index = len(news_df)

            for row in record:
                lrow_index +=1
                news_df.at[lrow_index, 'header'] = row

         else:
            news_df["header"] = record

    elif field_type == "sub_header":
        if "sub_header" in news_df.columns:
            news_df.reset_index(inplace = True)
            lrow_index = news_df["sub_header"].last_valid_index()
            if lrow_index is None:
                lrow_index = len(news_df)
            for row in record:
                lrow_index +=1
                news_df.at[lrow_index, 'sub_header'] = row
        else:
            news_df["sub_header"] = record

    return news_df

# load the scrap information into the database
def load_data_to_tables(url_list):
    for file_name,url in url_list.items():
        df_news_data = pd.read_csv(data_path + file_name +".csv")

        if df_news_data.empty is not True:
            df_news_data["website_category"] = "Finance"
            df_news_feed_list = [tuple(r) for r in df_news_data[['website', 'web_url', 'website_category',
                                'article_url', 'header', 'sub_header','published_date']].values.tolist()]
            sql_execute.bulk_insert_news_feeds(df_news_feed_list)


# scrap website data and load into the database
def load_config_files():
    # url_list = {
    #             "moneycontrol":"https://www.moneycontrol.com/news/business/markets/",
    #             "businessinsider":"https://www.businessinsider.com/moneygame",
    #             "nasdaq":"https://www.nasdaq.com/news-and-insights/topic/markets/world-markets"
    #             }

    url_list = {
                "moneycontrol":"https://www.moneycontrol.com/news/business/markets/",
                "businessinsider":"https://www.businessinsider.com/moneygame"
                }

    for file_name,url in url_list.items():
        print("File Name -{}  URL - {}".format(file_name,url))
        read_data(url,file_name)

    # save scrap news data into a database
    load_data_to_tables(url_list)

if __name__ == "__main__":

    while True:
        scheduled_sleeping_seconds = app_config.generic_market_scheduled_task_sleeping
        load_config_files()
        print("Last run was successful for Genric scraper, next run in {} seconds.".format(scheduled_sleeping_seconds))
        time.sleep(scheduled_sleeping_seconds)

    # url_list = {
    #             "moneycontrol":"https://www.moneycontrol.com/news/business/markets/",
    #             "businessinsider":"https://www.businessinsider.com/moneygame",
    #             "nasdaq":"https://www.nasdaq.com/news-and-insights/topic/markets/world-markets"
    #             }
    #
    # load_data_to_tables(url_list)
