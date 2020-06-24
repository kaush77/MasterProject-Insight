import re
import sys
import os
import datetime
import pandas as pd
import pandas_datareader.data as web
import pytz

sys.path.append(os.path.abspath('../../'))
sys.path.append(os.path.abspath('../../Database'))
import Database.exchange_index as sql_execute
import Database.database_log as database_log


def current_day(tZone):
    tz = pytz.timezone(tZone)
    current_datetime = datetime.datetime.now(tz)
    return current_datetime

def fetch_stock_data():

    # read indices details
    indices_list = sql_execute.read_daily_indices()
    indices_df = pd.DataFrame(indices_list, columns=['id','yahoo_symbol','time_zone','last_date'])

    indices_data_list = []
    for row in indices_df.itertuples(index=False):

        yahoo_symbol = row.yahoo_symbol
        start_date = row.last_date
        start_date =  datetime.datetime(start_date.year,start_date.month,start_date.day).replace(tzinfo=None)
        tZone = row.time_zone
        end_date = current_day(tZone).replace(tzinfo=None)

        if start_date.replace(tzinfo=None) <= end_date.replace(tzinfo=None):

            start = datetime.datetime(start_date.year,start_date.month,start_date.day)
            end = datetime.datetime(end_date.year,end_date.month,end_date.day)
            symbol = "^"+yahoo_symbol

            # read data
            try:
                df_index = web.DataReader(symbol, 'yahoo', start, end)
                df_index["id"] = int(row.id)

                df_index = df_index.reset_index()
                if df_index.empty is not True:
                    index_list = [tuple(r) for r in df_index[['id','High','Low','Open','Close','Adj Close','Date']].values.tolist()]

                    # insert data into database table
                    sql_execute.bulk_insert_indices_data(index_list)
            except:
                database_log.error_log("India - Loader - india_reuters : read_information", "no record found")


if __name__ == "__main__":

    fetch_stock_data()
    # print(indices_data_list[0])
    # print(df_indices_data)
    # print(df_indices_data.tail())
