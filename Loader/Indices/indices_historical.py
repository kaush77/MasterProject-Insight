import re
import sys
import os
import datetime
import pandas as pd
import pandas_datareader.data as web

sys.path.append(os.path.abspath('../../Database'))
import exchange_index as sql_execute


def fetch_stock_data():

    # read indices details
    indices_list = sql_execute.read_indices()
    indices_df = pd.DataFrame(indices_list, columns=['id', 'index_symbol', 'yahoo_symbol','start_date'])

    indices_data_list = []

    for row in indices_df.itertuples(index=False):

        yahoo_symbol = row.yahoo_symbol
        start_date = row.start_date
        end_date = datetime.date.today()

        start = datetime.datetime(start_date.year,start_date.month,start_date.day)
        end = datetime.datetime(end_date.year,end_date.month,end_date.day)
        symbol = "^"+yahoo_symbol

        # read data
        df_index = web.DataReader(symbol, 'yahoo', start, end)
        df_index["id"] = int(row.id)

        df_index = df_index.reset_index()

        if df_index.empty is not True:
            index_list = [tuple(r) for r in df_index[['id','High','Low','Open','Close','Adj Close','Date']].values.tolist()]

            # insert data into database table
            sql_execute.bulk_insert_indices_data(index_list)


if __name__ == "__main__":

    fetch_stock_data()
    # print(indices_data_list[0])
    # print(df_indices_data)
    # print(df_indices_data.tail())
