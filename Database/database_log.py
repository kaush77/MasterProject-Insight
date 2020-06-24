import sys
import os
from Database.connection import DatabaseConnection,CursorFromConnectionFromPool
import pandas as pd
from datetime import datetime

# initialise database connection pool
DatabaseConnection.initialise()


def process_log(source, message):

    try:

        sql_insert_query = """ INSERT INTO process_log (source,message) VALUES (%s,%s) """
        record = (source, message)

        with CursorFromConnectionFromPool() as cursor:
            cursor.execute(sql_insert_query, record)

    except Exception as error:
        print(error)


def error_log(source, message):

    try:

        sql_insert_query = """ INSERT INTO error_log (source,message) VALUES (%s,%s) """
        record = (source, str(message))

        with CursorFromConnectionFromPool() as cursor:
            cursor.execute(sql_insert_query, record)

    except Exception as error:
        pass
        # file_log.log_error('process_log', error)
