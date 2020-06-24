from psycopg2 import pool
import psycopg2

# Database creds
t_host = "localhost"
t_port = "5432" #default postgres port
t_dbname = "master_project"
t_user = "postgres"
t_pw = "root"

def load_news_data_to_process(data_type):
    ps_connection = psycopg2.connect(user=t_user,
                                  password=t_pw,
                                  host=t_host,
                                  port=t_port,
                                  database=t_dbname)

    cursor = ps_connection.cursor()

    #call stored procedure
    if data_type == 'newsfeed':
        cursor.execute("call process_news_data('newsfeed')")
    elif data_type == 'twitter':
        cursor.execute("call process_twitter_data('twitter')")

    ps_connection.commit()


class DatabaseConnection:

    __connection_pool = None

    @classmethod
    def initialise(cls):
        cls.__connection_pool = \
            pool.SimpleConnectionPool(1, 10, user=t_user, password=t_pw,
                                      database=t_dbname, host=t_host)

    @classmethod
    def get_connection(cls):
        return cls.__connection_pool.getconn()

    @classmethod
    def return_connection(cls, connection):
        cls.__connection_pool.putconn(connection)

    @classmethod
    def close_all_connection(cls):
        cls.__connection_pool.closeall()


class CursorFromConnectionFromPool:
    def __init__(self):
        self.connection = None
        self.cursor = None

    def __enter__(self):
        self.connection = DatabaseConnection.get_connection()
        self.cursor = self.connection.cursor()
        return self.cursor

    def __exit__(self, exception_type, exception_value, exception_traceback):

        if exception_value is not None:
            self.connection.rollback()
        else:
            self.cursor.close()
            self.connection.commit()

        DatabaseConnection.return_connection(self.connection)
