import pandas as pd
import psycopg2
from configparser import ConfigParser


def config(filename='database.ini', section='postgresql'):
    parser = ConfigParser()
    parser.read(filename)

    # get section, default to postgresql
    db = {}
    if parser.has_section(section):
        params = parser.items(section)
        for param in params:
            db[param[0]] = param[1]
    else:
        raise Exception('Section {0} not found in the {1} file'.format(section, filename))

    return db


class KomatsuPostgres:

    @staticmethod
    def connect():
        """
        Connect to postgres server.
        """
        conn = None
        try:
            print('Connecting to server...')
            params = config()
            conn = psycopg2.connect(**params)
            cur = conn.cursor()
            cur.execute("SELECT version()")
            db_version = cur.fetchone()
            print('Database version:')
            print(db_version)
            cur.close()

        except (Exception, psycopg2.DatabaseError) as error:
            print(error)
        finally:
            if conn is not None:
                conn.close()
                print('Connected. Connection closed.')

    @staticmethod
    def create_sentences_table():
        """
        Create sentences table.
        """
        conn = None
        try:

            print('Creating table...')
            params = config()
            conn = psycopg2.connect(**params)
            cur = conn.cursor()
            cur.execute("CREATE TABLE sentences( uuid UUID UNIQUE NOT NULL, sentence TEXT NOT NULL);")
            cur.close()

        except (Exception, psycopg2.DatabaseError) as error:
            print(error)
        finally:
            if conn is not None:
                conn.close()
                print('Sentences table created. Connection closed.')

    @staticmethod
    def copy_sentences():
        """
        Copy sentences to from CSV.
        """
        conn = None
        try:

            print('Copying sentences...')
            params = config()
            conn = psycopg2.connect(**params)
            cur = conn.cursor()
            cur.execute("COPY sentences(uuid,sentence) FROM 'sentences-comcrawl.csv' DELIMITER ',' CSV HEADER;")
            conn.commit()
            cur.close()

        except (Exception, psycopg2.DatabaseError) as error:
            print(error)
        finally:
            if conn is not None:
                conn.close()
                print('Sentences copied. Connection closed.')

    @staticmethod
    def write_sentences():
        """
        Write all postgres sentences to CSV.
        """
        conn = None
        try:

            print('Writing sentences...')
            params = config()
            conn = psycopg2.connect(**params)
            cur = conn.cursor()
            cur.execute("SELECT * FROM sentences;")
            sentences = cur.fetchone()
            pd.DataFrame(sentences).to_csv('all-sentences.csv')
            cur.close()

        except (Exception, psycopg2.DatabaseError) as error:
            print(error)
        finally:
            if conn is not None:
                conn.close()
