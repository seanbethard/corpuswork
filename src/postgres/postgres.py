#!/usr/bin/python
import pandas as pd
import psycopg2
from config import config


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
            print('Connection closed.')


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
            print('Connection closed.')


def write_sentences():
    """
    Write sentences to table.
    """
    conn = None
    try:

        print('Writing sentences...')
        params = config()
        conn = psycopg2.connect(**params)
        cur = conn.cursor()
        cur.execute("COPY sentences(uuid,sentence) FROM 'comcrawl.csv' DELIMITER ',' CSV HEADER;")
        conn.commit()
        cur.close()

    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()
            print('Connection closed.')


def read_and_write_sentences():
    """
    Read sentences from table and write them to CSV format.
    """
    conn = None
    try:

        print('Reading sentences...')
        params = config()
        conn = psycopg2.connect(**params)
        cur = conn.cursor()
        cur.execute("SELECT * FROM sentences;")
        sentences = cur.fetchone()
        pd.DataFrame(sentences).to_csv('sentences.csv')
        cur.close()

    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()
            print('Connection closed.')


if __name__ == '__main__':
    connect()
