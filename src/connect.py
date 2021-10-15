#!/usr/bin/python
import psycopg2
from config import config


def connect():
    """
    Connect to PostgreSQL.
    """
    conn = None
    try:

        print('Connecting to PostgreSQL...')
        params = config()
        conn = psycopg2.connect(**params)
        cur = conn.cursor()

        print('Database version:')
        cur.execute('SELECT version()')

        db_version = cur.fetchone()
        print(db_version)

        cur.close()

    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()
            print('Connection closed.')


if __name__ == '__main__':
    connect()
