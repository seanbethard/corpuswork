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

    def __init__(self):
        self.params = config()
        self.conn = psycopg2.connect(**self.params)
        self.cur = self.conn.cursor()

    @staticmethod
    def connect(self):
        """
        Connect to postgres server.
        """
        self.conn = None
        try:
            print('Connecting to server...')
            self.cur.execute("SELECT version()")
            db_version = self.cur.fetchone()
            print('Database version:')
            print(db_version)
            self.cur.close()

        except (Exception, psycopg2.DatabaseError) as error:
            print(error)
        finally:
            if self.conn is not None:
                self.conn.close()
                print('Connected. Connection closed.')

    @staticmethod
    def create_sentences_table(self):
        """
        Create sentences table.
        """
        self.conn = None
        try:

            print('Creating table...')
            self.cur.execute("CREATE TABLE sentences( uuid UUID UNIQUE NOT NULL, sentence TEXT NOT NULL);")
            self.cur.close()

        except (Exception, psycopg2.DatabaseError) as error:
            print(error)
        finally:
            if self.conn is not None:
                self.conn.close()
                print('Sentences table created. Connection closed.')

    @staticmethod
    def copy_sentences(self):
        """
        Copy sentences to from CSV.
        """
        self.conn = None
        try:

            print('Copying sentences...')
            self.cur.execute("COPY sentences(uuid,sentence) FROM 'sentences-comcrawl.csv' DELIMITER ',' CSV HEADER;")
            self.conn.commit()
            self.cur.close()

        except (Exception, psycopg2.DatabaseError) as error:
            print(error)
        finally:
            if self.conn is not None:
                self.conn.close()
                print('Sentences copied. Connection closed.')

    @staticmethod
    def write_sentences(self):
        """
        Write all postgres sentences to CSV.
        """
        self.conn = None
        try:

            print('Writing sentences...')
            self.cur.execute("SELECT * FROM sentences;")
            sentences = self.cur.fetchone()
            pd.DataFrame(sentences).to_csv('all-sentences.csv')
            self.cur.close()

        except (Exception, psycopg2.DatabaseError) as error:
            print(error)
        finally:
            if self.conn is not None:
                self.conn.close()
