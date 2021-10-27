import pandas as pd
import spacy
from bs4 import BeautifulSoup
import uuid
import csv
import pickle
from comcrawl import IndexClient
from nltk.corpus import wordnet as wn
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
import psycopg2
from configparser import ConfigParser


def find_comcrawl_sentences(comcrawl_csv):
    """
    Pull sentences out of CommonCrawl HTML with spaCy.
    https://spacy.io/usage/linguistic-features#sbd

    :param comcrawl_csv: str: comcrawl HTML results
    """
    sbd = spacy.load("en_core_web_lg")
    df = pd.read_csv(comcrawl_csv)

    html_pages = df['html'].tolist()

    uuids = []
    sentences = []

    for page in html_pages:

        soup = BeautifulSoup(page, 'html.parser')

        for paragraph in soup.find_all("p"):
            spacy_doc = sbd(paragraph.get_text())
            for sentence in spacy_doc.sents:
                sentences.append(sentence.text)
                uuids.append(str(uuid.uuid4()))

    with open('sentences-' + comcrawl_csv, mode='w') as f:

        fieldnames = ['uuid', 'sentence']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        assert len(uuids) == len(sentences)

        for i in range(len(uuids)):
            writer.writerow({'uuid': uuids[i], 'sentence': sentences[i]})


def prepare_leipzig_sentences(sents):
    """
    Prepare Leipzig sentences in TSV format for postgres.

    :param sents: path to sentences.txt
    """

    df = pd.read_csv(sents, sep='\t')
    df = df.iloc[:, 1]

    uuids = []

    for i in range(len(df)):
        uuids.append(str(uuid.uuid4()))

    with open(sents + '.csv', mode='w') as f:
        fieldnames = ['uuid', 'sentence']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(len(uuids)):
            writer.writerow({'uuid': uuids[i], 'sentence': df[i]})


class Collection:
    """
    Methods for CommonCrawl.
    https://github.com/michaelharms/comcrawl
    """

    @staticmethod
    def get_raw_html(source, create_index=False):
        """
        Writes HTML results from CommonCrawl API to CSV format.
        :param source: str: search query
        :param create_index: boolean: set to false to load pickled index
        :returns: DataFrame of comcrawl results
        """
        if create_index:
            print('Creating index...')
            client = IndexClient()
            print('Pickling index...')
            with open('client.pickle', 'wb') as f:
                pickle.dump(client, f, protocol=pickle.HIGHEST_PROTOCOL)

        else:
            print("Loading pickled index...")
            with open('client.pickle', 'rb') as f:
                client = pickle.load(f)

        client.search(source)
        print("Downloading from index...")
        client.results = (pd.DataFrame(client.results)
                          .sort_values(by="timestamp")
                          .drop_duplicates("urlkey", keep="last")
                          .to_dict("records"))

        client.download()
        pd.DataFrame(client.results).to_csv("comcrawl.csv")

    @staticmethod
    def sentence_finder(comcrawl_csv):
        """
        Pull sentences out of CommonCrawl HTML with spaCy.
        https://spacy.io/usage/linguistic-features#sbd

        :param comcrawl_csv: str: comcrawl HTML results
        """
        sbd = spacy.load("en_core_web_lg")
        df = pd.read_csv(comcrawl_csv)

        html_pages = df['html'].tolist()

        uuids = []
        sentences = []

        print('Finding sentences...')
        for page in html_pages:

            if isinstance(page, str):

                soup = BeautifulSoup(page, 'html.parser')

                for paragraph in soup.find_all("p"):
                    spacy_doc = sbd(paragraph.get_text())
                    for sentence in spacy_doc.sents:
                        sentences.append(sentence.text)
                        uuids.append(str(uuid.uuid4()))

        with open('sentences-' + comcrawl_csv, mode='w') as f:

            fieldnames = ['uuid', 'sentence']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            assert len(uuids) == len(sentences)

            for i in range(len(uuids)):
                writer.writerow({'uuid': uuids[i], 'sentence': sentences[i]})


class SentenceAnalyzer:

    @staticmethod
    def get_sentiment_scores(sentence):

        # add sentence
        scores = {'sentence': sentence}

        # add nltk scores
        sia = SentimentIntensityAnalyzer()
        scores.update(sia.polarity_scores(sentence))

        # add textblob scores
        polarity_subjectivity = TextBlob(sentence).sentiment._asdict()
        scores.update(polarity_subjectivity)

        return scores

    @staticmethod
    def evaluate_polarity_score(sentiment_scores):
        for result in sentiment_scores:
            if result['polarity'] > 0.5:
                print(result['sentence'])

    @staticmethod
    def get_synsets(sentence):
        tokens = sentence.split()
        synsets = [wn.synsets(token) for token in tokens]
        return synsets


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

    @staticmethod
    def connect(self):
        """
        Connect to postgres server.
        """
        conn = None
        try:

            print('Connecting to server...')
            conn = psycopg2.connect(**self.params)
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
    def create_sentences_table(self):
        """
        Create sentences table.
        """
        conn = None
        try:

            print('Creating table...')
            conn = psycopg2.connect(**self.params)
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
    def copy_sentences(self):
        """
        Copy sentences to from CSV.
        """
        conn = None
        try:

            print('Copying sentences...')
            conn = psycopg2.connect(**self.params)
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
    def write_sentences(self):
        """
        Write all postgres sentences to CSV.
        """
        conn = None
        try:

            print('Writing sentences...')
            conn = psycopg2.connect(**self.params)
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
