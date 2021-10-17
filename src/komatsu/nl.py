from comcrawl import IndexClient
import pandas as pd
import pickle
import spacy
from bs4 import BeautifulSoup
import uuid
import csv


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
