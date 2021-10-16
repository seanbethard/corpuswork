import pandas as pd
import spacy
from bs4 import BeautifulSoup
import uuid
import csv


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


if __name__ == '__main__':
    comcrawl_csv = ''
    sentence_finder(comcrawl_csv)
