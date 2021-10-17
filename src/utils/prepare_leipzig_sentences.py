import csv
import pandas as pd
import uuid


def prepare_sentences(sents):
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


if __name__ == '__main__':
    sentences = ''
    prepare_sentences(sentences)
