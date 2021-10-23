from komatsu.nl import Collection
from postgres.postgres import KomatsuPostgres
import pandas as pd
from komatsu.nl import SentenceAnalyzer

if __name__ == "__main__":

    # define search query
    search_query = ''

    # get comcrawl HTML
    Collection.get_raw_html(search_query, create_index=True)

    # identify sentences in HTML and extract them
    Collection.sentence_finder('comcrawl.csv')

    # write sentences to postgres
    db = KomatsuPostgres()
    db.connect(db)
    db.create_sentences_table(db)
    db.copy_sentences(db)

    # get all sentences from postgres
    db.write_sentences(db)

    # analyze sentiment
    df = pd.read_csv('sentences-comcrawl.csv')
    sentences = df.iloc[:, 1]

    sentiment_scores = []
    for sentence in sentences:
        sentiment_scores.append(SentenceAnalyzer.get_sentiment_scores(sentence))

    SentenceAnalyzer.evaluate_polarity_score(sentiment_scores)
