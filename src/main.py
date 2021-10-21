from komatsu.nl import Collection
from postgres.postgres import KomatsuPostgres


if __name__ == "__main__":

    # define search query
    search_query = ''

    # get comcrawl HTML
    Collection.get_raw_html(search_query, create_index=False)

    # identify sentences in HTML and extract them
    Collection.sentence_finder('comcrawl.csv')

    # write sentences to postgres
    db = KomatsuPostgres()
    db.connect(db)
    db.create_sentences_table(db)
    db.copy_sentences(db)

    # get all sentences from postgres
    db.write_sentences(db)
