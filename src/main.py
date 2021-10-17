from komatsu.nl import Collection


if __name__ == "__main__":

    # define search query
    search_query = ''

    # get raw HTML
    Collection.get_raw_html(search_query, create_index=False)

    # get sentences
    Collection.sentence_finder('comcrawl.csv')
