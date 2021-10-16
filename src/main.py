from komatsu.nl import Collection


if __name__ == "__main__":
    search_query = ''
    Collection.get_raw_html(search_query, create_index=False)

