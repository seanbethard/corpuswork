from comcrawl import IndexClient
import pandas as pd
import pickle


class Collection:
    """
    Methods for CommonCrawl.
    https://github.com/michaelharms/comcrawl
    """

    @staticmethod
    def get_raw_html(source, create_index=False):
        """
        Writes HTML results from CommonCrawl API to CSV format.
        source: string: URL to search
        create_index: boolean: reload index
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
        pd.DataFrame(client.results).to_csv("results.csv")
