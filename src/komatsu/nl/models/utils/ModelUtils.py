import pandas as pd
import glob


class ModelUtils:

    @staticmethod
    def load_goemotions_dataset():
        # gsutil cp -r gs://gresearch/goemotions/data/full_dataset/ .
        path = r'full_dataset'
        all_files = glob.glob(path + "/*.csv")

        li = []

        for filename in all_files:
            df = pd.read_csv(filename, index_col=None, header=0)
            li.append(df)

        return pd.concat(li, axis=0, ignore_index=True)

    @staticmethod
    def print_sentence_info(sentences, word2idx, data):
        print("Max sequence length: %s" % max(len(sentence) for sentence in sentences))
        print("Min sequence length: %s" % min(len(sentence) for sentence in sentences))
        s = sorted(len(s) for s in sentences)
        print("Median sequence length: %s" % s[len(s) // 2])
        print("Number of unique tokens: %s" % len(word2idx))
        print("Shape of data tensor:", data.shape)


