from comcrawl import IndexClient
import pandas as pd
import pickle
import spacy
from bs4 import BeautifulSoup
import uuid
import csv
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
import glob
import os
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, GlobalMaxPooling1D
from keras.layers import Conv1D, MaxPooling1D, Embedding, LSTM
from keras.models import Model
from sklearn.metrics import roc_auc_score
import tensorflow as tf
from nltk.corpus import wordnet as wn
from tensorflow.keras.callbacks import EarlyStopping


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


class EmotionClassifier:

    def __init__(self):
        self.MAX_SEQ_LEN = 100
        self.MAX_V_SIZE = 20000
        self.EMBEDDING_DIM = 100
        self.VALIDATION_SPLIT = 0.2
        self.BATCH_SIZE = 128
        self.EPOCHS = 10
        self.goemotions = self.load_goemotions_dataset(self)
        self.word2vec = self.load_word2vec(self)
        self.sentences = self.goemotions['text'].fillna('DUMMY_VALUE').values
        self.possible_labels = ['admiration',
                                'amusement',
                                'anger',
                                'annoyance',
                                'approval',
                                'caring',
                                'confusion',
                                'curiosity',
                                'desire',
                                'disappointment',
                                'disapproval',
                                'disgust',
                                'embarrassment',
                                'excitement',
                                'fear',
                                'gratitude',
                                'grief',
                                'joy',
                                'love',
                                'nervousness',
                                'optimism',
                                'pride',
                                'realization',
                                'relief',
                                'remorse',
                                'sadness',
                                'surprise',
                                'neutral']
        self.targets = self.goemotions[self.possible_labels].values
        self.tokenizer = Tokenizer(num_words=self.MAX_V_SIZE)
        self.tokenizer.fit_on_texts(self.sentences)
        self.seqs = self.tokenizer.texts_to_sequences(self.sentences)
        self.word2idx = self.tokenizer.word_index
        self.data = pad_sequences(self.seqs, maxlen=self.MAX_SEQ_LEN)
        self.num_words, self.embedding_matrix = self.load_embedding_matrix(self)
        self.convnet = self.load_convnet(self)
        self.lstm = self.load_lstm(self)
        self.checkpoint_path_convnet = "training_convnet/cp.ckpt"
        self.checkpoint_dir_convnet = os.path.dirname(self.checkpoint_path_convnet)
        self.checkpoint_path_lstm = "training_lstm/cp.ckpt"
        self.checkpoint_dir_lstm = os.path.dirname(self.checkpoint_path_lstm)

    @staticmethod
    def load_goemotions_dataset(self):
        # gsutil cp -r gs://gresearch/goemotions/data/full_dataset/ .
        path = r'full_dataset'
        all_files = glob.glob(path + "/*.csv")

        li = []

        for filename in all_files:
            df = pd.read_csv(filename, index_col=None, header=0)
            li.append(df)

        return pd.concat(li, axis=0, ignore_index=True)

    @staticmethod
    def load_word2vec(self):
        # !wget http://nlp.stanford.edu/data/glove.6B.zip
        # !unzip glove.6B.zip
        print("Loading GloVe vectors...")
        word2vec = {}
        with open(os.path.join('glove.6B/glove.6B.%sd.txt' % self.EMBEDDING_DIM)) as f:
            for line in f:
                values = line.split()
                word = values[0]
                vec = np.asarray(values[1:], dtype='float32')
                word2vec[word] = vec

        print("Loaded %s vectors." % len(word2vec))
        return word2vec

    @staticmethod
    def load_embedding_matrix(self):
        print("Filling GloVe vectors...")
        num_words = min(self.MAX_V_SIZE, len(self.word2idx) + 1)
        embedding_matrix = np.zeros((num_words, self.EMBEDDING_DIM))
        for word, i in self.word2idx.items():
            if i < self.MAX_V_SIZE:
                # returns Null if not found
                embedding_vector = self.word2vec.get(word)
                if embedding_vector is not None:
                    # oov tokens will be zero
                    embedding_matrix[i] = embedding_vector
        return num_words, embedding_matrix

    @staticmethod
    def load_convnet(self):
        """
        1D convnet with global maxpooling.")
        """

        embedding_layer = Embedding(
            self.num_words,
            self.EMBEDDING_DIM,
            weights=[self.embedding_matrix],
            input_length=self.MAX_SEQ_LEN,
            trainable=False
        )

        input_ = Input(shape=(self.MAX_SEQ_LEN,))
        x = embedding_layer(input_)
        x = Conv1D(128, 3, activation='relu')(x)
        x = MaxPooling1D(3)(x)
        x = Conv1D(128, 3, activation='relu')(x)
        x = MaxPooling1D(3)(x)
        x = Conv1D(128, 3, activation='relu')(x)
        x = GlobalMaxPooling1D()(x)
        x = Dense(128, activation='relu')(x)
        output = Dense(len(self.possible_labels), activation='sigmoid')(x)

        model = Model(input_, output)

        # compile model
        model.compile(
            loss='binary_crossentropy',
            optimizer='rmsprop',
            metrics=['accuracy']
        )

        print(model.summary())

        return model

    @staticmethod
    def load_lstm(self):
        """
        LSTM with global maxpooling.
        """

        # hidden state dimensionality
        M = 15

        embedding_layer = Embedding(
            self.num_words,
            self.EMBEDDING_DIM,
            weights=[self.embedding_matrix],
            input_length=self.MAX_SEQ_LEN,
            trainable=False
        )

        i = Input(shape=(self.MAX_SEQ_LEN,))
        x = embedding_layer(i)

        x = LSTM(M, return_sequences=True)(x)
        x = GlobalMaxPooling1D()(x)

        x = Dense(28, activation='softmax')(x)

        model = Model(i,x)

        model.compile(
            loss='categorical_crossentropy',
            optimizer='rmsprop',
            metrics=['accuracy']
        )

        print(model.summary())

        return model

    @staticmethod
    def print_sentence_info(self):
        print("Max sequence length: %s" % max(len(sentence) for sentence in self.sentences))
        print("Min sequence length: %s" % min(len(sentence) for sentence in self.sentences))
        s = sorted(len(s) for s in self.sentences)
        print("Median sequence length: %s" % s[len(s) // 2])
        print("Number of unique tokens: %s" % len(self.word2idx))
        print("Shape of data tensor:", self.data.shape)

    @staticmethod
    def train_convnet(self):

        early_callback = EarlyStopping(monitor='val_accuracy', patience=1)

        convnet_callback = tf.keras.callbacks.ModelCheckpoint(filepath=self.checkpoint_path_convnet,
                                                         save_weights_only=True,
                                                         verbose=1)

        # train model
        r = self.convnet.fit(
            self.data,
            self.targets,
            batch_size=self.BATCH_SIZE,
            epochs=self.EPOCHS,
            validation_split=self.VALIDATION_SPLIT,
            callbacks=[early_callback, convnet_callback]
        )

    @staticmethod
    def train_lstm(self):

        early_callback = EarlyStopping(monitor='val_accuracy', patience=1)

        lstm_callback = tf.keras.callbacks.ModelCheckpoint(filepath=self.checkpoint_path_lstm,
                                                           save_weights_only=True,
                                                           verbose=1)

        # train model
        r = self.lstm.fit(
            self.data,
            self.targets,
            batch_size=self.BATCH_SIZE,
            epochs=self.EPOCHS,
            validation_split=self.VALIDATION_SPLIT,
            callbacks=[early_callback, lstm_callback]
        )

    def evaluate_model(self):

        lstm = self.load_lstm(self)
        lstm.load_weights(self.checkpoint_path_lstm)

        # plot mean auc for each label
        lstm_predictions = lstm.predict(self.data)

        print("Mean auc lstm:")
        self.print_mean_auc_score(self, lstm_predictions)

    @staticmethod
    def print_mean_auc_score(self, predictions):

        aucs = []

        for i in range(6):
            auc = roc_auc_score(self.targets[:,i], predictions[:,i])
            aucs.append(auc)

        print(np.mean(aucs))

    def predict_sentence(self, sentence):
        """
        Make sentence-level prediction.

        :param sentence: str: sentence
        :return: EagerTensor: predictions
        """

        seqs = self.tokenizer.texts_to_sequences(sentence)
        data = pad_sequences(seqs, maxlen=self.MAX_SEQ_LEN)

        lstm = self.load_lstm(self)
        lstm.load_weights(self.checkpoint_path_lstm)

        lstm_predictions = lstm.predict(data, batch_size=32)
        return lstm_predictions
