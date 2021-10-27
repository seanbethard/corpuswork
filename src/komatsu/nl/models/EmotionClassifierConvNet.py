import os
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, GlobalMaxPooling1D
from keras.layers import Conv1D, MaxPooling1D, Embedding, LSTM
from keras.models import Model
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from src.komatsu.nl.models.utils.ModelUtils import ModelUtils
from sklearn.metrics import roc_auc_score


class EmotionClassifierConvNet:

    def __init__(self):
        self.MAX_SEQ_LEN = 100
        self.MAX_V_SIZE = 20000
        self.EMBEDDING_DIM = 100
        self.VALIDATION_SPLIT = 0.2
        self.BATCH_SIZE = 128
        self.EPOCHS = 10
        self.goemotions = ModelUtils.load_goemotions_dataset()
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
        self.checkpoint_path_convnet = "training_convnet/cp.ckpt"
        self.checkpoint_dir_convnet = os.path.dirname(self.checkpoint_path_convnet)
        self.checkpoint_path_lstm = "training_lstm/cp.ckpt"
        self.checkpoint_dir_lstm = os.path.dirname(self.checkpoint_path_lstm)

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

    def predict_sentence(self, sentence):
        """
        Make sentence-level prediction.

        :param sentence: str: sentence
        :return: EagerTensor: predictions
        """

        seqs = self.tokenizer.texts_to_sequences(sentence)
        data = pad_sequences(seqs, maxlen=self.MAX_SEQ_LEN)

        convnet = self.load_convnet(self)
        convnet.load_weights(self.checkpoint_path_convnet)

        convnet_predictions = convnet.predict(data, batch_size=32)
        return convnet_predictions

    @staticmethod
    def evaluate(self):

        convnet = self.load_convnet(self)
        convnet.load_weights(self.checkpoint_path_convnet)

        # plot mean auc for each label
        convnet_predictions = convnet.predict(self.data)

        print("Mean auc convnet:")
        self.print_mean_auc_score(self, convnet_predictions)

    @staticmethod
    def print_mean_auc_score(self, predictions):

        aucs = []

        for i in range(6):
            auc = roc_auc_score(self.targets[:,i], predictions[:,i])
            aucs.append(auc)

        print(np.mean(aucs))
