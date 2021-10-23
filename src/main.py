from komatsu.nl import Collection
from postgres.postgres import KomatsuPostgres
import pandas as pd
from komatsu.nl import SentenceAnalyzer
import glob
import os
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, GlobalMaxPooling1D
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

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

    # gsutil cp -r gs://gresearch/goemotions/data/full_dataset/ .
    path = r'full_dataset'
    all_files = glob.glob(path + "/*.csv")

    li = []

    for filename in all_files:
        df = pd.read_csv(filename, index_col=None, header=0)
        li.append(df)

    df = pd.concat(li, axis=0, ignore_index=True)

    # !wget http://nlp.stanford.edu/data/glove.6B.zip

    MAX_SEQ_LEN = 100
    MAX_V_SIZE = 20000
    EMBEDDING_DIM = 100
    VALIDATION_SPLIT = 0.2
    BATCH_SIZE = 128
    EPOCHS = 10

    print("Loading GloVe vectors...")
    word2vec = {}
    with open(os.path.join('glove.6B/glove.6B.%sd.txt' % EMBEDDING_DIM)) as f:
        for line in f:
            values = line.split()
            word = values[0]
            vec = np.asarray(values[1:], dtype='float32')
            word2vec[word] = vec

    print("Loaded %s vectors." % len(word2vec))

    # prepare targets
    sentences = df['text'].fillna('DUMMY_VALUE').values
    possible_labels = ['admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief', 'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization', 'relief', 'remorse', 'sadness', 'surprise', 'neutral']
    targets = df[possible_labels].values

    # get median sequence length
    print("Max sequence length: %s" % max(len(sentence) for sentence in sentences))
    print("Min sequence length: %s" % min(len(sentence) for sentence in sentences))
    s = sorted(len(s) for s in sentences)
    print("Median sequence length: %s" % s[len(s) // 2])

    # get word index
    tokenizer = Tokenizer(num_words=MAX_V_SIZE)
    tokenizer.fit_on_texts(sentences)
    seqs = tokenizer.texts_to_sequences(sentences)
    word2idx = tokenizer.word_index
    print("Number of unique tokens: %s" % len(word2idx))

    # N x T matrix
    data = pad_sequences(seqs, maxlen=MAX_SEQ_LEN)
    print("Shape of data tensor:", data.shape)

    # embedding matrix
    print("Filling GloVe vectors...")
    num_words = min(MAX_V_SIZE, len(word2idx) + 1) # keras word indexes start at 1; 0 is reserved for padding
    embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
    for word, i in word2idx.items():
        if i < MAX_V_SIZE:
            # get() returns Null if not found
            embedding_vector = word2vec.get(word)
            if embedding_vector is not None:
                # oov tokens will be zero
                embedding_matrix[i] = embedding_vector

    # embedding layer
    embedding_layer = Embedding(
        num_words,
        EMBEDDING_DIM,
        weights=[embedding_matrix],
        input_length=MAX_SEQ_LEN,
        trainable=False

    )

    # model definition
    input_ = Input(shape=(MAX_SEQ_LEN,))
    x = embedding_layer(input_)
    x = Conv1D(128, 3, activation='relu')(x)
    x = MaxPooling1D(3)(x)
    x = Conv1D(128, 3, activation='relu')(x)
    x = MaxPooling1D(3)(x)
    x = Conv1D(128, 3, activation='relu')(x)
    x = GlobalMaxPooling1D()(x)
    x = Dense(128, activation='relu')(x)
    output = Dense(len(possible_labels), activation='sigmoid')(x)

    model = Model(input_, output)

    # compile model
    model.compile(
        loss='binary_crossentropy',
        optimizer='rmsprop',
        metrics=['accuracy']
    )

    # fit model
    print("Training 1D convnet with global maxpooling...")
    r = model.fit(
        data,
        targets,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_split=VALIDATION_SPLIT
    )

    # plot loss per iteration
    plt.plot(r.history['loss'], label='loss')
    plt.plot(r.history['val_loss'], label='val_loss')

    # plot acc per iteration
    plt.plot(r.history['accuracy'], label='acc')
    plt.plot(r.history['val_accuracy'], label='val_accuracy')

    # plot mean auc for each label
    p = model.predict(data)

    aucs = []
    for j in range(6):
        auc = roc_auc_score(targets[:,j], p[:,j])
        aucs.append(auc)

    print(np.mean(aucs))
