from komatsu.nl.models.EmotionClassifierConvNet import EmotionClassifierConvNet
from komatsu.nl.models.EmotionClassifierLSTM import EmotionClassifierLSTM
import pandas as pd


def get_sentences():
    df = pd.read_csv('sentences-comcrawl.csv')
    sentences = df.iloc[:, 1]
    return sentences


if __name__ == "__main__":
    ec_convnet = EmotionClassifierConvNet()
    ec_lstm = EmotionClassifierLSTM()

    ec_convnet.train_convnet(ec_convnet)
    ec_lstm.train_lstm(ec_lstm)

    # mean auc scores
    ec_convnet.evaluate(ec_convnet)
    ec_lstm.evaluate(ec_lstm)

    # sentence-level predictions
    sentences = get_sentences()
    convnet_predictions = ec_convnet.predict_sentence(sentences)
    lstm_predictions = ec_lstm.predict_sentence(sentences)
