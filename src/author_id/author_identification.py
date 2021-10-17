import collections
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import losses
import tensorflow_text as tf_text
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization


class AuthorIdentification:

    def __init__(self):
        self.batch_size = 64
        self.validation_size = 5000
        self.vocab_size = 10002
        self.max_seq_len = 250
        self.autotune = tf.data.AUTOTUNE
        self.buffer_size = 50000
        self.author_dir = 'authors/'
        self.authors = ['foucault.txt', 'kittler.txt', 'kripke.txt']
        self.tokenizer = tf_text.UnicodeScriptTokenizer()
        self.train_data, self.validation_data, self.vocab, self.vocab_table, self.all_labeled_data = self.prepare_datasets()

    def configure_dataset(self, dataset):
        return dataset.cache().prefetch(buffer_size=self.autotune)

    def tokenize(self, text, unused_label):
        lower_case = tf_text.case_fold_utf8(text)
        return self.tokenizer.tokenize(lower_case)

    def configure_dataset(self, dataset):
        return dataset.cache().prefetch(buffer_size=self.autotune)

    def create_model(self, num_labels):
        model = tf.keras.Sequential([
            layers.Embedding(self.vocab_size, 64, mask_zero=True),
            layers.Conv1D(64, 5, padding="valid", activation="relu", strides=2),
            layers.GlobalMaxPooling1D(),
            layers.Dense(num_labels)
        ])
        return model

    def labeler(self, example, index):
        return example, tf.cast(index, tf.int64)

    def prepare_datasets(self):
        # label text lines
        labeled_data_sets = []
        for i, file_name in enumerate(self.authors):
            lines_dataset = tf.data.TextLineDataset(self.author_dir + file_name)
            labeled_dataset = lines_dataset.map(lambda ex: self.labeler(ex, i))
            labeled_data_sets.append(labeled_dataset)

        # combine into one dataset
        all_labeled_data = labeled_data_sets[0]
        for labeled_dataset in labeled_data_sets[1:]:
            all_labeled_data = all_labeled_data.concatenate(labeled_dataset)

        all_labeled_data = all_labeled_data.shuffle(
            self.buffer_size, reshuffle_each_iteration=False)

        tf.data.experimental.save(all_labeled_data, 'all_labeled_data_ds')

        # tokenize samples

        tokenized_ds = all_labeled_data.map(self.tokenize)

        # build vocab
        tokenized_ds = self.configure_dataset(tokenized_ds)
        vocab_dict = collections.defaultdict(lambda: 0)

        for toks in tokenized_ds.as_numpy_iterator():
            for tok in toks:
                vocab_dict[tok] += 1

        vocab = sorted(vocab_dict.items(), key=lambda x: x[1], reverse=True)
        vocab = [token for token, count in vocab]
        vocab = vocab[:self.vocab_size]
        vocab_size = len(vocab)

        keys = vocab
        values = range(2, len(vocab) + 2)

        # map tokens to integers
        init = tf.lookup.KeyValueTensorInitializer(
            keys, values, key_dtype=tf.string, value_dtype=tf.int64)

        vocab_table = tf.lookup.StaticVocabularyTable(init, 1)

        def preprocess_text(text, label):
            standardized = tf_text.case_fold_utf8(text)
            tokenized = self.tokenizer.tokenize(standardized)
            vectorized = vocab_table.lookup(tokenized)
            return vectorized, label

        # vectorize dataset
        all_encoded_data = all_labeled_data.map(preprocess_text)

        # split into train and test
        train_data = all_encoded_data.skip(self.validation_size).shuffle(self.buffer_size)
        validation_data = all_encoded_data.take(self.validation_size)

        train_data = train_data.padded_batch(self.batch_size)
        validation_data = validation_data.padded_batch(self.batch_size)

        # optimize for performance
        train_data = self.configure_dataset(train_data)
        validation_data = self.configure_dataset(validation_data)
        return train_data, validation_data, vocab, vocab_table, all_labeled_data

    def train_and_evaluate(self):
        model = self.create_model(num_labels=3)
        model.compile(
            optimizer='adam',
            loss=losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'])

        history = model.fit(self.train_data, self.validation_data, epochs=3)

        loss, accuracy = model.evaluate(self.validation_data)

        print("Loss: ", loss)
        print("Accuracy: {:2.2%}".format(accuracy))

        preprocess_layer = TextVectorization(
            max_tokens=self.vocab_size,
            standardize=tf_text.case_fold_utf8,
            split=self.tokenizer.tokenize,
            output_mode='int',
            output_sequence_length=self.max_seq_len)
        preprocess_layer.set_vocabulary(self.vocab)

        export_model = tf.keras.Sequential(
            [preprocess_layer, model,
             layers.Activation('sigmoid')])

        export_model.compile(
            loss=losses.SparseCategoricalCrossentropy(from_logits=False),
            optimizer='adam',
            metrics=['accuracy'])

        test_ds = self.all_labeled_data.take(self.validation_size).batch(self.batch_size)
        test_ds = self.configure_dataset(test_ds)
        loss, accuracy = export_model.evaluate(test_ds)
        print("Loss: ", loss)
        print("Accuracy: {:2.2%}".format(accuracy))

        inputs = [
            "The combinatory rules of a given discourse network correspond to its rules  of decomposition.",  # Label: 1
            "Suppose both the speaker and hearer are under a false impression, and that the man to whom they refer is a teetotaler, drinking sparkling water.",
            # Label: 2
            "It belongs, even in minor cases, to the ceremonies by which power is manifested.",  # Label: 0
        ]
        predicted_scores = export_model.predict(inputs)
        predicted_labels = tf.argmax(predicted_scores, axis=1)
        for input, label in zip(inputs, predicted_labels):
            print("Sentence: ", input)
            print("Predicted label: ", label.numpy())


if __name__ == '__main__':
    auth_id = AuthorIdentification()
    auth_id.train_and_evaluate()

