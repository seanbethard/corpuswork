import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import losses
import tensorflow_text as tf_text
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
import pickle

with open('vocab.p', 'rb') as v:
    vocab = pickle.load(v)

BATCH_SIZE = 64
VALIDATION_SIZE = 5000
VOCAB_SIZE = 10002
MAX_SEQUENCE_LENGTH = 250
AUTOTUNE = tf.data.AUTOTUNE

train_data = tf.data.experimental.load('train_ds')
validation_data = tf.data.experimental.load('valid_ds')
all_labeled_data = tf.data.experimental.load('all_labeled_data_ds')

tokenizer = tf_text.UnicodeScriptTokenizer()


def configure_dataset(dataset):
    return dataset.cache().prefetch(buffer_size=AUTOTUNE)


def create_model(vocab_size, num_labels):
    model = tf.keras.Sequential([
        layers.Embedding(vocab_size, 64, mask_zero=True),
        layers.Conv1D(64, 5, padding="valid", activation="relu", strides=2),
        layers.GlobalMaxPooling1D(),
        layers.Dense(num_labels)
    ])
    return model


model = create_model(vocab_size=VOCAB_SIZE, num_labels=3)
model.compile(
    optimizer='adam',
    loss=losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'])

history = model.fit(train_data, validation_data=validation_data, epochs=3)

loss, accuracy = model.evaluate(validation_data)

print("Loss: ", loss)
print("Accuracy: {:2.2%}".format(accuracy))

preprocess_layer = TextVectorization(
    max_tokens=VOCAB_SIZE,
    standardize=tf_text.case_fold_utf8,
    split=tokenizer.tokenize,
    output_mode='int',
    output_sequence_length=MAX_SEQUENCE_LENGTH)
preprocess_layer.set_vocabulary(vocab)

export_model = tf.keras.Sequential(
    [preprocess_layer, model,
     layers.Activation('sigmoid')])

export_model.compile(
    loss=losses.SparseCategoricalCrossentropy(from_logits=False),
    optimizer='adam',
    metrics=['accuracy'])


test_ds = all_labeled_data.take(VALIDATION_SIZE).batch(BATCH_SIZE)
test_ds = configure_dataset(test_ds)
loss, accuracy = export_model.evaluate(test_ds)
print("Loss: ", loss)
print("Accuracy: {:2.2%}".format(accuracy))

inputs = [
    "The combinatory rules of a given discourse network correspond to its rules  of decomposition.",  # Label: 1
    "Suppose both the speaker and hearer are under a false impression, and that the man to whom they refer is a teetotaler, drinking sparkling water.",  # Label: 2
    "It belongs, even in minor cases, to the ceremonies by which power is manifested.",  # Label: 0
]
predicted_scores = export_model.predict(inputs)
predicted_labels = tf.argmax(predicted_scores, axis=1)
for input, label in zip(inputs, predicted_labels):
    print("Sentence: ", input)
    print("Predicted label: ", label.numpy())
