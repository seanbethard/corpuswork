import collections
import tensorflow as tf
import tensorflow_text as tf_text


BUFFER_SIZE = 50000
BATCH_SIZE = 64
VALIDATION_SIZE = 5000
VOCAB_SIZE = 10000
MAX_SEQUENCE_LENGTH = 250
AUTOTUNE = tf.data.AUTOTUNE
PARENT_DIR = 'authors/'
FILE_NAMES = ['foucault.txt', 'kittler.txt', 'kripke.txt']
import pickle


def configure_dataset(dataset):
    return dataset.cache().prefetch(buffer_size=AUTOTUNE)


def labeler(example, index):
    return example, tf.cast(index, tf.int64)


def tokenize(text, unused_label):
    lower_case = tf_text.case_fold_utf8(text)
    return tokenizer.tokenize(lower_case)


def preprocess_text(text, label):
  standardized = tf_text.case_fold_utf8(text)
  tokenized = tokenizer.tokenize(standardized)
  vectorized = vocab_table.lookup(tokenized)
  return vectorized, label


# label text lines
labeled_data_sets = []
for i, file_name in enumerate(FILE_NAMES):
    lines_dataset = tf.data.TextLineDataset(PARENT_DIR + file_name)
    labeled_dataset = lines_dataset.map(lambda ex: labeler(ex, i))
    labeled_data_sets.append(labeled_dataset)

# combine into one dataset
all_labeled_data = labeled_data_sets[0]
for labeled_dataset in labeled_data_sets[1:]:
    all_labeled_data = all_labeled_data.concatenate(labeled_dataset)

all_labeled_data = all_labeled_data.shuffle(
    BUFFER_SIZE, reshuffle_each_iteration=False)

tf.data.experimental.save(all_labeled_data, 'all_labeled_data_ds')

# tokenize samples
tokenizer = tf_text.UnicodeScriptTokenizer()
tokenized_ds = all_labeled_data.map(tokenize)

# build vocab
tokenized_ds = configure_dataset(tokenized_ds)
vocab_dict = collections.defaultdict(lambda: 0)

for toks in tokenized_ds.as_numpy_iterator():
    for tok in toks:
      vocab_dict[tok] += 1

vocab = sorted(vocab_dict.items(), key=lambda x: x[1], reverse=True)
vocab = [token for token, count in vocab]
vocab = vocab[:VOCAB_SIZE]
vocab_size = len(vocab)

with open('vocab.p', 'wb') as v:
    pickle.dump(vocab, v)

keys = vocab
values = range(2, len(vocab) + 2)

# map tokens to integers
init = tf.lookup.KeyValueTensorInitializer(
    keys, values, key_dtype=tf.string, value_dtype=tf.int64)

num_oov_buckets = 1
vocab_table = tf.lookup.StaticVocabularyTable(init, num_oov_buckets)

# vectorize dataset
all_encoded_data = all_labeled_data.map(preprocess_text)

# split into train and test
train_data = all_encoded_data.skip(VALIDATION_SIZE).shuffle(BUFFER_SIZE)
validation_data = all_encoded_data.take(VALIDATION_SIZE)

train_data = train_data.padded_batch(BATCH_SIZE)
validation_data = validation_data.padded_batch(BATCH_SIZE)

# add padding and OOV tokens to vocab
vocab_size += 2

# optimize for performance
train_data = configure_dataset(train_data)
validation_data = configure_dataset(validation_data)

tf.data.experimental.save(train_data, 'train_ds')
tf.data.experimental.save(validation_data, 'valid_ds')
