import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds
from keras.layers import Bidirectional, LSTM, Dense, Embedding
from keras.models import Sequential

(train_data, test_data), info = tfds.load(
    'imdb_reviews',
    split=[tfds.Split.TRAIN, tfds.Split.TEST],
    as_supervised=True,
    with_info=True
)

tokenizer = keras.preprocessing.text.Tokenizer(num_words=10000, oov_token='<OOV>')
train_sentences, train_labels = zip(*[(sent.numpy().decode('utf8'), label.numpy()) for sent, label in train_data])
tokenizer.fit_on_texts(train_sentences)
train_sequences = tokenizer.texts_to_sequences(train_sentences)
train_padded = keras.preprocessing.sequence.pad_sequences(train_sequences, maxlen=100, padding='post', truncating='post')

test_sentences, test_labels = zip(*[(sent.numpy().decode('utf8'), label.numpy()) for sent, label in test_data])
test_sequences = tokenizer.texts_to_sequences(test_sentences)
test_padded = keras.preprocessing.sequence.pad_sequences(test_sequences, maxlen=100, padding='post', truncating='post')

train_labels = np.array(train_labels)
test_labels = np.array(test_labels)


model = Sequential([
    Embedding(10000, 32),
    Bidirectional(LSTM(32, dropout=0.5, recurrent_dropout=0.5)),
    Dense(1, activation='sigmoid')
])


model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
history = model.fit(train_padded, train_labels, epochs=10, validation_data=(test_padded, test_labels), callbacks=[early_stop])


loss, accuracy = model.evaluate(test_padded, test_labels)
print(f"Test Accuracy: {accuracy:.4f}")