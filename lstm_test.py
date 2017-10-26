# LSTM for sequence classification in the IMDB dataset
import numpy
# from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.optimizers import SGD
# from keras.layers.embeddings import Embedding
# from keras.preprocessing import sequence
from pprint import pprint
from utils import load_dataset

FEATURE_COUNT = 6
MIN_SAMPLE_LENGTH = 30
LEARNING_RATE = 0.1
DECAY = 1e-5
MOMENTUM = 0.03


X_train, y_train, X_test, y_test = load_dataset()
pprint(X_train.shape)
pprint(y_train.shape)

model = Sequential()
model.add(
    LSTM(
        units=100,
        input_shape=(MIN_SAMPLE_LENGTH, FEATURE_COUNT),
        return_sequences=False,
        unroll=True,
        consume_less='cpu'
    )
)
model.add(Dense(25, activation='softmax'))
sgd_optimizer = SGD(lr=LEARNING_RATE, decay=DECAY, momentum=MOMENTUM)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(X_train, y_train, nb_epoch=3, batch_size=3)

scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
exit(0)


# # fix random seed for reproducibility
# numpy.random.seed(7)
# # load the dataset but only keep the top n words, zero the rest
# top_words = 5000
# (X_train, y_train), (X_test, y_test) = imdb.load_data(nb_words=top_words)
# # truncate and pad input sequences
# max_review_length = 500
# X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
# X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)
# # create the model
# embedding_vector_length = 32
# model = Sequential()
# model.add(Embedding(top_words, embedding_vector_length, input_length=max_review_length))
# model.add(LSTM(100))
# model.add(Dense(1, activation='sigmoid'))
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# print(model.summary())
# model.fit(X_train, y_train, nb_epoch=3, batch_size=64)
# # Final evaluation of the model
# scores = model.evaluate(X_test, y_test, verbose=0)
# print("Accuracy: %.2f%%" % (scores[1]*100))