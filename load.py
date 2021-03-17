import io 
import os
import json
import tensorflow as tf

# import libraries for reading data, exploring and plotting
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator


# library for train test split
from sklearn.model_selection import train_test_split

# deep learning libraries for text pre-processing
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Modeling 
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense, Dropout, LSTM, Bidirectional
from keras_preprocessing.text import tokenizer_from_json

max_len = 50 
trunc_type = "post" 
padding_type = "post" 
oov_tok = "<OOV>" 
vocab_size = 500
embeding_dim = 16
drop_value = 0.2 # dropout
n_dense=24

def load_tokenizer():
    with open('tokenizer.json') as f: 
            data = json.load(f) 
            tokenizer = tokenizer_from_json(data)
    return tokenizer

def load_model():
    model = Sequential()
    model.add(Embedding(vocab_size, embeding_dim, input_length=max_len))
    model.add(GlobalAveragePooling1D())
    model.add(Dense(24, activation='relu'))
    model.add(Dropout(drop_value))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',optimizer='adam' ,metrics=['accuracy'])
    model.load_weights('my_model.h5')
    return model

def predict_spam(predict_msg,tokenizer, model):
    new_seq = tokenizer.texts_to_sequences(predict_msg)
    padded = pad_sequences(new_seq, maxlen =max_len,
                      padding = padding_type,
                      truncating=trunc_type)
    prediction_msg=model.predict(padded)
    if prediction_msg<=0.4:
        return 'HAM'
    else:
        return 'SPAM'
    

