import numpy as np
import os

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import classification_report
from keras.layers import Input, Dense, Conv1D, MaxPooling1D, GlobalMaxPooling1D, Flatten
from keras.models import Model
from keras.optimizers import Adam
import random
from keras import initializers

random.seed(42)
np.random.seed(42)

DATA_FOLDER = './HMP_Dataset'

def load_raw_data(data_folder = DATA_FOLDER):
    '''
        Loading raw accelerometer data from HMP dataset
    '''
    X_raw, Y_raw = [], []
    print(os.getcwd())
    for i, (root, dirs, files) in enumerate(os.walk(data_folder)):
        if i == 0:
            continue
        path = root.split(os.sep)
        data_class = path[-1]
        for file in files:
            full_path = root + '/' + file
            try:
                data = open(full_path).readlines()
                num_data = np.array([[int(xi) for xi in x.strip().split()] for x in data])
            except:
                print(full_path)
            X_raw.append(num_data)
            Y_raw.append(data_class)
    return X_raw, Y_raw


def load_data(data_folder = DATA_FOLDER):
    X_raw, y_raw = load_raw_data(data_folder)
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    y = to_categorical(y)
    X = pad_sequences(X_raw, value=[0, 0, 0])
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, random_state=666)
    return X_train, X_test, Y_train, Y_test, le



