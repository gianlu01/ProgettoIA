import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout

#----------------------------[ IPERPARAMETERS ]--------------------------------

# SIZE = train set dimension (percentage) [0:SIZE] for train and [SIZE:len(x)]
#        for test.

SIZE = 0.70


# EPOCHS = number of traning

EPOCHS = 20


# PREDICTION_DAY = how many days need to look back in order to train

PREDICTION_DAY = 7


# UNITS = number of LSTM units

UNITS = 256


# DROPOUT = the dropout value

DROPOUT = 0.2


# DENSE = number of dense units

DENSE = 1

#------------------------------------------------------------------------------



def TENSORFLOW_Prediction(data : pd.DataFrame):

    
    lunghezza = int(SIZE*len(data.values))
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data.values.reshape(-1, 1))

    train = scaled_data[:lunghezza, :]
    train_x = []
    train_y = []

    for a in range(PREDICTION_DAY, len(train)):
    
        #es: da 0 - 30, 30, coppia (x, y) dato x -> risultato y
        #es: da 1 - 31, 31
        #Addestramento su 30 giorni con target il 30esimo
         
        train_x.append(train[a-PREDICTION_DAY:a, 0])
        train_y.append(train[a, 0])

    train_x = np.array(train_x)
    train_x = np.reshape(train_x, (train_x.shape[0], train_x.shape[1], 1))
    train_y = np.array(train_y)

    test = scaled_data[lunghezza - PREDICTION_DAY:, :]
    test_x = []
    #test_y = scaled_data[int(SIZE*lunghezza):, :]
    test_y = data.values[lunghezza:]

    for a in range(PREDICTION_DAY, len(test)):
         
        test_x.append(test[a-PREDICTION_DAY:a, 0])
    
    test_x = np.array(test_x)
    test_x = np.reshape(test_x, (test_x.shape[0], test_x.shape[1], 1))


    #Definizione del modello
    model = Sequential()
    model.add(LSTM(units = UNITS, return_sequences = True, input_shape=(train_x.shape[1], 1)))
    model.add(Dropout(DROPOUT))
    model.add(LSTM(units = UNITS, return_sequences=True))
    model.add(Dropout(DROPOUT))
    model.add(LSTM(units = UNITS))
    model.add(Dropout(DROPOUT))
    model.add(Dense(units = DENSE))
    model.summary()
    model.compile(optimizer = "adam", loss="mean_squared_error")
    model_data = model.fit(train_x, train_y, epochs = EPOCHS)

    pred = model.predict(test_x)
    pred = scaler.inverse_transform(pred)

    sqm = np.sqrt(mean_squared_error(test_y, pred))

    return pred, sqm

