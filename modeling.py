##################################################
# modeling.py
#
# the purpose for this file is to make it more
# efficient to model all shops
#
##################################################


# Imports
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam


# Variables
shops = [ 5,  4,  6,  3,  2,  7, 10, 12, 28, 31, 26, 25, 22, 24, 21, 15, 16, 18, 14, 19, 42, 50, 49, 53, 52, 47, 48, 57, 58, 59, 55, 56, 36, 37, 35, 38, 34, 46, 41, 44, 39, 45]
count = 0


def train_split(shop):
    """train-test-splitting"""

    # read in cleaned shop data
    ds = pd.read_csv('./data/clean_shops/shop_' + str(shop) + '.csv')

    # train-test-splitting:
    # I want my model to predict the last month's worth of data and look at mse
    X = ds.iloc[:,:-1]
    y = ds.iloc[:,-1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, random_state=7)

    # reshape train data into a 3d matrix
    X_train_series = X_train.values.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test_series = X_test.values.reshape((X_test.shape[0], X_test.shape[1], 1))
    print('Train set shape: ', X_train_series.shape)
    print('Test set shape: ', X_test_series.shape)

    return X_train_series, X_test_series, y_train, y_test


def compile_test(ds, shop):

    test = pd.read_csv('./data/test.csv')
    test = test.loc[test['shop_id']==shop,]
    test = test.merge(ds, left_on = 'item_id', right_on = ds.index, how='left')
    test.fillna(0, inplace=True)
    test = test.drop(columns = ['ID', 'shop_id']).set_index('item_id')
    test.head()

    return test


def neural_net(X_train_series, X_test_series, y_train, y_test, shop):
    """using lstm to train model"""

    # inspired by https://www.kaggle.com/dimitreoliveira/deep-learning-for-time-series-forecasting
    model_lstm = Sequential()

    # 2 hidden layers - 1 Dense, 1 LSTM
    model_lstm.add(Dense(64, activation='relu', input_shape=(X_train_series.shape[1], X_train_series.shape[2]))) # (33,1)
    model_lstm.add(LSTM(32, activation='relu', input_shape=(X_train_series.shape[1], X_train_series.shape[2]))) # (33,1)

    # output layer
    model_lstm.add(Dense(1))

    # compile
    model_lstm.compile(loss='mse', optimizer=Adam(lr=.0001))

    # Early Stopping 
    early_stop = EarlyStopping(min_delta = 0, patience = 5)

    model_lstm.fit(X_train_series, 
                  y_train, 
                  batch_size = 512, 
                  validation_data=(X_test_series, y_test), 
                  callbacks = [early_stop], 
                  epochs=15, 
                  verbose=1)

    return model_lstm


def after_predict(predictions, shop, test):

    preds = pd.DataFrame(predictions, index = test.index)
    preds['shop_id'] = shop
    preds = preds.reset_index()
    preds = preds.reindex(columns = ['shop_id', 'item_id', 0])
    preds.to_csv('./data/output/shop_' + str(shop) + '.csv', index=False)

    return preds
    

    



