#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import keras
import os.path


# In[ ]:


def parse_lstm_data(X, y, timesteps=10):
    tracking = []
    info_tracking = []
    features = []
    labels = []
    
    info = np.concatenate((X[:,[2]], X[:,-2:]), axis = 1)
    info = np.c_[ info[:,:], y ]  
    
    " Feature Scaling "
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X[:,3:] = sc.fit_transform(X[:,3:])
    
    # print('Parsing the data to LSTM format...')
    for i in range(2, len(X), 2):
        team_a_id = X[i-2,1]
        team_b_id = X[i-1,1]
        team_a_abbv = X[i-2,0]
        team_b_abbv = X[i-1,0]
        team_a_previous_games = X[(X[:,1] == team_a_id) & (X[:,2] < X[i-1,2]),:]
        team_b_previous_games = X[(X[:,1] == team_b_id) & (X[:,2] < X[i-1,2]),:]
        if len(team_a_previous_games) >= timesteps and len(team_b_previous_games) >= timesteps:
            game_tracking = np.concatenate((team_a_previous_games[-1*timesteps:, 1:], team_b_previous_games[-1*timesteps:, 1:]), axis = 1)
            game = np.concatenate((team_a_previous_games[-1*timesteps:, 3:], team_b_previous_games[-1*timesteps:, 3:]), axis = 1)
            tracking.append(game_tracking)
            features.append(game)
            info_tracking.append(info[i-2,:])
            labels.append(y[i-2])
            
    features = np.array(features).astype(np.float32)
    labels = np.array(labels).astype(np.float32)
    return features, labels, info_tracking


# In[ ]:


def build_lstm(X_train, y_train, X_test = None, y_test = None):
    " Building the LSTM "
    
    lstm = keras.Sequential()
    lstm.add(keras.layers.LSTM(12, input_shape=(X_train.shape[1], X_train.shape[2])))
    lstm.add(keras.layers.Dropout(0.9))
    
    # Output layer
    lstm.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
    
    " Compiling the LSTM "
    optimiser = tf.keras.optimizers.Adam()
    lstm.compile(optimizer=optimiser,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    lstm.summary()
    
    " Training the LSTM on the Training set "
    
    if X_test != None:
        history = lstm.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size = 32, epochs = 100)
    else:
        history = lstm.fit(X_train, y_train, batch_size = 32, epochs = 100)
        
    return lstm, history


# In[ ]:


def import_dataset(season = '2018-2018'):
    
    " Importing the dataset "
    
    my_path = os.path.abspath(os.path.dirname(__file__))
    path = os.path.join(my_path, '../data/LSTM/{}.csv'.format(season))
    dataset = pd.read_csv(path)
    dataset['DATE'] = pd.to_datetime(dataset['DATE'])
    X = dataset.iloc[:, 1:-1].values
    y = dataset.iloc[:, -1].values
    
    return X,y
    


# In[ ]:


def lstm_no_validation(season = '2018-2018'):
    X, y = import_dataset(season)
    
    features, labels, info = parse_lstm_data(X, y)
    
    lstm, history = build_lstm(features, labels)
    
    return lstm, history
    


# In[ ]:


def lstm(season = '2018-2018'):
    X, y = import_dataset(season)
    
    " Splitting the dataset into the Training set and Test set "
    
    features, labels, info = parse_lstm_data(X, y)
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.25)
    
    " Predicting single result "
    
    # print(lstm.predict(sc.transform([[1, 0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])) > 0.5)
    
    " Predicting results with a margin of certainty"
    
    y_pred = lstm.predict(X_test)
    
    rows = y_pred.shape[0]
    cols = y_pred.shape[1]
    
    y_less_risk_test = []
    y_less_risk_pred = []
    
    for y in range(0, rows -1):
      if y_pred[y][0] <= 0.4 or y_pred[y][0] >= 0.6:
        y_less_risk_test.append(y_test[y])
        y_less_risk_pred.append(y_pred[y] > 0.5)
    
    y_less_risk_test = np.array(y_less_risk_test)
    y_less_risk_pred = np.array(y_less_risk_pred)
    
#     from sklearn.metrics import confusion_matrix, accuracy_score
#     cm = confusion_matrix(y_less_risk_test, y_less_risk_pred)
    # print('Predictions with a margin of certainty for the validation set')
    # print(cm)
    # print(accuracy_score(y_less_risk_test, y_less_risk_pred))
    
    y_pred = (y_pred > 0.5)
    
    " Predicting results for all data"
    
#     from sklearn.metrics import confusion_matrix, accuracy_score
#     cm = confusion_matrix(y_validation, y_pred)
    # print('\nPredictions for the entire validation set')
#     acc_score = accuracy_score(y_validation, y_pred)
    # print(cm)
    # print(acc_score)
    
    return cm, acc_score, lstm

