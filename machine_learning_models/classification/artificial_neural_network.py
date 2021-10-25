# -*- coding: utf-8 -*-
"""Artificial Neural Network.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1aqQOyHl0GlK3nd_XP5EteZq_md234cat
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import keras
import os.path

def ann(season = '2017-2017'):

    " Importing the dataset "
    
    my_path = os.path.abspath(os.path.dirname(__file__))
    path = os.path.join(my_path, '../../data/seasons/winner/{}.csv'.format(season))
    dataset = pd.read_csv(path)
    # dataset['WINNER'] = dataset['WINNER'].map({'A': 1, 'B': 0})
    X = dataset.iloc[:, 5:-1].values
    y = dataset.iloc[:, -1].values
    
    " Splitting the dataset into the Training set and Test set "
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    X_test, X_validation, y_test, y_validation = train_test_split(X_test, y_test, test_size=0.2)
    
    " Feature Scaling "
    
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    X_validation = sc.transform(X_validation)
    
    " Building the ANN "
    
    ann = keras.Sequential([
    
        # input layer
        tf.keras.layers.Dense(units=68, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
        keras.layers.Dropout(0.9),
    
        # 1st dense layer
        tf.keras.layers.Dense(units=68, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
        keras.layers.Dropout(0.9),
        
        tf.keras.layers.Dense(units=1, activation='sigmoid')
    ])
    
    " Compiling the ANN "
    
    ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    " Training the ANN on the Training set "
    
    history = ann.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size = 32, epochs = 100)
    
    " Overfit check "
    
    fig, axs = plt.subplots(2)
    
    # create accuracy sublpot
    axs[0].plot(history.history["accuracy"], label="train accuracy")
    axs[0].plot(history.history["val_accuracy"], label="test accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy eval (ANN)")
    
    # create error sublpot
    axs[1].plot(history.history["loss"], label="train error")
    axs[1].plot(history.history["val_loss"], label="test error")
    axs[1].set_ylabel("Error")
    axs[1].set_xlabel("Epoch")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Error eval (ANN)")
    
    plt.show()
    
    " Predicting single result "
    
    # print(ann.predict(sc.transform([[1, 0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])) > 0.5)
    
    " Predicting results with a margin of certainty"
    
    y_pred = ann.predict(X_validation)
    
    rows = y_pred.shape[0]
    cols = y_pred.shape[1]
    
    y_less_risk_test = []
    y_less_risk_pred = []
    
    for y in range(0, rows -1):
      if y_pred[y][0] <= 0.4 or y_pred[y][0] >= 0.6:
        y_less_risk_test.append(y_validation[y])
        y_less_risk_pred.append(y_pred[y] > 0.5)
    
    y_less_risk_test = np.array(y_less_risk_test)
    y_less_risk_pred = np.array(y_less_risk_pred)
    
    from sklearn.metrics import confusion_matrix, accuracy_score
    cm = confusion_matrix(y_less_risk_test, y_less_risk_pred)
    # print('Predictions with a margin of certainty for the validation set')
    # print(cm)
    # print(accuracy_score(y_less_risk_test, y_less_risk_pred))
    
    y_pred = (y_pred > 0.5)
    
    " Predicting results for all data"
    
    from sklearn.metrics import confusion_matrix, accuracy_score
    cm = confusion_matrix(y_validation, y_pred)
    # print('\nPredictions for the entire validation set')
    acc_score = accuracy_score(y_validation, y_pred)
    # print(cm)
    # print(acc_score)
    
    return cm, acc_score, ann
    
if __name__ == "__main__":
    ann()