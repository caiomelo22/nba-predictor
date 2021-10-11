# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 15:05:48 2021

@author: caiog
"""

import pandas as pd

def import_data():
    " Importing the dataset "
    
    dataset = pd.read_csv('../../data/seasons/score/2019-2019.csv')
    X = dataset.iloc[:, 5:-2].values
    y = dataset.iloc[:, -2:].values
    
    " Splitting the dataset into the Training set and Test set "
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
    
    return dataset, X, y, X_train, X_test, y_train, y_test