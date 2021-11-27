# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 18:32:11 2021

@author: caiog
"""


import pandas as pd
from pathlib import Path
import os.path

def import_data_classification(season = '2018-2018', no_test=False):
    " Importing the dataset "
    
    my_path = os.path.abspath(Path(os.path.abspath(os.path.dirname(__file__))).parent.absolute())
    # print(my_path +  '\data\seasons\winner\{}.csv'.format(season))
    path = my_path +  '\data\seasons\winner\{}.csv'.format(season)
    # path = os.path.join(my_path, '/data/seasons/winner/{}.csv'.format(season))
    # print(path)
    dataset = pd.read_csv(path)
    X = dataset.iloc[:, 5:-1].values
    y = dataset.iloc[:, -1:].values
    
    " Splitting the dataset into the Training set and Test set and Feature Scaling"
    
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    
    X_test = None
    y_test = None
    sc = StandardScaler()
    if no_test:
        X_train = X
        y_train = y
        X_train = sc.fit_transform(X_train)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
    
    
    return dataset, X, y, X_train, X_test, y_train, y_test

def import_data_regression(season = '2018-2018'):
    " Importing the dataset "
    
    my_path = os.path.abspath(os.path.dirname(__file__))
    path = os.path.join(my_path, '../data/seasons/score/{}.csv'.format(season))
    dataset = pd.read_csv(path)
    X = dataset.iloc[:, 5:-2].values
    y = dataset.iloc[:, -2:].values
    
    " Splitting the dataset into the Training set and Test set "
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
    
    return dataset, X, y, X_train, X_test, y_train, y_test