#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd


# In[ ]:


def import_data_classification(dataset, includes_test=False):
    " Splitting the dataset "
    
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1:].values
    
    " Splitting the dataset into the Training set and Test set and Feature Scaling"
    
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    
    X_test = None
    y_test = None
    sc = StandardScaler()
    if not includes_test:
        X_train = X
        y_train = y
        X_train = sc.fit_transform(X_train)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
    
    
    return dataset, X, y, X_train, X_test, y_train, y_test

