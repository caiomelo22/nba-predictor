#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from pathlib import Path
import os.path


# In[ ]:


def import_data_classification(season = '2019-2019', no_test=False):
    " Importing the dataset "
    
    my_path = os.path.abspath(Path(os.path.abspath(os.path.dirname(__file__))).parent.absolute())
    path = my_path +  '\data\{}.csv'.format(season)
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

