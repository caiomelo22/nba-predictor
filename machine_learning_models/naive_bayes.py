#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
from statistics import mean
import pandas as pd
import import_data_helper as idh


# In[ ]:


def naive_bayes(season = '2018-2018', no_test = False):
    " Importing the dataset "
    
    dataset, X, y, X_train, X_test, y_train, y_test = idh.import_data_classification(season)
    
    " Training the model on the Training set "
    
    from sklearn.naive_bayes import GaussianNB
    classifier = GaussianNB()
    classifier.fit(X_train, y_train.ravel())
    
    " Predicting a new result "
    
    # print(classifier.predict(sc.transform([[30,87000]])))
    
    " Predicting the Test set results and"
    " Making the Confusion Matrix "
    from sklearn.metrics import confusion_matrix, accuracy_score
    cm = None
    acc_score = None
    
    if not no_test:
        y_pred = classifier.predict(X_test)
        cm = confusion_matrix(y_test.ravel(), y_pred.ravel())
        acc_score = accuracy_score(y_test, y_pred)
    
    return cm, acc_score, classifier

