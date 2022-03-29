#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from statistics import mean
from sklearn.calibration import CalibratedClassifierCV
import import_data_helper as idh


# In[ ]:


def kernel_svm(season = '2018-2018', no_test = False):
    " Importing the dataset "
    
    dataset, X, y, X_train, X_test, y_train, y_test = idh.import_data_classification(season, no_test)
    
    " Training the model on the Training set "
    
    from sklearn.svm import SVC
    classifier = SVC(kernel = 'rbf', random_state = 0, probability = True)
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


