#!/usr/bin/env python
# coding: utf-8

# In[1]:


from statistics import mean
import import_data_helper as idh


# In[2]:


def gradient_boosting(dataset, includes_test = False):
    " Importing the dataset "
    
    dataset, X, y, X_train, X_test, y_train, y_test = idh.import_data_classification(dataset, includes_test)
    
    " Training the Ridge model on the Training set "
    
    from sklearn.ensemble import GradientBoostingClassifier
    classifier = GradientBoostingClassifier(random_state = 0)
    classifier.fit(X_train, y_train.ravel())
    
    " Predicting the Test set results and"
    " Making the Confusion Matrix "
    from sklearn.metrics import confusion_matrix, accuracy_score
    cm = None
    acc_score = None
    
    if includes_test:
        y_pred = classifier.predict(X_test)
        cm = confusion_matrix(y_test.ravel(), y_pred.ravel())
        acc_score = accuracy_score(y_test, y_pred)
    
    return cm, acc_score, classifier

