#!/usr/bin/env python
# coding: utf-8

# In[1]:


from statistics import mean
import import_data_helper as idh


# In[2]:


def logistic_regression(dataset, includes_test = False):
    " Importing the dataset "
    
    dataset, X, y, X_train, X_test, y_train, y_test = idh.import_data_classification(dataset, includes_test)
    
    " Training the Logistic Regression model on the Training set "
    
    from sklearn.linear_model import LogisticRegression
    classifier = LogisticRegression(random_state = 0)
    classifier.fit(X_train, y_train.ravel())
    
    " Predicting a new result "
    
    # print(classifier.predict(sc.transform([[113.1, 111.1, 0.4654999999999999, 0.37170000000000003, 0.8198000000000001, 43.9, 12.9, 5.8, 0.3, 1516.9609920473383, 105.3, 108.2, 0.44279999999999997, 0.36429999999999996, 0.7499999999999998, 43.7, 15.2, 4.7, 0.8, 1496.2140416710959]])))
    
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

