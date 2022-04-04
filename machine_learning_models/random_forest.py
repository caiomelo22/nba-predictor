#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from statistics import mean
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score
import import_data_helper as idh


# In[ ]:


def random_search_cv(classifier, X_train, y_train, X_test, y_test):
    " Random Search CV Optmizer "
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                    'max_features': max_features,
                    'max_depth': max_depth,
                    'min_samples_split': min_samples_split,
                    'min_samples_leaf': min_samples_leaf,
                    'bootstrap': bootstrap}
                   
    rf_random = RandomizedSearchCV(estimator = classifier, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
    
    rf_random.fit(X_train, y_train)
    
    best_random = rf_random.best_estimator_
    best_parameters = rf_random.cv_results_
    print(best_parameters)
    print(best_random)
    predictions = best_random.predict(X_test)
    cm = confusion_matrix(y_test, predictions)
    print(cm)
    accuracy_score(y_test, predictions)


# In[ ]:


def random_forest(dataset, includes_test = False, random_search = False):
    " Importing the dataset "
    
    dataset, X, y, X_train, X_test, y_train, y_test = idh.import_data_classification(dataset, includes_test)
    
    " Training the model on the Training set "
    
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import RandomizedSearchCV
    
    # Prameters achieved by running the Random Search CV Optmizer code below
    classifier = RandomForestClassifier(n_estimators = 1000, min_samples_leaf=2, min_samples_split=5, max_features='sqrt', 
                                        max_depth=10, bootstrap = True, criterion = 'entropy', random_state = 0)
    classifier.fit(X_train, y_train.ravel())
    
    if random_search:
        random_search_cv(classifier, X_train, y_train, X_test, y_test)
    
    """Feature Importance"""
    
    feat_importances = pd.Series(classifier.feature_importances_, index=dataset.iloc[:, 5:-1].columns)
    feat_importances.nlargest(30).plot(kind='barh')
    title = 'Feature Importance'
    plt.ylabel('Features')
    plt.xlabel("Feature Importance")
    plt.title(title)
    plt.savefig('charts/{}.png'.format(title.replace(' ','_').lower()), dpi=300)
    plt.show()
    
    " Predicting a new result "
    
    # print('CHA x DEN', classifier.predict_proba(sc.transform([[109.9, 109.8, 0.45630000000000004, 0.3693, 0.6975, 44.1, 12.7, 5.7, 0.4852941176470588, 0.4, 1453.4539592152416, -1, 113.2, 109.9, 0.47219999999999995, 0.34249999999999997, 0.8288999999999997, 42.4, 13.2, 5.5, 0.6470588235294118, 0.8, 1622.89716558378, -2]])))
    
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

