# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 15:59:07 2021

@author: caiog
"""
import initialize
from logistic_regression import logistic_regression
from artificial_neural_network import ann
from kernel_svm import kernel_svm
from lstm import lstm
from naive_bayes import naive_bayes
from random_forest import random_forest

from artificial_neural_network_regression import ann_regression
from lstm_regression import lstm_regression
from multiple_linear_regression import multiple_linear_regression
from polynomial_regression import polynomial_regression
from random_forest_regression import random_forest_regression

if __name__ == "__main__":
    season = '2008-2016'
    results = []
    print('Executing the logistic Regression model...')
    res = logistic_regression(season)
    results.append(dict(model='logistic Regression',cm=res[0], acc=res[1]))
    print('Executing the Kernel SVM model...')
    res = kernel_svm(season)
    results.append(dict(model='Kernel SVM',cm=res[0], acc=res[1]))
    print('Executing the Naive Bayes model...')
    res = naive_bayes(season)
    results.append(dict(model='Naive Bayes',cm=res[0], acc=res[1]))
    print('Executing the Random Forest model...')
    res = random_forest(season)
    results.append(dict(model='Random Forest',cm=res[0], acc=res[1]))
    print('Executing the Artificial Neural Network model...')
    res = ann(season)
    results.append(dict(model='ANN',cm=res[0], acc=res[1]))
    print('Executing the LSTM model...')
    res = lstm(season)
    results.append(dict(model='LSTM',cm=res[0], acc=res[1]))
    
    results_regression = []
    print('Executing the Multiple Linear Regression model...')
    res = multiple_linear_regression(season)
    results_regression.append(dict(model='Multiple Linear Regression',r2_score=res[0], m2_error=res[1]))
    print('Executing the Polynomial Regression model...')
    res = polynomial_regression(season)
    results_regression.append(dict(model='Polynomial Regression',r2_score=res[0], m2_error=res[1]))
    print('Executing the Random Forest Regression model...')
    res = random_forest_regression(season)
    results_regression.append(dict(model='Random Forest Regression',r2_score=res[0], m2_error=res[1]))
    # print('Executing the ANN Regression model...')
    # res = ann_regression(season)
    # results_regression.append(dict(model='ANN Regression',r2_score=res[0], m2_error=res[1]))
    # print('Executing the LSTM Regression model...')
    # res = lstm_regression(season)
    # results_regression.append(dict(model='LSTM Regression',r2_score=res[0], m2_error=res[1]))
    
    print('\nResults Classification:')
    results.sort(key=lambda x: x['acc'], reverse=True)
    [print('{}:\t{:.2f}'.format(x['model'], x['acc'])) for x in results]
    
    print('\nResults Regression:')
    results_regression.sort(key=lambda x: x['r2_score'], reverse=True)
    [print('{}:\t{:.2f}'.format(x['model'], x['r2_score'])) for x in results_regression]
