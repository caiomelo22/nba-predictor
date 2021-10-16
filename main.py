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

import os.path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import matplotlib.dates as mdates
from sklearn.metrics import confusion_matrix, accuracy_score

def plot_chart(title, x_label, y_label):
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.title(title)
    plt.show()

def plot_hist(title, x_label, y_label, data):
    plt.hist(data, density=False, bins=30)  # density=False would make counts
    plot_chart(title, x_label, y_label)
    
def plot_bar(title, x_label, y_label, x_data, y_data):
    ax= plt.subplot()
    plt.bar(x_data, y_data) 
    # plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
    plt.xticks(fontsize=10, rotation=90)
    plot_chart(title, x_label, y_label)
    
def plot_pie_chart(title, labels, data):
    data_converted = np.unique(data, return_counts=True)[1]
    plt.pie(data_converted, labels = labels, startangle = 90, shadow = True, autopct='%.2f%%')
    plt.title(title)
    plt.show() 

if __name__ == "__main__":
    season = '2008-2017'
    results = []
    print('Executing the logistic Regression model...')
    logisticRegression = logistic_regression(season)
    results.append(dict(model='logistic Regression',cm=logisticRegression[0], acc=logisticRegression[1], classifier=logisticRegression[2]))
    print('Executing the Kernel SVM model...')
    res = kernel_svm(season)
    results.append(dict(model='Kernel SVM',cm=res[0], acc=res[1], classifier=res[2]))
    print('Executing the Naive Bayes model...')
    res = naive_bayes(season)
    results.append(dict(model='Naive Bayes',cm=res[0], acc=res[1], classifier=res[2]))
    print('Executing the Random Forest model...')
    res = random_forest(season)
    results.append(dict(model='Random Forest',cm=res[0], acc=res[1], classifier=res[2]))
    print('Executing the Artificial Neural Network model...')
    res = ann(season)
    results.append(dict(model='ANN',cm=res[0], acc=res[1], classifier=res[2]))
    print('Executing the LSTM model...')
    res = lstm(season)
    results.append(dict(model='LSTM',cm=res[0], acc=res[1], classifier=res[2]))
    
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
    # # print('Executing the ANN Regression model...')
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
    
    my_path = os.path.abspath(os.path.dirname(__file__))
    path = os.path.join(my_path, 'data/seasons/winner/2018-2018.csv')
    dataset = pd.read_csv(path)
    X = dataset.iloc[:, 5:-1].values
    y = dataset.iloc[:, -1].values
    
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_transformed = sc.fit_transform(X)
    
    y_pred = results[0]['classifier'].predict(X_transformed)
    y_prob = results[0]['classifier'].predict_proba(X_transformed)
    # y_pred = logisticRegression[2].predict(X_transformed)
    # y_prob = logisticRegression[2].predict_proba(X_transformed)
    cm = confusion_matrix(y.ravel(), y_pred.ravel())
    acc_score = accuracy_score(y, y_pred)
    print(cm)
    print(acc_score)
    
    profit = 0
    money_by_date = []
    money_by_team = dict()
    bets = []
    money_by_date.append([dataset.iloc[0,2], 0, 0])
    for index, game in dataset.iterrows():
        if game['GAME_DATE'] != money_by_date[-1][0]:
            money_by_date.append([game['GAME_DATE'], 0, money_by_date[-1][2]])
        
        game_money = 0
        if y_prob[index,0] >= y_prob[index,1]:
            bet_value = 10*y_prob[index,0]
        else:
            bet_value = 10*y_prob[index,1]
        # bet_value = 10
        if (y_pred[index] == 1 and game['ODDS_A'] > 1.25) or (y_pred[index] == 0 and game['ODDS_B'] > 1.25):
            if game['TEAM_A'] not in money_by_team:
                money_by_team[game['TEAM_A']] = 0
            if game['TEAM_B'] not in money_by_team:
                money_by_team[game['TEAM_B']] = 0
                
            if game['WINNER'] == y_pred[index] and game['WINNER'] == 1:
                bets.append(['A', game['ODDS_A'], y_prob[index,1], 1])
                game_money = (bet_value*game['ODDS_A'] - bet_value)
                money_by_team[game['TEAM_A']] += game_money
            elif game['WINNER'] == y_pred[index] and game['WINNER'] == 0:
                bets.append(['B', game['ODDS_B'], y_prob[index,0], 1])
                game_money = (bet_value*game['ODDS_B'] - bet_value)
                money_by_team[game['TEAM_B']] += game_money
            else:
                game_money = -bet_value
                if y_pred[index] == 1:
                    bets.append(['A', game['ODDS_A'], y_prob[index,1], 0])
                    money_by_team[game['TEAM_A']] += game_money
                else:
                    bets.append(['B', game['ODDS_B'], y_prob[index,0], 0])
                    money_by_team[game['TEAM_B']] += game_money
            
        money_by_date[-1][1] += game_money
        money_by_date[-1][2] += game_money
        profit += game_money
        
        # print(index, game['ODDS_A'], game['ODDS_B'], game['WINNER'], y_pred[index], game_money)
        
    money_by_date = np.array(money_by_date, dtype=str)
    correct_bets = list(filter(lambda x: x[3] == 1, bets))
    missed_bets = list(filter(lambda x: x[3] == 0, bets))
    correct_bets_odds = np.array(list(map(lambda x: x[1], correct_bets)))
    missed_bets_odds = np.array(list(map(lambda x: x[1], missed_bets)))
    correct_bets_prob = np.array(list(map(lambda x: x[2], correct_bets)))
    missed_bets_prob = np.array(list(map(lambda x: x[2], missed_bets)))
    correct_bets_home = np.array(list(map(lambda x: x[0], correct_bets)))
    missed_bets_home = np.array(list(map(lambda x: x[0], missed_bets)))
    
    money_by_team = dict(sorted(money_by_team.items(), key=lambda x: x[1]))
    money_by_team_labels = np.array(list(money_by_team.keys()), dtype=str)
    money_by_team_values = np.array(list(money_by_team.values()), dtype=np.float32)
    
    plot_hist('Correct Bets', 'Odds', 'X Times', correct_bets_odds)
    
    plot_hist('Missed Bets', 'Odds', 'X Times', missed_bets_odds)
    
    plot_hist('Correct Bets', 'Probability', 'X Times', correct_bets_prob)
    
    plot_hist('Missed Bets', 'Probability', 'X Times', missed_bets_prob)
    
    plot_pie_chart('Correct Bets', ['Home', 'Away'], correct_bets_home)
    
    plot_pie_chart('Missed Bets', ['Home', 'Away'], missed_bets_home)
    
    plot_bar('Profit By Team', 'Teams', 'Profit', money_by_team_labels, money_by_team_values)
    
    xpoints = money_by_date[:,0].astype(np.datetime64)
    ypoints = money_by_date[:,2].astype(np.float32)
    
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=15))
    plt.plot(xpoints, ypoints)
    plt.ylabel("Profit($)")
    plt.xlabel("Date")
    plt.title("Profit by Date")
    plt.gcf().autofmt_xdate()
    plt.show()
    
    print('Profit:', profit)
