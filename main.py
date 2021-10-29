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

import pickle

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
    
def check_model_performance_on_game(game, prediction, bet_value):
    if game['WINNER'] == prediction and game['WINNER'] == 1:
        game_money = (bet_value*game['ODDS_A'] - bet_value)
    elif game['WINNER'] == prediction and game['WINNER'] == 0:
        game_money = (bet_value*game['ODDS_B'] - bet_value)
    else:
        game_money = -bet_value
    return game_money

if __name__ == "__main__":
    season = '2008-2017'
    results = []
    print('Executing the logistic Regression model...')
    Pkl_Filename = "models/LogisticRegressionModel.pkl"  
    try:
        with open(Pkl_Filename, 'rb') as file:  
            logisticRegression = pickle.load(file)
    except:
        logisticRegression = logistic_regression(season)
        with open(Pkl_Filename, 'wb') as file:  
            pickle.dump(logisticRegression, file)
    results.append(dict(model='logistic Regression',cm=logisticRegression[0], acc=logisticRegression[1], classifier=logisticRegression[2]))
    
    print('Executing the Kernel SVM model...')
    Pkl_Filename = "models/KernelSVM.pkl"  
    try:
        with open(Pkl_Filename, 'rb') as file:  
            res = pickle.load(file)
    except:
        res = kernel_svm(season)
        with open(Pkl_Filename, 'wb') as file:  
            pickle.dump(res, file)
    results.append(dict(model='Kernel SVM',cm=res[0], acc=res[1], classifier=res[2]))
    
    print('Executing the Naive Bayes model...')
    Pkl_Filename = "models/NaiveBayes.pkl"  
    try:
        with open(Pkl_Filename, 'rb') as file:  
            res = pickle.load(file)
    except:
        res = naive_bayes(season)
        with open(Pkl_Filename, 'wb') as file:  
            pickle.dump(res, file)
    results.append(dict(model='Naive Bayes',cm=res[0], acc=res[1], classifier=res[2]))
    
    print('Executing the Random Forest model...')
    Pkl_Filename = "models/RandomForest.pkl"  
    try:
        with open(Pkl_Filename, 'rb') as file:  
            res = pickle.load(file)
    except:
        res = random_forest(season)
        with open(Pkl_Filename, 'wb') as file:  
            pickle.dump(res, file)
    results.append(dict(model='Random Forest',cm=res[0], acc=res[1], classifier=res[2]))
    
    # print('Executing the Artificial Neural Network model...')
    # Pkl_Filename = "models/ANN.pkl"  
    # try:
    #     with open(Pkl_Filename, 'rb') as file:  
    #         res = pickle.load(file)
    # except:
    #     res = ann(season)
    #     # with open(Pkl_Filename, 'wb') as file:  
    #     #     pickle.dump(res, file)
    # results.append(dict(model='ANN',cm=res[0], acc=res[1], classifier=res[2]))
    
    # print('Executing the LSTM model...')
    # Pkl_Filename = "models/LSTM.pkl"  
    # try:
    #     with open(Pkl_Filename, 'rb') as file:  
    #         res = pickle.load(file)
    # except:
    #     res = lstm(season)
    #     # with open(Pkl_Filename, 'wb') as file:  
    #     #     pickle.dump(res, file)
    # results.append(dict(model='LSTM',cm=res[0], acc=res[1], classifier=res[2]))
    
    results_regression = []
    print('Executing the Multiple Linear Regression model...')
    Pkl_Filename = "models/MultipleLinearRegression.pkl"  
    try:
        with open(Pkl_Filename, 'rb') as file:  
            res = pickle.load(file)
    except:
        res = multiple_linear_regression(season)
        with open(Pkl_Filename, 'wb') as file:  
            pickle.dump(res, file)
    results_regression.append(dict(model='Multiple Linear Regression',r2_score=res[0], m2_error=res[1]))
    
    print('Executing the Polynomial Regression model...')
    Pkl_Filename = "models/PolynomialRegression.pkl"  
    try:
        with open(Pkl_Filename, 'rb') as file:  
            res = pickle.load(file)
    except:
        res = polynomial_regression(season)
        with open(Pkl_Filename, 'wb') as file:  
            pickle.dump(res, file)
    results_regression.append(dict(model='Polynomial Regression',r2_score=res[0], m2_error=res[1]))
    
    print('Executing the Random Forest Regression model...')
    Pkl_Filename = "models/RandomForestRegression.pkl"  
    try:
        with open(Pkl_Filename, 'rb') as file:  
            res = pickle.load(file)
    except:
        res = random_forest_regression(season)
        with open(Pkl_Filename, 'wb') as file:  
            pickle.dump(res, file)
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
    
    print("\nGetting the matchup baseline...")
    
    matchups = dataset[((dataset['MATCHUP_A'] >= dataset['MATCHUP_B']) & (dataset['WINNER'] == 1)) | 
                       ((dataset['MATCHUP_B'] > dataset['MATCHUP_A']) & (dataset['WINNER'] == 0))]
    print("Baseline Last Matchups: {}/{} -> {}".format(len(matchups.index),len(dataset.index),100*len(matchups.index)/len(dataset.index)))
    
    print("\nGetting the odds baseline...")
    
    matchups = dataset[((dataset['ODDS_A'] <= dataset['ODDS_B']) & (dataset['WINNER'] == 1)) | 
                       ((dataset['ODDS_B'] < dataset['ODDS_A']) & (dataset['WINNER'] == 0))]
    print("Baseline Odds: {}/{} -> {}".format(len(matchups.index),len(dataset.index),100*len(matchups.index)/len(dataset.index)))
    
    print('\nGetting the feature correlation matrix...')
    
    import seaborn as sns
    
    try:
        dependent_variables = dataset.iloc[:,5:-1]
        corrmat = dependent_variables.corr()
        top_corr_features = corrmat.index
        plt.figure(figsize=(13,13))
        #plot heat map
        sns.set(font_scale=0.6)
        g=sns.heatmap(dependent_variables.corr(),annot=True,cmap='Blues', fmt='0.1g')
    except:
        print('No correlation matrix for the selected model.')
    
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_transformed = sc.fit_transform(X)
    
    print('\nGetting model with the best predictions...')
    
    modelCont = 0
    highestAcc = 0
    while True:
        try:
            results[modelCont]['pred'] = results[modelCont]['classifier'].predict(X_transformed)
            results[modelCont]['acc_test'] = accuracy_score(y, results[modelCont]['pred'])
            if results[modelCont]['acc_test'] > highestAcc:
                y_pred = results[modelCont]['pred']
                highestAcc = results[modelCont]['acc_test']
                print('Using predictions from {} model: {}'.format(results[modelCont]['model'], results[modelCont]['acc_test']))
            modelCont += 1
        except IndexError:
            break
    
    print('\nGetting the probabilities of the best model possible...')
    results.sort(key=lambda x: x['acc_test'], reverse=True)
    
    for res in results:
        try:
            y_prob = res['classifier'].predict_proba(X_transformed)
            print('Using the {} model for probability tracking!'.format(res['model']))
            break
        except AttributeError:
            continue
    
    print('\nDisplaying data for the {} model...'.format(results[0]['model']))
    cm = confusion_matrix(y.ravel(), y_pred.ravel())
    acc_score = accuracy_score(y, y_pred)
    print(cm)
    print(acc_score)
    
    profit = 0
    money_by_date = []
    money_by_team = dict()
    best_model_tracking = []
    bets = []
    best_model_tracking.append([dataset.iloc[0,2], 0, 0])
    money_by_date.append([dataset.iloc[0,2], dict(zip([x['model'] for x in results], [0 for x in results])),  dict(zip([x['model'] for x in results], [0 for x in results]))])
    for index, game in dataset.iterrows():
        if game['GAME_DATE'] != money_by_date[-1][0]:
            best_model_tracking.append([game['GAME_DATE'],0,best_model_tracking[-1][2]])
            money_by_date.append([game['GAME_DATE'],  dict(zip([x['model'] for x in results], [0 for x in results])), dict(money_by_date[-1][2])])
        
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
            
            game_money = check_model_performance_on_game(game, y_pred[index], bet_value)
            if game['WINNER'] == y_pred[index] and game['WINNER'] == 1:
                bets.append(['A', game['ODDS_A'], y_prob[index,1], 1])
                money_by_team[game['TEAM_A']] += game_money
            elif game['WINNER'] == y_pred[index] and game['WINNER'] == 0:
                bets.append(['B', game['ODDS_B'], y_prob[index,0], 1])
                money_by_team[game['TEAM_B']] += game_money
            else:
                if y_pred[index] == 1:
                    bets.append(['A', game['ODDS_A'], y_prob[index,1], 0])
                    money_by_team[game['TEAM_A']] += game_money
                else:
                    bets.append(['B', game['ODDS_B'], y_prob[index,0], 0])
                    money_by_team[game['TEAM_B']] += game_money
        
        profit += game_money
        best_model_tracking[-1][1] += game_money
        best_model_tracking[-1][2] += game_money
        
        for model in money_by_date[-1][1]:
            game_money_model = check_model_performance_on_game(game, next(x['pred'][index] for x in results if x['model'] == model), bet_value)
            money_by_date[-1][1][model] += game_money_model
            money_by_date[-1][2][model] += game_money_model
            
    best_model_pred = results[0]['pred']
    best_model_tracking = np.array([x[2] for x in best_model_tracking], dtype=np.float32)
    best_model_actual_tracking = np.array([x[2][results[0]['model']] for x in money_by_date], dtype=np.float32)
        
    # print(index, game['ODDS_A'], game['ODDS_B'], game['WINNER'], y_pred[index], game_money)
        
    # money_by_date = np.array(money_by_date, dtype=str)
    # correct_bets = list(filter(lambda x: x[3] == 1, bets))
    # missed_bets = list(filter(lambda x: x[3] == 0, bets))
    # correct_bets_odds = np.array(list(map(lambda x: x[1], correct_bets)))
    # missed_bets_odds = np.array(list(map(lambda x: x[1], missed_bets)))
    # correct_bets_prob = np.array(list(map(lambda x: x[2], correct_bets)))
    # missed_bets_prob = np.array(list(map(lambda x: x[2], missed_bets)))
    # correct_bets_home = np.array(list(map(lambda x: x[0], correct_bets)))
    # missed_bets_home = np.array(list(map(lambda x: x[0], missed_bets)))
    
    # money_by_team = dict(sorted(money_by_team.items(), key=lambda x: x[1]))
    # money_by_team_labels = np.array(list(money_by_team.keys()), dtype=str)
    # money_by_team_values = np.array(list(money_by_team.values()), dtype=np.float32)
    
    # print('\nPlotting charts...')
    
    # plot_hist('Missed Bets', 'Odds', 'X Times', missed_bets_odds)
    
    # plot_hist('Correct Bets', 'Odds', 'X Times', correct_bets_odds)
    
    # plot_hist('Correct Bets', 'Probability', 'X Times', correct_bets_prob)
    
    # plot_hist('Missed Bets', 'Probability', 'X Times', missed_bets_prob)
    
    # plot_pie_chart('Correct Bets', ['Home', 'Away'], correct_bets_home)
    
    # plot_pie_chart('Missed Bets', ['Home', 'Away'], missed_bets_home)
    
    # plot_bar('Profit By Team', 'Teams', 'Profit', money_by_team_labels, money_by_team_values)
    
    # xpoints = money_by_date[:,0].astype(np.datetime64)
    # ypoints = money_by_date[:,2].astype(np.float32)
    
    # plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    # plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=15))
    # plt.plot(xpoints, ypoints)
    # plt.ylabel("Profit($)")
    # plt.xlabel("Date")
    # plt.title("Profit by Date")
    # plt.gcf().autofmt_xdate()
    # plt.show()
    
    print('\nProfit:', profit)
