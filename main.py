# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 15:59:07 2021

@author: caiog
"""
import initialize
from logistic_regression import logistic_regression
from artificial_neural_network import ann_no_validation
from kernel_svm import kernel_svm
from lstm import lstm_no_validation, parse_lstm_data
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
from keras.models import model_from_json

import pickle
import dill

threshold = 1.75

def plot_chart(title, x_label, y_label):
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.title(title)
    plt.savefig('charts/{}.png'.format(title.replace(' ','_').lower()), dpi=300)
    plt.show()

def plot_hist(title, x_label, y_label, data):
    plt.hist(data, density=False, bins=20)  # density=False would make counts
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
    plt.savefig('charts/{}.png'.format(title.replace(' ','_').lower()), dpi=300)
    plt.show() 
    

def check_game_with_odds(game, bet_value):
    game_money = 0
    if (game['ODDS_A'] <= game['ODDS_B'] and game['ODDS_A'] > threshold) or (game['ODDS_A'] > game['ODDS_B'] and game['ODDS_B'] > threshold):
        prediction = game['ODDS_A'] <= game['ODDS_B']
        if game['WINNER'] == prediction and game['WINNER'] == 1:
            game_money = (bet_value*game['ODDS_A'] - bet_value)
        elif game['WINNER'] == prediction and game['WINNER'] == 0:
            game_money = (bet_value*game['ODDS_B'] - bet_value)
        else:
            game_money = -bet_value
    return game_money
    
def check_game_with_matchups(game, bet_value):
    game_money = 0
    if (game['MATCHUP_A'] > game['MATCHUP_B'] and game['ODDS_A'] > threshold) or (game['MATCHUP_A'] < game['MATCHUP_B'] and game['ODDS_B'] > threshold):
        prediction = game['MATCHUP_A'] > game['MATCHUP_B']
        if game['WINNER'] == prediction and game['WINNER'] == 1:
            game_money = (bet_value*game['ODDS_A'] - bet_value)
        elif game['WINNER'] == prediction and game['WINNER'] == 0:
            game_money = (bet_value*game['ODDS_B'] - bet_value)
        else:
            game_money = -bet_value
    return game_money

def check_model_performance_on_game_lstm(game, prediction, bet_value):
    game_money = 0
    if (prediction == 1 and game[1] > threshold) or (prediction == 0 and game[2] > threshold):
        if game[3] == prediction and game[3] == 1:
            game_money = (bet_value*game[1] - bet_value)
        elif game[3] == prediction and game[3] == 0:
            game_money = (bet_value*game[2] - bet_value)
        else:
            game_money = -bet_value
    return game_money

def check_model_performance_on_game(game_lstm, prediction, bet_value):
    game_money = 0
    if (prediction == 1 and game['ODDS_A'] > threshold) or (prediction == 0 and game['ODDS_B'] > threshold):
        if game['WINNER'] == prediction and game['WINNER'] == 1:
            game_money = (bet_value*game['ODDS_A'] - bet_value)
        elif game['WINNER'] == prediction and game['WINNER'] == 0:
            game_money = (bet_value*game['ODDS_B'] - bet_value)
        else:
            game_money = -bet_value
    return game_money

def load_neural_net(model_name, model):
    try:
        # load json and create model
        json_file = open('models/{}.json'.format(model_name), 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        res = model_from_json(loaded_model_json)
        # load weights into new model
        res.load_weights("models/{}.h5".format(model_name))
    except:
        res = model(season)[0]
        # serialize model to JSON
        model_json = res.to_json()
        with open("models/{}.json".format(model_name), "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        res.save_weights("models/{}.h5".format(model_name))
    return res
    

if __name__ == "__main__":
    season = '2008-2017'
    season_test = '2018-2018'
    results = []
    
    print('\nGetting data for the 2018 season for testing...')
    my_path = os.path.abspath(os.path.dirname(__file__))
    
    dataset_train = pd.read_csv(os.path.join(my_path, 'data/seasons/winner/{}.csv'.format(season)))
    dataset_regression = pd.read_csv(os.path.join(my_path, 'data/seasons/score/{}.csv'.format(season_test)))
    dataset_lstm = pd.read_csv(os.path.join(my_path, 'data/seasons/winner/LSTM/{}.csv'.format(season_test)))
    path = os.path.join(my_path, 'data/seasons/winner/{}.csv'.format(season_test))
    dataset = pd.read_csv(path)
    
    X = dataset.iloc[:, 5:-1].values
    y = dataset.iloc[:, -1].values
    
    X_lstm = dataset_lstm.iloc[:, 1:-1].values
    y_lstm = dataset_lstm.iloc[:, -1].values
    
    X_regression = dataset_regression.iloc[:, 5:-2].values
    y_regression = dataset_regression.iloc[:, -2:].values
    
    print('\nExecuting the logistic Regression model...')
    Pkl_Filename = "models/LogisticRegressionModel.pkl"  
    try:
        with open(Pkl_Filename, 'rb') as file:  
            logisticRegression = pickle.load(file)
    except:
        logisticRegression = logistic_regression(season, True)
        with open(Pkl_Filename, 'wb') as file:  
            pickle.dump(logisticRegression, file)
    results.append(dict(model='logistic Regression',cm=logisticRegression[0], acc=logisticRegression[1], classifier=logisticRegression[2]))
    
    print('Executing the Kernel SVM model...')
    Pkl_Filename = "models/KernelSVM.pkl"  
    try:
        with open(Pkl_Filename, 'rb') as file:  
            res = pickle.load(file)
    except:
        res = kernel_svm(season, True)
        with open(Pkl_Filename, 'wb') as file:  
            pickle.dump(res, file)
    results.append(dict(model='Kernel SVM',cm=res[0], acc=res[1], classifier=res[2]))
    
    print('Executing the Naive Bayes model...')
    Pkl_Filename = "models/NaiveBayes.pkl"  
    try:
        with open(Pkl_Filename, 'rb') as file:  
            res = pickle.load(file)
    except:
        res = naive_bayes(season, True)
        with open(Pkl_Filename, 'wb') as file:  
            pickle.dump(res, file)
    results.append(dict(model='Naive Bayes',cm=res[0], acc=res[1], classifier=res[2]))
    
    print('Executing the Random Forest model...')
    Pkl_Filename = "models/RandomForest.pkl"  
    try:
        with open(Pkl_Filename, 'rb') as file:  
            res = pickle.load(file)
    except:
        res = random_forest(season, True)
        with open(Pkl_Filename, 'wb') as file:  
            pickle.dump(res, file)
    results.append(dict(model='Random Forest',cm=res[0], acc=res[1], classifier=res[2]))
    
    print('Executing the Artificial Neural Network model...')
    res = load_neural_net("ANN", ann_no_validation)
    results.append(dict(model='ANN', classifier=res))
    
    print('Executing the LSTM model...')
    res = load_neural_net("LSTM", lstm_no_validation)
    results.append(dict(model='LSTM', classifier=res))
    
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
    results_regression.append(dict(model='Multiple Linear Regression',r2_score=res[0], m2_error=res[1], regressor=res[2]))
    
    print('Executing the Polynomial Regression model...')
    Pkl_Filename = "models/PolynomialRegression.pkl"  
    try:
        with open(Pkl_Filename, 'rb') as file:  
            res = pickle.load(file)
    except:
        res = polynomial_regression(season)
        with open(Pkl_Filename, 'wb') as file:  
            pickle.dump(res, file)
    results_regression.append(dict(model='Polynomial Regression',r2_score=res[0], m2_error=res[1], regressor=res[2]))
    
    print('Executing the Random Forest Regression model...')
    Pkl_Filename = "models/RandomForestRegression.pkl"  
    try:
        with open(Pkl_Filename, 'rb') as file:  
            res = pickle.load(file)
    except:
        res = random_forest_regression(season)
        with open(Pkl_Filename, 'wb') as file:  
            pickle.dump(res, file)
    results_regression.append(dict(model='Random Forest Regression',r2_score=res[0], m2_error=res[1], regressor=res[2]))
    
    matchups_baseline = dataset_train[((dataset_train['MATCHUP_A'] >= dataset_train['MATCHUP_B']) & (dataset_train['WINNER'] == 1)) | 
                       ((dataset_train['MATCHUP_B'] > dataset_train['MATCHUP_A']) & (dataset_train['WINNER'] == 0))]
    odds_baseline = dataset_train[((dataset_train['ODDS_A'] <= dataset_train['ODDS_B']) & (dataset_train['WINNER'] == 1)) | 
                       ((dataset_train['ODDS_B'] < dataset_train['ODDS_A']) & (dataset_train['WINNER'] == 0))]
    
    # print('\nResults Classification ({}):'.format(season))
    # results.sort(key=lambda x: x['acc'], reverse=True)
    # [print('{}:\t{:.4f}'.format(x['model'], x['acc'])) for x in results]
    # print('Baseline Last Machups:\t{:.4f}'.format(100*len(matchups_baseline.index)/len(dataset_train.index)))
    # print('Baseline Odds:\t{:.4f}'.format(100*len(odds_baseline.index)/len(dataset_train.index)))
    
    # print('\nResults Regression ({}):'.format(season))
    # results_regression.sort(key=lambda x: x['r2_score'], reverse=True)
    # [print('{}:\t{:.4f}'.format(x['model'], x['r2_score'])) for x in results_regression]
    
    print('\nGetting the feature correlation matrix...')
    
    import seaborn as sns
    
    try:
        dependent_variables = dataset.iloc[:,5:20]
        corrmat = dependent_variables.corr()
        top_corr_features = corrmat.index
        plt.figure(figsize=(13,13))
        title = 'Feature Correlation'
        plt.title(title)
        #plot heat map
        sns.set(font_scale=0.6)
        g=sns.heatmap(dependent_variables.corr(),annot=True,cmap='Blues', fmt='0.1g')
        plt.savefig('charts/{}.png'.format(title.replace(' ','_').lower()), dpi=300)
        plt.show()
    except:
        print('No correlation matrix for the selected model.')
        
    print('\nGetting regression models with the best results...')
    
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.metrics import r2_score
    poly_reg = PolynomialFeatures(degree = 2)
    X_poly = poly_reg.fit_transform(X_regression)
    
    modelCont = 0
    highestAcc = 0
    while True:
        try:
            if results_regression[modelCont]['model'] == 'Polynomial Regression':
                results_regression[modelCont]['pred_regression'] = results_regression[modelCont]['regressor'].predict(X_poly)
            else:
                results_regression[modelCont]['pred_regression'] = results_regression[modelCont]['regressor'].predict(X_regression)
            results_regression[modelCont]['r2_score_test'] = r2_score(y_regression, results_regression[modelCont]['pred_regression'])
            modelCont += 1
        except IndexError:
            break
        
    results_regression.sort(key=lambda x: x['r2_score_test'], reverse=True)
    
    print('\nResults Regression ({}):'.format(season_test))
    results_regression.sort(key=lambda x: x['r2_score_test'], reverse=True)
    [print('{}:\t{:.4f}'.format(x['model'], x['r2_score_test'])) for x in results_regression]
    
    
    print('\nGetting classification model with the best predictions...')
    
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_transformed = sc.fit_transform(X)
    
    modelCont = 0
    highestAcc = 0
    while True:
        try:
            if results[modelCont]['model'] == 'LSTM':
                features_lstm, labels_lstm, info_lstm = parse_lstm_data(X_lstm, y_lstm)
                pred = results[modelCont]['classifier'].predict(features_lstm)
                results[modelCont]['pred'] = pred
                pred_winner = (results[modelCont]['pred'] > 0.5)
                # pred_winner = np.array(pred_winner).astype(np.float32)
                results[modelCont]['pred_winner'] = pred_winner
                results[modelCont]['acc_test'] = accuracy_score(labels_lstm, results[modelCont]['pred_winner'])
            else:
                results[modelCont]['pred'] = results[modelCont]['classifier'].predict(X_transformed)
                if results[modelCont]['model'] == 'ANN':
                    results[modelCont]['pred'] = (results[modelCont]['pred'] > 0.5)
                results[modelCont]['acc_test'] = accuracy_score(y, results[modelCont]['pred'])
            if results[modelCont]['acc_test'] > highestAcc:
                y_pred = results[modelCont]['pred']
                highestAcc = results[modelCont]['acc_test']
                print('Using predictions from {} model: {}'.format(results[modelCont]['model'], results[modelCont]['acc_test']))
            modelCont += 1
        except IndexError:
            break
        
    results.sort(key=lambda x: x['acc_test'], reverse=True)
    
    matchups_baseline = dataset[((dataset['MATCHUP_A'] >= dataset['MATCHUP_B']) & (dataset['WINNER'] == 1)) | 
                       ((dataset['MATCHUP_B'] > dataset['MATCHUP_A']) & (dataset['WINNER'] == 0))]
    odds_baseline = dataset[((dataset['ODDS_A'] <= dataset['ODDS_B']) & (dataset['WINNER'] == 1)) | 
                       ((dataset['ODDS_B'] < dataset['ODDS_A']) & (dataset['WINNER'] == 0))]
    
    print('\nResults Classification ({}):'.format(season_test))
    results.sort(key=lambda x: x['acc_test'], reverse=True)
    [print('{}:\t{:.4f}'.format(x['model'], x['acc_test'])) for x in results]
    print('Baseline Last Machups:\t{:.4f}'.format(100*len(matchups_baseline.index)/len(dataset.index)))
    print('Baseline Odds:\t{:.4f}'.format(100*len(odds_baseline.index)/len(dataset.index)))
    
    print('\nGetting the probabilities of the best model possible...')
    
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
    
    print("\nGetting data from the regular models for visualization...")
    profit = 0
    money_by_date = []
    bets_tracking_matchups = [0]
    bets_tracking_odds = [0]
    money_by_team = dict()
    bets = []
    index_lstm = 0
    pred_winner_lstm = [x['pred_winner'] for x in results if x['model'] == 'LSTM'][0]
    money_by_date.append([dataset.iloc[0,2], dict(zip([x['model'] for x in results], [0 for x in results])),  dict(zip([x['model'] for x in results], [0 for x in results]))])
    for index, game in dataset.iterrows():
        if game['GAME_DATE'] != money_by_date[-1][0]:    
            bets_tracking_matchups.append(bets_tracking_matchups[-1])
            bets_tracking_odds.append(bets_tracking_odds[-1])
            money_by_date.append([game['GAME_DATE'],  dict(zip([x['model'] for x in results], [0 for x in results])), dict(money_by_date[-1][2])])
        
        game_money = 0
        if y_prob[index,0] >= y_prob[index,1]:
            bet_value = 10*y_prob[index,0]
        else:
            bet_value = 10*y_prob[index,1]
        # bet_value = 10
        if (y_pred[index] == 1 and game['ODDS_A'] > threshold) or (y_pred[index] == 0 and game['ODDS_B'] > threshold):
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
        bets_tracking_matchups[-1] += check_game_with_matchups(game, bet_value)
        bets_tracking_odds[-1] += check_game_with_odds(game, bet_value)
        
        for model in money_by_date[-1][1]:
            if model == 'LSTM':
                if index_lstm < len(info_lstm) and info_lstm[index_lstm][0] == money_by_date[-1][0]:
                    prediction = pred_winner_lstm[index_lstm]
                    game_money_model = check_model_performance_on_game_lstm(info_lstm[index_lstm], prediction, bet_value)
                    index_lstm += 1
                else:
                    game_money_model = 0
            else:
                prediction = next(x['pred'][index] for x in results if x['model'] == model)
                game_money_model = check_model_performance_on_game(game, prediction, bet_value)
            money_by_date[-1][1][model] += game_money_model
            money_by_date[-1][2][model] += game_money_model
            
    
    print('\nProfit:', profit)
    
    print('\nPlotting charts...')
            
    models_tracking =  [np.array([x[2][model] for x in money_by_date], dtype=np.float32) for model in money_by_date[-1][1]]
        
    money_by_date = np.array(money_by_date, dtype=str)
    correct_bets = list(filter(lambda x: x[3] == 1, bets))
    missed_bets = list(filter(lambda x: x[3] == 0, bets))
    correct_bets_odds = np.array(list(map(lambda x: x[1], correct_bets)))
    missed_bets_odds = np.array(list(map(lambda x: x[1], missed_bets)))
    # correct_bets_prob = np.array(list(map(lambda x: x[2], correct_bets)))
    # missed_bets_prob = np.array(list(map(lambda x: x[2], missed_bets)))
    correct_bets_home = np.array(list(map(lambda x: x[0], correct_bets)))
    missed_bets_home = np.array(list(map(lambda x: x[0], missed_bets)))
    
    money_by_team = dict(sorted(money_by_team.items(), key=lambda x: x[1]))
    money_by_team_labels = np.array(list(money_by_team.keys()), dtype=str)
    money_by_team_values = np.array(list(money_by_team.values()), dtype=np.float32)
    
    plot_hist('Missed Bets by Odds', 'Odds', 'X Times', missed_bets_odds)
    
    plot_hist('Correct Bets by Odds', 'Odds', 'X Times', correct_bets_odds)
    
    # plot_hist('Correct Bets', 'Probability', 'X Times', correct_bets_prob)
    
    # plot_hist('Missed Bets', 'Probability', 'X Times', missed_bets_prob)
    
    plot_pie_chart('Correct Bets by Home-Away', ['Home', 'Away'], correct_bets_home)
    
    plot_pie_chart('Missed Bets by Home-Away', ['Home', 'Away'], missed_bets_home)
    
    plot_bar('Profit by Team', 'Teams', 'Profit', money_by_team_labels, money_by_team_values)
    
    xpoints = money_by_date[:,0].astype(np.datetime64)
    # ypoints = money_by_date[:,2].astype(np.float32)
    
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=15))
    for model in models_tracking:
        plt.plot(xpoints, model)
    plt.plot(xpoints, bets_tracking_matchups)
    plt.plot(xpoints, bets_tracking_odds)
    
    title = "Profit by Date"
    plt.legend([x['model'] for x in results] + ['Matchups Baseline', 'Odds Baseline'], loc='lower left')
    plt.ylabel("Profit($)")
    plt.xlabel("Date")
    plt.title(title)
    plt.gcf().autofmt_xdate()
    plt.savefig('charts/{}.png'.format(title.replace(' ','_').lower()), dpi=300)
    plt.show()
