import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.svm import SVC

warnings.filterwarnings("ignore")
print("Setup Complete")

def get_models(random_state, voting_classifier_models=["logistic_regression"]) -> dict:
    models_dict = {
        "naive_bayes": {
            "estimator": GaussianNB(),
            "params": None,
            "score": None,
        },
        "knn": {
            "estimator": KNeighborsClassifier(),
            "params": None,
            "score": None,
        },
        "logistic_regression": {
            "estimator": LogisticRegression(random_state=random_state),
            "params": None,
            "score": None,
        },
        "svm": {
            "estimator": SVC(probability=True, random_state=random_state),
            "params": None,
            "score": None,
        },
        "sgd": {
            "estimator": SGDClassifier(random_state=random_state, loss='log_loss'),
            "params": None,
            "score": None,
        },
        "random_forest_default": {
            "estimator": RandomForestClassifier(random_state=random_state),
            "params": None,
            "score": None,
        },
        "random_forest_500": {
            "estimator": RandomForestClassifier(
                random_state=random_state, n_estimators=500
            ),
            "params": None,
            "score": None,
        },
        "random_forest_750": {
            "estimator": RandomForestClassifier(
                random_state=random_state, n_estimators=750
            ),
            "params": None,
            "score": None,
        },
        "random_forest_1000": {
            "estimator": RandomForestClassifier(
                random_state=random_state, n_estimators=1000
            ),
            "params": None,
            "score": None,
        },
        "gradient_boosting": {
            "estimator": GradientBoostingClassifier(random_state=random_state),
            "params": None,
            "score": None,
        },
        "ada_boost": {
            "estimator": AdaBoostClassifier(random_state=random_state),
            "params": None,
            "score": None,
        },
        "mlp": {
            "estimator": MLPClassifier(random_state=0),
            "params": None,
            "score": None,
        },
    }

    voting_classifier_estimators = []

    for model in models_dict.keys():
        if model in voting_classifier_models:
            voting_classifier_estimators.append(
                (model, models_dict[model]["estimator"])
            )

    if voting_classifier_estimators:
        models_dict["voting_classifier"] = {
            "estimator": VotingClassifier(
                estimators=voting_classifier_estimators, voting="soft"
            )
        }

    return models_dict

def set_numerical_categorical_cols(X: pd.DataFrame):
    # "Cardinality" means the number of unique values in a column
    # Select categorical columns with relatively low cardinality
    categorical_cols = [
        cname
        for cname in X.columns
        if X[cname].dtype == "object" and X[cname].nunique() < 10
    ]

    # Select numerical columns
    numerical_cols = [cname for cname in X.columns if X[cname].dtype == "float64"]

    # Select int columns
    int_cols = [cname for cname in X.columns if X[cname].dtype in ["int64", "int32"]]

    return categorical_cols, numerical_cols, int_cols

def build_pipeline(X_train, y_train, model, preprocess):
    categorical_cols, numerical_cols, int_cols = set_numerical_categorical_cols(X_train)

    # Preprocessing for numerical data
    numerical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", MinMaxScaler()),
        ]
    )

    # Preprocessing for int data
    int_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", MinMaxScaler()),
        ]
    )

    # Preprocessing for categorical data
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse=False)),
        ]
    )

    # Bundle preprocessing for numerical and categorical data
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, numerical_cols),
            ("int", int_transformer, int_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )

    steps = []

    if preprocess:
        steps.append(("preprocessor", preprocessor))

    steps.append(("model", model))

    # Bundle preprocessing and modeling code in a pipeline
    pipeline = Pipeline(
        steps=steps
    )

    # Preprocessing of training data, fit model
    pipeline.fit(X_train, y_train)

    return pipeline

def simulate(
    matches,
    start_season,
    season,
    features,
    betting_starts_after_n_games,
    strategy,
    verbose=1,
    random_state=0,
    preprocess=True,
    voting_classifier_models=["logistic_regression"]
):
    matches_filtered = matches[
        (matches["season"] >= start_season) & (matches["season"] <= season)
    ]
    matches_filtered.dropna(subset=features, inplace=True)

    train_set = matches_filtered[matches_filtered["season"] < season]
    test_set = matches_filtered[matches_filtered["season"] == season]

    # Prepare features and labels
    X_train = train_set[features]
    y_train = train_set["result"]

    X_test = test_set[features]
    _ = test_set["result"]

    models_dict = get_models(random_state,voting_classifier_models)

    if not len(X_train):
        return matches, models_dict

    for model in models_dict.keys():
        my_pipeline = build_pipeline(X_train, y_train, models_dict[model]["estimator"], preprocess)
        if not len(X_test):
            continue

        # Predict on the test set
        y_pred = my_pipeline.predict(X_test)
        y_pred_proba = my_pipeline.predict_proba(X_test)  # Get all probabilities

        # Get the order of classes (e.g., ['H', 'A'])
        class_order = my_pipeline.classes_

        # Map probabilities to correct outcomes based on class order
        home_win_idx = np.where(class_order == "H")[0][0]
        away_win_idx = np.where(class_order == "A")[0][0]

        # Save predictions and probabilities
        matches.loc[X_test.index, f"PredictedRes_{model}"] = y_pred
        matches.loc[X_test.index, f"Proba_HomeWin_{model}"] = y_pred_proba[
            :, home_win_idx
        ]
        matches.loc[X_test.index, f"Proba_AwayWin_{model}"] = y_pred_proba[
            :, away_win_idx
        ]

        models_dict[model]["pipeline"] = my_pipeline

    return matches, models_dict

def get_bet_unit_value(odds, probs, bankroll, strategy, default_value=1, default_bankroll_pct=0.05):
    if strategy == "kelly":
        q = 1 - probs  # Probability of losing
        b = odds - 1  # Net odds received on the bet (including the stake)
        kelly_fraction = ((probs * b - q) / b) * 0.5
        return round(
            min(kelly_fraction, 1.0), 4
        )  * bankroll # Limit the bet to 100% of the bankroll
    elif strategy == "bankroll_pct":
        return default_bankroll_pct * bankroll
    else:
        return default_value

# Function to calculate profit based on prediction
def elo_bet_profit(row, start_season, min_odds, bankroll, strategy, default_value, default_bankroll_pct):
    if row["season"] == start_season:
        return 0
    
    bet_on = None

    if row['home_elo'] > row['away_elo'] and row['home_odds'] > min_odds:
        bet_on = 'H'
        odds = row['home_odds']
    elif row['away_odds'] > min_odds:
        bet_on = 'A'
        odds = row['away_odds']
    
    if bet_on == None:
        return 0

    bet_value = get_bet_unit_value(odds, 1, bankroll, strategy, default_value, default_bankroll_pct)
    
    if row["result"] == bet_on:
        profit_elo = (odds*bet_value) - bet_value  # Profit from winning bet
    else:
        profit_elo = - bet_value  # Loss from losing bet

    return profit_elo

def home_bet_profit(row, start_season, min_odds, bankroll, strategy, default_value, default_bankroll_pct):
    if row["season"] == start_season:
        return 0

    bet_value = get_bet_unit_value(row['home_odds'], 1, bankroll, strategy, default_value, default_bankroll_pct)
    
    if row['home_odds'] < min_odds:
        return 0
    elif row["result"] == "H":
        return (row['home_odds'] * bet_value) - bet_value  # Profit from winning bet
    else:
        return - bet_value

def bet_worth_it(prediction, odds, pred_odds, min_odds, bet_value):
    if (
        bet_value < 0 # Value not worth it
        or prediction == None # No prediction
        or pd.isna(prediction) # No prediction
        or odds < min_odds
        or odds < pred_odds
    ):
        return False

    return True

def bet_profit_ml(row, model, min_odds, bankroll, strategy="kelly", default_value=1, default_bankroll_pct=0.05):
    selected_odds = None
    selected_pred_odds = None
    if row[f"PredictedRes_{model}"] == 'H':
        selected_odds = row['home_odds']
        selected_pred_odds = row[f"Proba_HomeWin_{model}"]
    elif row[f"PredictedRes_{model}"] == 'A':
        selected_odds = row['away_odds']
        selected_pred_odds = row[f"Proba_AwayWin_{model}"]

    bet_value = get_bet_unit_value(selected_odds, selected_pred_odds, bankroll, strategy, default_value, default_bankroll_pct)

    if not bet_worth_it(
        row[f"PredictedRes_{model}"],
        selected_odds,
        selected_pred_odds,
        min_odds,
        bet_value,
    ):
        return 0

    if row["result"] == row[f"PredictedRes_{model}"]:
        if row[f"PredictedRes_{model}"] == 'H':
            profit_ml = bet_value*row['home_odds'] - bet_value
        elif row[f"PredictedRes_{model}"] == 'A':
            profit_ml = bet_value*row['away_odds'] - bet_value
    else:
        profit_ml = -bet_value

    return profit_ml

def get_simulation_results(matches, start_season, min_odds, plot_threshold, random_state, bankroll, strategy, default_value, default_bankroll_pct):
    # Calculate profits for each model
    matches[f'ProfitElo'] = matches.apply(lambda row: elo_bet_profit(row, start_season, min_odds, bankroll, strategy, default_value, default_bankroll_pct), axis=1)
    matches[f'CumulativeProfitElo'] = matches[f'ProfitElo'].cumsum()

    matches['ProfitHome'] = matches.apply(lambda row: home_bet_profit(row, start_season, min_odds, bankroll, strategy, default_value, default_bankroll_pct), axis=1)
    matches['CumulativeProfitHome'] = matches['ProfitHome'].cumsum()

    # Plot cumulative profit
    plt.figure(figsize=(12, 8))

    if matches[f'CumulativeProfitElo'].iloc[-1] > plot_threshold:
        plt.plot(matches["date"], matches[f'CumulativeProfitElo'], label=f'Cumulative Profit Elo')

    if matches[f'CumulativeProfitHome'].iloc[-1] > plot_threshold:
        plt.plot(matches["date"], matches[f'CumulativeProfitHome'], label=f'Cumulative Profit Home')

    model_names = get_models(random_state).keys()

    for model_name in model_names:
        matches[f'ProfitML_{model_name}'] = matches.apply(lambda row: bet_profit_ml(row, model_name, min_odds, bankroll, strategy, default_value, default_bankroll_pct), axis=1)
        matches[f'CumulativeProfitML_{model_name}'] = matches[f'ProfitML_{model_name}'].cumsum()

        if matches[f'CumulativeProfitML_{model_name}'].iloc[-1] > plot_threshold:
            plt.plot(matches["date"], matches[f'CumulativeProfitML_{model_name}'], label=f'Cumulative Profit ML {model_name}')

    plt.title(f'Cumulative Profit from Betting over {min_odds}')
    plt.xlabel('Game')
    plt.ylabel('Cumulative Profit')
    plt.legend()
    plt.grid(True)
    plt.show()

    # display(matches)

    cum_profit_home = round(matches.iloc[-1]['CumulativeProfitHome'], 4)
    len_home = len(matches[matches['ProfitHome'] != 0])

    print(f"Home method ({cum_profit_home}/{len_home}):", round(cum_profit_home / len_home, 4))

    best_model_name = None
    best_model_profit = -1000

    for model_name in model_names:
        cum_profit_ml = round(matches.iloc[-1][f'CumulativeProfitML_{model_name}'], 4)
        len_ml = len(matches[matches[f"ProfitML_{model_name}"] != 0])

        if cum_profit_ml > best_model_profit:
            best_model_name = model_name
            best_model_profit = cum_profit_ml

        print(f"ML method with {model_name.ljust(20)} --> ({str(cum_profit_ml).rjust(7)}/{str(len_ml).rjust(3)}):", 
        round(cum_profit_ml / len_ml, 4))

    # Evaluate best model
    best_models_predicted_matches = matches[matches[f"PredictedRes_{best_model_name}"].notna()]
    y_pred = best_models_predicted_matches[f"PredictedRes_{best_model_name}"]
    y_test = best_models_predicted_matches["result"]

    print(f"\nProfit for {best_model_name}: ${round(matches.iloc[-1][f'CumulativeProfitML_{best_model_name}'], 4)}")
    print(f"Accuracy for {best_model_name}: {accuracy_score(y_test, y_pred):.2f}")
    print(f"Classification Report for {best_model_name}:")
    print(classification_report(y_test, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred, labels=['H', 'A'])
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=['H', 'A'], yticklabels=['H', 'A'])
    plt.title(f"Confusion Matrix for {best_model_name}")
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

    return best_model_name