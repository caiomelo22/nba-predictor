# NBA Predictor
This project focuses on predicting NBA matches using various machine learning models. It's important to note that this is purely a study based on past seasons and is not intended as a financial resource.

## Description
The project is divided in three parts:
- [Classifiation Simulator](https://github.com/caiomelo22/nba-predictor/blob/master/src/classification_simulator.ipynb) --> This notebook was made for you to test your classification machine learning models and simulate how they would perform in previous seasons. In this notebook you can configure settings such as models features, minimum odds, bankroll and much more. After testing and simulating, you can save your pipeline to make current predictions.
- [Classification Predictor](https://github.com/caiomelo22/nba-predictor/blob/master/src/classification_predictor.ipynb) --> This notebook scrapes and predicts today's NBA games using the pipelines generated in the previous notebook.
- [Regression Simulator](https://github.com/caiomelo22/nba-predictor/blob/master/src/regression_simulator.ipynb) --> This notebook was made for you to test your regression machine learning models and simulate how they would perform in previous seasons. After testing and simulating, you can save your pipeline to make current predictions.
- [Regression Predictor](https://github.com/caiomelo22/nba-predictor/blob/master/src/regression_predictor.ipynb) --> This notebook scrapes and predicts today's NBA games using the pipelines generated in the previous notebook.

## Setup
To fetch data from your MySQL database, you need to set the following variables in your environment:
```
HOST=localhost
DATABASE=nba-db
USER=user
PASSWORD=pwd
```

## Disclaimer
Once again, this project was created for research purposes and should not be considered a means to profit from the sports betting market.
