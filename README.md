# NBA Predictor
This project focuses on predicting NBA matches using various machine learning models. It's important to note that this is purely a study based on past seasons and is not intended as a financial resource.

## Description
The project is divided in three parts:
- Generating a CSV file with statistics from previous games/seasons. This is done in [data.ipynb](https://github.com/caiomelo22/nba-predictor/blob/master/src/data.ipynb) by fetching data from your database and aggregating information for use in machine learning models. If you'd like to explore the project that gathers the data used in this one, check out [NBA Data](https://github.com/caiomelo22/nba-data).
- Testing models and simulating how the predictions made would perform against the moneyline market odds. This is done in [simulator.ipynb](https://github.com/caiomelo22/nba-predictor/blob/master/src/simulator.ipynb) and utilizes data generated in the previous notebook. In this notebook, you can also save the best-performing models for use in our final notebook.
- The final notebook scrapes and predicts today's NBA games using the models generated in the previous notebook. This is done in [predictor.ipynb](https://github.com/caiomelo22/nba-predictor/blob/master/src/predictor.ipynb).

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
