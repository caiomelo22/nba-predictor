# NBA Predictor
Project that focus on predicting NBA matches using some machine learning models. I want to stress that this is just a study based on the past seasons and not a financial resource, by all means.

## Description
The purpose of this project is to collect data and develop machine learning models that can predict the outcome of NBA matches. With our predictions made, we apply them to the betting market, simulating what would happen if we used this predictions for financial purposes.

In this project, we cover the following tasks:
  - Data scrapping from the Odds Portal website in order to collect the odds for NBA matches. Once the necessary data is extracted, a csv containing this information is generated so the process of web scrapping is only needed once.
  - Collect data from the NBA API, generating a csv file that represents a window of the given seasons. This csv contains both raw and artificial data made from the api information.
  - Development of a few machine learning models that predict the outcome of NBA matches. The models are trained with the data from the csv created in the last topic.
  - Test of the developed models, displaying the accuracy of each one.
  - Application of the predictions on the betting market, simultating the profit made from each model.
  - Display of how much was invested on each model, the profit and margin of profit.
  - Display of charts that represent the distribution of the missed/hit bets by odd and host.
  - Display of a chart that represents the money made from each team in the league.
  - Display of a chart that represents the progression of money invested over a given window of seasons.
