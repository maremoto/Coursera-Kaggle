# Coursera-Kaggle final project
Course "How to win a datascience competition: Learn from top Kagglers" final project.

Hello, this is my final project submission with an unexpected outcome... it is not much complex, unfortunately I had not a lot of time to spend on it.

## Usage:

Firstly read the brief documentation to have a comprehensive overview (***0_FinalProjectDocumentation.pdf***).

Then to see (and reproduce) the evolution of the project work, just follow the python notebooks in the order of their names.

**NOTE:** Create a ./data folder and copy the competition data inside it, create a ./features and ./submissions empty folders too.

## Spoiler:

* The features generated are lags from the monthly sold item counts and revenues (price * items), and an expanding mean encoding of item category.

* The initial solution was a staking of models (NeuralNet, Linear ElasticNet and LightGBM) with another Linear Regressor or Decission Tree or Linear Convex Mix, but at the end, a LGB model alone performs better! 

## Environment:

There is a requirements.txt file, but in fact the main used tools are:
```
numpy 1.17.0
pandas 0.25.0
sklearn 0.21.3
scipy 1.3.0
torch 1.1.0
lightgbm 2.0.6
```
