# Coursera-Kaggle final project
Course "How to win a datascience competition: Learn from top Kagglers" final project.

## Usage:

Firstly read the brief documentation to have a comprehensive overview (***0_FinalProjectDocumentation.pdf***).

Then to see (and reproduce) the evolution of the project work, just follow the python notebooks in the order of their names.

**NOTE:** Create a ./data folder and copy the competition data inside it, and create a ./features and ./submissions empty folders too.

## Spoiler:

* The features generated are lags from the monthly sold item counts and revenues (price * items), and an expanding mean encoding of item category.

* The solution is a staking of models, Linear ElasticNet and LightGBM as 1st level with a Linear Convex Mix as 2nd level, but at the end, a LGB model alone performs almos the same! 

## Environment:

There is available a requirements.txt file, but in fact the main used tools are:
```
numpy 1.17.0
pandas 0.25.0
sklearn 0.21.3
scipy 1.3.0
torch 1.1.0
lightgbm 2.0.6
```
