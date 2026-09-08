# Rossmann Store Sales: Exploratory Analysis and Prediction

A Fall 2021 CSE 519 course project using the [Rossmann Store Sales](https://www.kaggle.com/c/rossmann-store-sales) competition to practice data integration, exploratory analysis, hypothesis testing, tree-based regression, and Kaggle submission workflows.

## Project overview

The notebook merges store metadata with historical sales and examines patterns involving store identity, promotions, holidays, competition distance, and other store attributes. It then compares decision tree and random forest regressors using root mean square percentage error (RMSPE).

The reported validation RMSPE values were approximately 0.203 for the decision tree and 0.202 for the random forest. The associated Kaggle submissions scored about 0.724 and 0.706 on the public and private leaderboards, respectively.

The project is a historical learning exercise. The analyses are exploratory, and the reported leaderboard performance should not be treated as a current production benchmark.

## Repository contents

- `analysis.ipynb` — primary Jupyter notebook
- `Kaggle_Challenge.png` — leaderboard screenshot

The competition's train, test, and store data are not included.

## Author

Kai Li.
