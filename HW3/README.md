# Homework 3 - Income Prediction

## Packages:
- xgboost
- catboost
- numpy
- pandas
- sklearn

## Usage:
1. 利用trainfile訓練RandomForestClassifier, GradientBoostingClassifier, XGBClassifier, CatBoostClassifier，並對testfile進行預測
    ```bash
    $ python 0750730.py [trainfile] [testfile]
    ```
    - Format:
        - ID: the index 
        - ans: salary >50K is represented as 1 salary <=50K is represented as 0
        ```bash
        ID,ans
        0,0
        1,0
        2,1
        ...
        ```