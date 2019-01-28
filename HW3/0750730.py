# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 22:14:37 2018

@author: JamesChiou
"""
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier


def onehot_encoding(data):
    workclass_onehot = pd.get_dummies(data['workclass'], prefix='workclass')
    education_onehot = pd.get_dummies(data['education'], prefix='education')
    status_onehot = pd.get_dummies(data['marital-status'],
                                   prefix='marital-status')
    occupation_onehot = pd.get_dummies(data['occupation'], prefix='occupation')
    relationship_onehot = pd.get_dummies(data['relationship'],
                                         prefix='relationship')
    race_onehot = pd.get_dummies(data['race'], prefix='race')
    data['sex'] = data['sex'].map({' Male': 0, ' Female': 1})
    country_onehot = pd.get_dummies(data['native-country'],
                                    prefix='native-country')

    data = data.drop(['workclass', 'education', 'marital-status', 'occupation',
                      'relationship', 'race', 'native-country'], 1)
    data = pd.concat([data, workclass_onehot, education_onehot, status_onehot,
                      occupation_onehot, relationship_onehot, race_onehot,
                      country_onehot], axis=1)
    return data


# Read raw file
if(len(sys.argv) > 1):
    trainfile = sys.argv[1]
    testfile = sys.argv[2]
else:
    # use default setting
    trainfile = 'data/train.csv'
    testfile = 'data/test.csv'
train = pd.read_csv(trainfile, header=None)
test = pd.read_csv(testfile, header=None)
y_all_train = pd.DataFrame(train[14]).values.ravel()
x_all_train = train.drop([14], axis=1)

# Combine train & test data for data preprocessing
data = pd.concat((x_all_train, test))
data.rename(columns={0: 'age', 1: 'workclass', 2: 'fnlwgt', 3: 'education',
                     4: 'education-num', 5: 'marital-status', 6: 'occupation',
                     7: 'relationship', 8: 'race', 9: 'sex',
                     10: 'capital-gain', 11: 'capital-loss',
                     12: 'hours-per-week', 13: 'native-country'}, inplace=True)

# Data preprocessing
# Drop weird feature
data.drop(['fnlwgt'], axis=1, inplace=True)

# Do one-hot encoding
data = onehot_encoding(data)

# Split train and test data
x_train = data[0:31654]
x_test = data[31654:]

# RandomForestClassifier
rfc = RandomForestClassifier(max_depth=150, max_features=15,
                             min_samples_leaf=1, min_samples_split=50,
                             n_estimators=120, oob_score=True)

# GradientBoostingClassifier
gbc = GradientBoostingClassifier(learning_rate=0.21, max_depth=4,
                                 max_features=41, min_samples_leaf=10,
                                 min_samples_split=15, n_estimators=204,
                                 subsample=0.9)

# XGBClassifier
xgbc = XGBClassifier(learning_rate=0.32, min_child_weight=0.45, max_depth=4,
                     gamma=0.45, subsample=0.95, colsample_bytree=0.9,
                     colsample_bylevel=0.45, n_estimators=250)

# CATClassifier
cbc = CatBoostClassifier(learning_rate=0.1, verbose=0)


# Unsupervised learning
def semi_label(model_list, iteration=10, threshold=0.99):
    for model in model_list:
        print(model)

    ypredprob = 0
    for model in model_list:
        model.fit(x_train, y_all_train)
        ypredprob += model.predict_proba(x_test)
    ypredprob = ypredprob / len(model_list)

    old_x_train = x_train.copy()
    old_y_train = y_all_train.copy()
    temp_y_train = pd.DataFrame()

    for i in range(0, iteration):
        index_above_threshold = np.where(ypredprob > threshold+0.0001*i)
        new_x_train = x_test.iloc[index_above_threshold[0]]
        new_y_train = index_above_threshold[1]
        if (index_above_threshold[0].size - temp_y_train.size) <= 0:
            break

        print('Iter: ' + str(i) + ' New data number: ' +
              str(index_above_threshold[0].size - temp_y_train.size))

        temp_y_train = new_y_train.copy()

        new_x_train = pd.concat((old_x_train, new_x_train))
        new_y_train = np.concatenate((old_y_train, new_y_train), axis=0)

        ypredprob = 0
        for model in model_list:
            model.fit(new_x_train, new_y_train)
            ypredprob += model.predict_proba(x_test)
        ypredprob = ypredprob / len(model_list)


semi_label([rfc, gbc, xgbc, cbc])

# Hard voting
ypred_1 = rfc.predict(x_test)
ypred_2 = cbc.predict(x_test)
ypred_3 = gbc.predict(x_test)
ypred_4 = xgbc.predict(x_test)
ypred_all = (0.2*ypred_1+0.2*ypred_2+0.2*ypred_3+0.4*ypred_4)
ypred = (ypred_all[:] > 0.5).astype(np.int)

# Save submission file
sub = pd.DataFrame(ypred, columns=['ans'])
sub.index.name = 'ID'
sub.to_csv('answer.csv', index=True)
