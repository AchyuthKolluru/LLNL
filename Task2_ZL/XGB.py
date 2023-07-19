#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 09:39:46 2023

@author: zhexingli
"""

# import packages
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score

# Load data
path = '/Users/zhexingli/Desktop/UCR/LLNLDSC/cardiac_challenge-main/'
df_mitbih_train = pd.read_csv(path+'ecg_dataset/mitbih_train.csv', header = None)
df_mitbih_test = pd.read_csv(path+'ecg_dataset/mitbih_test.csv', header = None)

# Split data into X and y
X_train = df_mitbih_train.iloc[:,:-1]
Y_train = df_mitbih_train.iloc[:,-1]
X_test = df_mitbih_test.iloc[:,:-1]
Y_test = df_mitbih_test.iloc[:,-1]

'''
# scales train and test dataset
scaler = StandardScaler()
scaled_train_data = scaler.fit_transform(X_train)
scaled_test_data = scaler.transform(test_data)
'''
'''
xgb_base = xgb.XGBClassifier()
xgb_base.fit(X_train, Y_train)

# Define parameter grid
param_grid = {
    'max_depth': [3, 5, 7, 10],
    'learning_rate': [0.01, 0.1, 0.2, 0.3],
    'n_estimators': [100, 200, 300, 400, 500],
    'gamma': [0, 0.1, 0.2]
}

# Create another XGB Classifier object
xgb_tune = xgb.XGBClassifier(use_label_encoder=False)

# Create GridSearchCV object
gcv = GridSearchCV(xgb_tune, param_grid, cv=5)

# Fit GridSearchCV object to data
gcv.fit(X_train, Y_train, eval_metric='logloss')

# Get the best parameters
print("Best parameters found: ",gcv.best_params_)

# Use the best model to make predictions
predictions = gcv.predict(X_test)

# Get the accuracy of the best model
accuracy = accuracy_score(Y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
'''