#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 15:42:40 2023

@author: zhexingli
"""

# import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, \
     classification_report, precision_score, \
        recall_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from joblib import parallel_backend
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from mlxtend.plotting import plot_confusion_matrix
import matplotlib.colors as colors

# main folder path
path = '/Users/zhexingli/Desktop/UCR/LLNLDSC/cardiac_challenge-main/'

df_mitbih_train = pd.read_csv(path+'ecg_dataset/mitbih_train.csv', header = None)
df_mitbih_test = pd.read_csv(path+'ecg_dataset/mitbih_test.csv', header = None)

# shuffles training dataset
train = df_mitbih_train.sample(frac=1).reset_index(drop=True)

# grabs the data and labels
train_data = train.iloc[:,:-1]
train_label = train.iloc[:,-1:]

test_data = df_mitbih_test.iloc[:,:-1]
test_label = df_mitbih_test.iloc[:,-1:]

print(f'Shape of training data and label: {train_data.shape}, {train_label.shape}')
print(f'Shape of test data and label: {test_data.shape}, {test_label.shape}')

# scales train and test dataset
scaler = StandardScaler()
scaled_train_data = scaler.fit_transform(train_data)
scaled_test_data = scaler.transform(test_data)

# dimensionality reduction, check number of principal components
pca = PCA()
pca.fit(scaled_train_data)
pca.explained_variance_ratio_
'''
plt.figure(figsize=(10,8))
plt.plot(range(0,20),pca.explained_variance_ratio_.cumsum()[:20], marker='o', linestyle='-')
plt.xlabel('Number of Principal Components')
plt.ylabel('PCA Explained Variance')
'''

# Reduce dimensions with the selected principal components
pca_data = PCA(n_components=12)
pca_data.fit(scaled_train_data)
scores_pca_train = pca_data.transform(scaled_train_data)

# Reduce dimensions for the test set as well
scores_pca_test = pca_data.transform(scaled_test_data)

# Fit KNN with a base model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(scores_pca_train, train_label.values.ravel())

# Check results of the base model
knn_pred = knn.predict(scores_pca_test)
knn_acc = accuracy_score(test_label, knn_pred)
knn_f1 = f1_score(test_label, knn_pred,average='macro')
knn_prec = precision_score(test_label,knn_pred,average='macro')
knn_rec = recall_score(test_label,knn_pred,average='macro')
knn_confusion = confusion_matrix(test_label, knn_pred)

# Cross validation to select the best model
knn_tune = KNeighborsClassifier()
parameters_tune = {'n_neighbors': np.arange(1,10,1),
                  'leaf_size': np.arange(20,40,1),
                  'p': [1,2],
                  'weights': ['uniform','distance'],
                  'metric': ['minkowsiki','chebyshev']}

# Cross validation with grid search
knn_gridsearch = GridSearchCV(estimator=knn_tune, param_grid=parameters_tune, \
                              scoring='accuracy', cv=10, verbose=0, n_jobs=6)

# Fit the training set and predict on the test set with grid search
knn_search = knn_gridsearch.fit(scores_pca_train, train_label.values.ravel())
knn_grid_pred = knn_search.predict(scores_pca_test)


# Parameter setting that gave the best results on the hold out data.
print(knn_gridsearch.best_params_) 
#Mean cross-validated score of the best_estimator
print(f'Best Score - KNN: {knn_gridsearch.best_score_}.')

# Result summary
print('KNN Result summary')
print('********************')
print(f'Baseline Accuracy Score: {knn_acc}.')
print(f'Baseline F1 Score: {knn_f1}.')
print(f'Baseline Precision score: {knn_prec}.')
print(f'Baseline Recall score: {knn_rec}.')

print('********************')
print(f'Accuracy Score: KNN Best Model: {accuracy_score(test_label, knn_grid_pred)}.')
print(f'F1 score of best model: {f1_score(test_label, knn_grid_pred, average="macro")}.')
print(f'Precision score of best model: {precision_score(test_label,knn_grid_pred,average="macro")}.')
print(f'Recall score of best model: {recall_score(test_label,knn_grid_pred,average="macro")}.')
print('Classification report of best model: ')
print(classification_report(test_label, knn_grid_pred))
 

# Get the confusion matrix
cmat = confusion_matrix(test_label, knn_grid_pred)

# PLot the confusion matrix, with diagonals hightlighted, colorbar normalized and
# in logrithmic scale
fig, ax = plot_confusion_matrix(conf_mat=cmat,colorbar=True,show_absolute=True, \
                        show_normed=True,cmap='PuBu',norm_colormap=colors.LogNorm())
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('K-Nearest Neighbors')
plt.savefig(path+'KNN/cmatplot.png', dpi=500)
    

    



