#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 16 17:19:32 2023

@author: zhexingli
"""
%matplotlib auto

# import packages
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, \
     classification_report, precision_score, recall_score, roc_auc_score
from tensorflow.keras import regularizers
import numpy as np

# main folder path
path = '/Users/zhexingli/Desktop/UCR/LLNLDSC/cardiac_challenge-main/'

df_mitbih_train = pd.read_csv(path+'ecg_dataset/mitbih_train.csv', header = None)
df_mitbih_test = pd.read_csv(path+'ecg_dataset/mitbih_test.csv', header = None)

# grabs the data and labels
train_data = df_mitbih_train.iloc[:,:-1].values
train_label = df_mitbih_train.iloc[:,-1:].values

test_data = df_mitbih_test.iloc[:,:-1].values
test_label = df_mitbih_test.iloc[:,-1:].values

# reshape data to be compatible with Conv1D input shape
train_data = train_data.reshape(-1, 187, 1)
test_data = test_data.reshape(-1, 187, 1)

# Define 10-fold cross validation
kfold = KFold(n_splits=10, shuffle=True, random_state=1)

# Store accuracy and loss on cv data from each fold
accuracy, losses = [], []

# Store histories
hist = []

# Store models
models = []

for train_i, test_i in kfold.split(train_data, train_label):
    # build a CNN model
    model = tf.keras.models.Sequential()
    num_kernels = [16,32,64,64,128,128]

    for i in range(len(num_kernels)):
        if i == 0:
            model.add(layers.Conv1D(num_kernels[i],3,padding='same',activation='relu', input_shape=(187, 1)))
        else:
            model.add(layers.Conv1D(num_kernels[i],3,padding='same',activation='relu'))
        
        model.add(layers.BatchNormalization())
    
        # Add a max pooling layer after every 2 convolutions
        if (i+1) % 2 == 0:
            model.add(layers.MaxPooling1D(2))
            model.add(layers.Dropout(0.5))  # Add dropout here
        
    model.add(layers.Flatten())
    model.add(layers.Dense(100,activation='relu',kernel_regularizer=regularizers.l2(0.01)))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(5))
    
    model.compile(optimizer='Adam',loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),\
                  metrics=['accuracy'])
        
    # early stopping based on cv loss
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5,\
                                                      verbose=1)
    
    history = model.fit(train_data, train_label, epochs=10, batch_size=64,\
                        validation_data=(train_data[test_i], train_label[test_i]),\
                        callbacks=[early_stopping])
    
    hist.append(history.history)
    
    # evaluate the model
    loss, acc = model.evaluate(train_data[test_i], train_label[test_i], verbose=0)
    accuracy.append(acc)
    losses.append(loss)
    models.append(model)
    
# print cross-validation score
print(f'10-fold cv accuracy: {np.mean(accuracy)} "+/-" {np.std(accuracy)}.')
print(f'10-fold cv loss: {np.mean(loss)} "+/-" {np.std(loss)}.')

# grab the best model
best_model_index = np.argmin(losses)
best_model = models[best_model_index]

# evaluate the test data
model.evaluate(test_data,test_label,verbose=2)
test_logits = model.predict(test_data)
test_prob = tf.nn.softmax(test_logits)
cnn_pred = tf.argmax(test_prob, axis=1).numpy()

print(f'Accuracy score of CNN: {accuracy_score(test_label, cnn_pred)}.')
print(f'F1 score of CNN: {f1_score(test_label, cnn_pred, average="macro")}.')
print(f'Precision score of CNN: {precision_score(test_label, cnn_pred,)}.')
print(f'Recall score of CNN: {recall_score(test_label, cnn_pred, average="macro")}.')
print('Classification report of best model: ')
print(classification_report(test_label, cnn_pred))


cmat = confusion_matrix(test_label, cnn_pred)

# PLot the confusion matrix, with diagonals hightlighted, colorbar normalized and
# in logrithmic scale
fig, ax = plot_confusion_matrix(conf_mat=cmat,colorbar=True,show_absolute=True, \
                        show_normed=True,cmap='PuBu',norm_colormap=colors.LogNorm())
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('CNN')
plt.savefig(path+'CNN/cmatplot.png', dpi=500)





