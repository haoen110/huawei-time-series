#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")


# In[2]:


data_path = 'C:/Users/CRH/Desktop/cityu/huawei/Time Series Forecasting/Small Sample Time Series Forecasting/Time_Series'
fnames = [fname for fname in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, fname))]
print(len(fnames), "files")


# In[3]:


def process_diagonal_with_last_1_m0():
    data_full = []
    for fname in fnames:
        data = pd.read_csv(os.path.join(data_path, fname), index_col=0)
        data = data.drop([' item'], axis=1)
        data.columns = ['qty', 'qty1', 'qty2', 'qty3']
        data['x1_for_m0'] = data['qty1']
        data['x2_for_m0'] = data['qty2'].shift(1, axis=0)
        data['x3_for_m0'] = data['qty3'].shift(2, axis=0)
        data['x1_for_m1'] = data['qty2']
        data['x2_for_m1'] = data['qty3'].shift(2, axis=0)
        data['x1_for_m2'] = data['qty3']
        data['Last_M0'] = data['qty'].shift(1, axis=0)
        data['M0'] = data['qty']
        data['M1'] = data['qty'].shift(-1, axis=0)
        data['M2'] = data['qty'].shift(-2, axis=0)
        data['M3'] = data['qty'].shift(-3, axis=0)
        data = data.drop(['qty', 'qty1', 'qty2', 'qty3'], axis=1)
        data = data.dropna(how='any', axis=0)

        temp = np.zeros((42, 11))  # this should be modified
        for i in range(1215, 1257):
            try:
                temp[i-1215] = data.loc[i].values
            except:
                continue
        data_full.append(temp)
    return data_full


# In[4]:


# data_full = process_original_with_last_1_m0()
data_full = process_diagonal_with_last_1_m0()


# In[5]:


def split_m0(past_days):
#     split_ind = 38  # if original
    split_ind = 36  # if digonal
    X_train, Y_train = [], []
    X_test, Y_test = [], []

    for ind in range(past_days, split_ind):
        for item_ind in range(len(fnames)):
            X_train.append(data_full[item_ind][ind - past_days: ind + 1][:,:-4])
            Y_train.append(data_full[item_ind][ind][-4]) # change the number of y

    for ind in range(split_ind, split_ind + 6):
        for item_ind in range(len(fnames)):
            X_test.append(data_full[item_ind][ind - past_days: ind + 1][:,:-4])
            Y_test.append(data_full[item_ind][ind][-4]) # change the number of y
    X_train, Y_train = np.array(X_train), np.array(Y_train)
    X_test, Y_test = np.array(X_test), np.array(Y_test)
    if past_days == 0:
        X_train, Y_train, X_test, Y_test =         np.squeeze(X_train), np.squeeze(Y_train), np.squeeze(X_test), np.squeeze(Y_test) 
    print("The shape of X_train:", X_train.shape)
    print("The shape of Y_train:", Y_train.shape)
    print("The shape of X_test:", X_test.shape)
    print("The shape of Y_train:", Y_test.shape)
    return X_train, Y_train, X_test, Y_test

X_train, Y_train, X_test, Y_test = split_m0(0) 
Y_test=Y_test.reshape(-1,1)


# In[8]:


def acc_out(item_index):
    item_ytest = []
    item_ypred = []
    count = 0
    for i in range(6):
        item_ytest.append(Y_test[item_index+count*1146])
        item_ypred.append(Y_pred_test[item_index+count*1146])
        count += 1
    acc = []
    for i in range(6):
        for index, value in enumerate(item_ypred[i]): 
            if  max(item_ytest[i][index], item_ypred[i][index]) == 0:
                acc.append(np.nan)
                continue
            acc.append(np.divide(min(item_ytest[i][index], item_ypred[i][index]),
                                 max(item_ytest[i][index], item_ypred[i][index])))
    acc = np.array(acc)
    acc = acc.reshape(6, 1)
    return acc


# # Prediction of M0

# In[19]:


import statsmodels.api as sm

X_train=sm.add_constant(X_train)
X_test=sm.add_constant(X_test)
model=sm.OLS(Y_train,X_train).fit()
Y_pred_test = model.predict(X_test)
Y_pred_test=Y_pred_test.reshape(-1,1)

acc_list = []
for i in range(1146):
    acc_list.append(acc_out(i))
acc_list = np.array(acc_list)
print(np.nanmean(acc_list,axis=0))

print_model = model.summary()
print(print_model)


# # Prediction of M1

# In[22]:


def split_m1(past_days):
#     split_ind = 38  # if original
    split_ind = 36  # if digonal
    X_train, Y_train = [], []
    X_test, Y_test = [], []

    for ind in range(past_days, split_ind):
        for item_ind in range(len(fnames)):
            X_train.append(data_full[item_ind][ind - past_days: ind + 1][:,:-4])
            Y_train.append(data_full[item_ind][ind][-3]) # change the number of y

    for ind in range(split_ind, split_ind + 6):
        for item_ind in range(len(fnames)):
            X_test.append(data_full[item_ind][ind - past_days: ind + 1][:,:-4])
            Y_test.append(data_full[item_ind][ind][-3]) # change the number of y
    X_train, Y_train = np.array(X_train), np.array(Y_train)
    X_test, Y_test = np.array(X_test), np.array(Y_test)
    if past_days == 0:
        X_train, Y_train, X_test, Y_test =         np.squeeze(X_train), np.squeeze(Y_train), np.squeeze(X_test), np.squeeze(Y_test) 
    print("The shape of X_train:", X_train.shape)
    print("The shape of Y_train:", Y_train.shape)
    print("The shape of X_test:", X_test.shape)
    print("The shape of Y_train:", Y_test.shape)
    return X_train, Y_train, X_test, Y_test

X_train, Y_train, X_test, Y_test = split_m1(0) 
Y_test=Y_test.reshape(-1,1)


# In[23]:


X_train=sm.add_constant(X_train)
X_test=sm.add_constant(X_test)
model=sm.OLS(Y_train,X_train).fit()
Y_pred_test = model.predict(X_test)
Y_pred_test=Y_pred_test.reshape(-1,1)

acc_list = []
for i in range(1146):
    acc_list.append(acc_out(i))
acc_list = np.array(acc_list)
print(np.nanmean(acc_list,axis=0))

print_model = model.summary()
print(print_model)


# # Prediction of M2

# In[24]:


def split_m2(past_days):
#     split_ind = 38  # if original
    split_ind = 36  # if digonal
    X_train, Y_train = [], []
    X_test, Y_test = [], []

    for ind in range(past_days, split_ind):
        for item_ind in range(len(fnames)):
            X_train.append(data_full[item_ind][ind - past_days: ind + 1][:,:-4])
            Y_train.append(data_full[item_ind][ind][-2]) # change the number of y

    for ind in range(split_ind, split_ind + 6):
        for item_ind in range(len(fnames)):
            X_test.append(data_full[item_ind][ind - past_days: ind + 1][:,:-4])
            Y_test.append(data_full[item_ind][ind][-2]) # change the number of y
    X_train, Y_train = np.array(X_train), np.array(Y_train)
    X_test, Y_test = np.array(X_test), np.array(Y_test)
    if past_days == 0:
        X_train, Y_train, X_test, Y_test =         np.squeeze(X_train), np.squeeze(Y_train), np.squeeze(X_test), np.squeeze(Y_test) 
    print("The shape of X_train:", X_train.shape)
    print("The shape of Y_train:", Y_train.shape)
    print("The shape of X_test:", X_test.shape)
    print("The shape of Y_train:", Y_test.shape)
    return X_train, Y_train, X_test, Y_test

X_train, Y_train, X_test, Y_test = split_m2(0) 
Y_test=Y_test.reshape(-1,1)


# In[25]:


X_train=sm.add_constant(X_train)
X_test=sm.add_constant(X_test)
model=sm.OLS(Y_train,X_train).fit()
Y_pred_test = model.predict(X_test)
Y_pred_test=Y_pred_test.reshape(-1,1)

acc_list = []
for i in range(1146):
    acc_list.append(acc_out(i))
acc_list = np.array(acc_list)
print(np.nanmean(acc_list,axis=0))

print_model = model.summary()
print(print_model)


# # Prediction of M3

# In[26]:


def split_m3(past_days):
#     split_ind = 38  # if original
    split_ind = 36  # if digonal
    X_train, Y_train = [], []
    X_test, Y_test = [], []

    for ind in range(past_days, split_ind):
        for item_ind in range(len(fnames)):
            X_train.append(data_full[item_ind][ind - past_days: ind + 1][:,:-4])
            Y_train.append(data_full[item_ind][ind][-1]) # change the number of y

    for ind in range(split_ind, split_ind + 6):
        for item_ind in range(len(fnames)):
            X_test.append(data_full[item_ind][ind - past_days: ind + 1][:,:-4])
            Y_test.append(data_full[item_ind][ind][-1]) # change the number of y
    X_train, Y_train = np.array(X_train), np.array(Y_train)
    X_test, Y_test = np.array(X_test), np.array(Y_test)
    if past_days == 0:
        X_train, Y_train, X_test, Y_test =         np.squeeze(X_train), np.squeeze(Y_train), np.squeeze(X_test), np.squeeze(Y_test) 
    print("The shape of X_train:", X_train.shape)
    print("The shape of Y_train:", Y_train.shape)
    print("The shape of X_test:", X_test.shape)
    print("The shape of Y_train:", Y_test.shape)
    return X_train, Y_train, X_test, Y_test

X_train, Y_train, X_test, Y_test = split_m3(0) 
Y_test=Y_test.reshape(-1,1)


# In[27]:


X_train=sm.add_constant(X_train)
X_test=sm.add_constant(X_test)
model=sm.OLS(Y_train,X_train).fit()
Y_pred_test = model.predict(X_test)
Y_pred_test=Y_pred_test.reshape(-1,1)

acc_list = []
for i in range(1146):
    acc_list.append(acc_out(i))
acc_list = np.array(acc_list)
print(np.nanmean(acc_list,axis=0))

print_model = model.summary()
print(print_model)


# In[ ]:




