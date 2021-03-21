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


def process_diagonal(past_m0=0):
    data_full = []
    if past_m0 <= 2:
        row = 42
        start_period = 1215
    else:
        row = 42 - past_m0
        start_period = 1215 + (past_m0-2)
    col = 10 + past_m0
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
        for i in range(past_m0):
            name = 'Last_M0' + str(i+1)
            data[name] = data['qty'].shift(i+1, axis=0)
        data['M0'] = data['qty']
        data['M1'] = data['qty'].shift(-1, axis=0)
        data['M2'] = data['qty'].shift(-2, axis=0)
        data['M3'] = data['qty'].shift(-3, axis=0)
        data = data.drop(['qty', 'qty1', 'qty2', 'qty3'], axis=1)
        data = data.dropna(how='any', axis=0)

        temp = np.zeros((row, col))  # this should be modified
        for i in range(start_period, 1257):
            try:
                temp[i-start_period] = data.loc[i].values
            except:
                continue
        data_full.append(temp)
    return data_full


# In[5]:


# data_full = process_original_with_last_1_m0()
data_full = process_diagonal(1)


# In[6]:


def split_m0(past_days):
#     split_ind = 38  # if original
    split_ind = 36  # if digonal
    X_train, Y_train = [], []
    X_test, Y_test = [], []

    for ind in range(past_days, split_ind):
        for item_ind in range(len(fnames)):
            X_train.append(data_full[item_ind][ind - past_days: ind + 1][:,[0,1,2,6]])
            Y_train.append(data_full[item_ind][ind][-4]) # change the number of y

    for ind in range(split_ind, split_ind + 6):
        for item_ind in range(len(fnames)):
            X_test.append(data_full[item_ind][ind - past_days: ind + 1][:,[0,1,2,6]])
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


# In[7]:


import sklearn.preprocessing
from sklearn.linear_model import Lars
from sklearn.linear_model import Lars,Lasso,Ridge,RidgeCV,LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import PolynomialFeatures


# In[49]:


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


# In[8]:


def my_acc(Y_pred_test, Y_test):
    acc = []
    for index, value in enumerate(Y_pred_test):
        if max(Y_test[index], Y_pred_test[index]) == 0:
            acc.append(np.nan)
            continue
        acc.append((min(Y_test[index], Y_pred_test[index])/max(Y_test[index], Y_pred_test[index]))[0])
    return np.nanmean(acc)


# # Prediction of M0

# In[9]:


import statsmodels.api as sm

X_train=sm.add_constant(X_train)
X_test=sm.add_constant(X_test)
model=sm.OLS(Y_train,X_train).fit()
Y_pred_test = model.predict(X_test)
Y_pred_test=Y_pred_test.reshape(-1,1)
Y_pred_test = model.predict(X_test)
acc = my_acc(Y_pred_test, Y_test)

print_model = model.summary()
print('ACC:', acc)
print(print_model)


# In[10]:


Y_pred_test_m0=Y_pred_test
Y_pred_m0 = model.predict(X_train)
Y_pred_m0=Y_pred_m0.reshape(-1,1)


# In[11]:


Y_pred_m0.shape


# # Prediction of M1

# Prediction of M0 will be used to predict M1.

# In[12]:


def split_m1(past_days):
#     split_ind = 38  # if original
    split_ind = 36  # if digonal
    X_train, Y_train = [], []
    X_test, Y_test = [], []

    for ind in range(past_days, split_ind):
        for item_ind in range(len(fnames)):
            X_train.append(data_full[item_ind][ind - past_days: ind + 1][:,[3,4,6]])
            Y_train.append(data_full[item_ind][ind][-3]) # change the number of y

    for ind in range(split_ind, split_ind + 6):
        for item_ind in range(len(fnames)):
            X_test.append(data_full[item_ind][ind - past_days: ind + 1][:,[3,4,6]])
            Y_test.append(data_full[item_ind][ind][-3]) # change the number of y
    X_train, Y_train = np.array(X_train), np.array(Y_train)
    X_test, Y_test = np.array(X_test), np.array(Y_test)
    if past_days == 0:
        X_train, Y_train, X_test, Y_test =         np.squeeze(X_train), np.squeeze(Y_train), np.squeeze(X_test), np.squeeze(Y_test) 
    X_train=np.c_[X_train,Y_pred_m0]
    X_test=np.c_[X_test,Y_pred_test_m0]
    print("The shape of X_train:", X_train.shape)
    print("The shape of Y_train:", Y_train.shape)
    print("The shape of X_test:", X_test.shape)
    print("The shape of Y_train:", Y_test.shape)
    return X_train, Y_train, X_test, Y_test

X_train, Y_train, X_test, Y_test = split_m1(0) 
Y_test=Y_test.reshape(-1,1)


# In[13]:


X_train=sm.add_constant(X_train)
X_test=sm.add_constant(X_test)
model=sm.OLS(Y_train,X_train).fit()
Y_pred_test = model.predict(X_test)
Y_pred_test=Y_pred_test.reshape(-1,1)
Y_pred_test = model.predict(X_test)
acc = my_acc(Y_pred_test, Y_test)

print_model = model.summary()
print('ACC:', acc)
print(print_model)


# In[14]:


Y_pred_test_m1=Y_pred_test
Y_pred_m1 = model.predict(X_train)
Y_pred_m1=Y_pred_m1.reshape(-1,1)


# # Prediction of M2

# Prediction of M0, M1 will be used to predict M2.

# In[15]:


def split_m2(past_days):
#     split_ind = 38  # if original
    split_ind = 36  # if digonal
    X_train, Y_train = [], []
    X_test, Y_test = [], []

    for ind in range(past_days, split_ind):
        for item_ind in range(len(fnames)):
            X_train.append(data_full[item_ind][ind - past_days: ind + 1][:,[5,6]])
            Y_train.append(data_full[item_ind][ind][-2]) # change the number of y

    for ind in range(split_ind, split_ind + 6):
        for item_ind in range(len(fnames)):
            X_test.append(data_full[item_ind][ind - past_days: ind + 1][:,[5,6]])
            Y_test.append(data_full[item_ind][ind][-2]) # change the number of y
    X_train, Y_train = np.array(X_train), np.array(Y_train)
    X_test, Y_test = np.array(X_test), np.array(Y_test)
    if past_days == 0:
        X_train, Y_train, X_test, Y_test =         np.squeeze(X_train), np.squeeze(Y_train), np.squeeze(X_test), np.squeeze(Y_test) 
    X_train=np.c_[X_train,Y_pred_m0]
    X_train=np.c_[X_train,Y_pred_m1]
    X_test=np.c_[X_test,Y_pred_test_m0]
    X_test=np.c_[X_test,Y_pred_test_m1]
    print("The shape of X_train:", X_train.shape)
    print("The shape of Y_train:", Y_train.shape)
    print("The shape of X_test:", X_test.shape)
    print("The shape of Y_train:", Y_test.shape)
    return X_train, Y_train, X_test, Y_test

X_train, Y_train, X_test, Y_test = split_m2(0) 
Y_test=Y_test.reshape(-1,1)


# In[16]:


X_train=sm.add_constant(X_train)
X_test=sm.add_constant(X_test)
model=sm.OLS(Y_train,X_train).fit()
Y_pred_test = model.predict(X_test)
Y_pred_test=Y_pred_test.reshape(-1,1)
Y_pred_test = model.predict(X_test)
acc = my_acc(Y_pred_test, Y_test)

print_model = model.summary()
print('ACC:', acc)
print(print_model)


# In[17]:


Y_pred_test_m2=Y_pred_test
Y_pred_m2 = model.predict(X_train)
Y_pred_m2=Y_pred_m1.reshape(-1,1)


# # Prediction of M3

# Prediction of M0, M1, M2 will be used to predict M3.

# In[18]:


def split_m3(past_days):
#     split_ind = 38  # if original
    split_ind = 36  # if digonal
    X_train, Y_train = [], []
    X_test, Y_test = [], []

    for ind in range(past_days, split_ind):
        for item_ind in range(len(fnames)):
            X_train.append(data_full[item_ind][ind - past_days: ind + 1][:,[6]])
            Y_train.append(data_full[item_ind][ind][-1]) # change the number of y

    for ind in range(split_ind, split_ind + 6):
        for item_ind in range(len(fnames)):
            X_test.append(data_full[item_ind][ind - past_days: ind + 1][:,[6]])
            Y_test.append(data_full[item_ind][ind][-1]) # change the number of y
    X_train, Y_train = np.array(X_train), np.array(Y_train)
    X_test, Y_test = np.array(X_test), np.array(Y_test)
    if past_days == 0:
        X_train, Y_train, X_test, Y_test =         np.squeeze(X_train), np.squeeze(Y_train), np.squeeze(X_test), np.squeeze(Y_test) 
    X_train=np.c_[X_train,Y_pred_m0]
    X_train=np.c_[X_train,Y_pred_m1]
    X_train=np.c_[X_train,Y_pred_m2]
    X_test=np.c_[X_test,Y_pred_test_m0]
    X_test=np.c_[X_test,Y_pred_test_m1]
    X_test=np.c_[X_test,Y_pred_test_m2]
    print("The shape of X_train:", X_train.shape)
    print("The shape of Y_train:", Y_train.shape)
    print("The shape of X_test:", X_test.shape)
    print("The shape of Y_train:", Y_test.shape)
    return X_train, Y_train, X_test, Y_test

X_train, Y_train, X_test, Y_test = split_m3(0) 
Y_test=Y_test.reshape(-1,1)


# In[19]:


X_train=sm.add_constant(X_train)
X_test=sm.add_constant(X_test)
model=sm.OLS(Y_train,X_train).fit()
Y_pred_test = model.predict(X_test)
Y_pred_test=Y_pred_test.reshape(-1,1)
Y_pred_test = model.predict(X_test)
acc = my_acc(Y_pred_test, Y_test)

print_model = model.summary()
print('ACC:', acc)
print(print_model)


# # Add past 2 days M0

# In[65]:


def process_diagonal(past_m0=0):
    data_full = []
    if past_m0 <= 2:
        row = 42
        start_period = 1215
    else:
        row = 42 - (past_m0-2)
        start_period = 1215 + (past_m0-2)
    col = 10 + past_m0
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
        for i in range(past_m0):
            name = 'Last_M0' + str(i+1)
            data[name] = data['qty'].shift(i+1, axis=0)
        data['M0'] = data['qty']
        data['M1'] = data['qty'].shift(-1, axis=0)
        data['M2'] = data['qty'].shift(-2, axis=0)
        data['M3'] = data['qty'].shift(-3, axis=0)
        data = data.drop(['qty', 'qty1', 'qty2', 'qty3'], axis=1)
        data = data.dropna(how='any', axis=0)

        temp = np.zeros((row, col))  # this should be modified
        for i in range(start_period, 1257):
            try:
                temp[i-start_period] = data.loc[i].values
            except:
                continue
        data_full.append(temp)
    return data_full


# In[53]:


# data_full = process_original_with_last_2_m0()
data_full = process_diagonal(2)


# In[54]:


def split_m0(past_days):
#     split_ind = 38  # if original
    split_ind = 36  # if digonal
    X_train, Y_train = [], []
    X_test, Y_test = [], []

    for ind in range(past_days, split_ind):
        for item_ind in range(len(fnames)):
            X_train.append(data_full[item_ind][ind - past_days: ind + 1][:,[0,1,2,6,7]])
            Y_train.append(data_full[item_ind][ind][-4]) # change the number of y

    for ind in range(split_ind, split_ind + 6):
        for item_ind in range(len(fnames)):
            X_test.append(data_full[item_ind][ind - past_days: ind + 1][:,[0,1,2,6,7]])
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


# In[55]:


X_train=sm.add_constant(X_train)
X_test=sm.add_constant(X_test)
model=sm.OLS(Y_train,X_train).fit()
Y_pred_test = model.predict(X_test)
Y_pred_test=Y_pred_test.reshape(-1,1)

acc_list = []
for i in range(1146):
    acc_list.append(acc_out(i))
acc_list = np.array(acc_list)
print(acc_list.mean(axis=0))

print_model = model.summary()
print(print_model)


# In[56]:


Y_pred_test_m0=Y_pred_test
Y_pred_m0 = model.predict(X_train)
Y_pred_m0=Y_pred_m0.reshape(-1,1)


# In[57]:


def split_m1(past_days):
#     split_ind = 38  # if original
    split_ind = 36  # if digonal
    X_train, Y_train = [], []
    X_test, Y_test = [], []

    for ind in range(past_days, split_ind):
        for item_ind in range(len(fnames)):
            X_train.append(data_full[item_ind][ind - past_days: ind + 1][:,[3,4,6,7]])
            Y_train.append(data_full[item_ind][ind][-3]) # change the number of y

    for ind in range(split_ind, split_ind + 6):
        for item_ind in range(len(fnames)):
            X_test.append(data_full[item_ind][ind - past_days: ind + 1][:,[3,4,6,7]])
            Y_test.append(data_full[item_ind][ind][-3]) # change the number of y
    X_train, Y_train = np.array(X_train), np.array(Y_train)
    X_test, Y_test = np.array(X_test), np.array(Y_test)
    if past_days == 0:
        X_train, Y_train, X_test, Y_test =         np.squeeze(X_train), np.squeeze(Y_train), np.squeeze(X_test), np.squeeze(Y_test) 
    X_train=np.c_[X_train,Y_pred_m0]
    X_test=np.c_[X_test,Y_pred_test_m0]
    print("The shape of X_train:", X_train.shape)
    print("The shape of Y_train:", Y_train.shape)
    print("The shape of X_test:", X_test.shape)
    print("The shape of Y_train:", Y_test.shape)
    return X_train, Y_train, X_test, Y_test

X_train, Y_train, X_test, Y_test = split_m1(0) 
Y_test=Y_test.reshape(-1,1)


# In[58]:


X_train=sm.add_constant(X_train)
X_test=sm.add_constant(X_test)
model=sm.OLS(Y_train,X_train).fit()
Y_pred_test = model.predict(X_test)
Y_pred_test=Y_pred_test.reshape(-1,1)

acc_list = []
for i in range(1146):
    acc_list.append(acc_out(i))
acc_list = np.array(acc_list)
print(acc_list.mean(axis=0))

print_model = model.summary()
print(print_model)


# In[59]:


Y_pred_test_m1=Y_pred_test
Y_pred_m1 = model.predict(X_train)
Y_pred_m1=Y_pred_m1.reshape(-1,1)


# In[60]:


def split_m2(past_days):
#     split_ind = 38  # if original
    split_ind = 36  # if digonal
    X_train, Y_train = [], []
    X_test, Y_test = [], []

    for ind in range(past_days, split_ind):
        for item_ind in range(len(fnames)):
            X_train.append(data_full[item_ind][ind - past_days: ind + 1][:,[5,6,7]])
            Y_train.append(data_full[item_ind][ind][-2]) # change the number of y

    for ind in range(split_ind, split_ind + 6):
        for item_ind in range(len(fnames)):
            X_test.append(data_full[item_ind][ind - past_days: ind + 1][:,[5,6,7]])
            Y_test.append(data_full[item_ind][ind][-2]) # change the number of y
    X_train, Y_train = np.array(X_train), np.array(Y_train)
    X_test, Y_test = np.array(X_test), np.array(Y_test)
    if past_days == 0:
        X_train, Y_train, X_test, Y_test =         np.squeeze(X_train), np.squeeze(Y_train), np.squeeze(X_test), np.squeeze(Y_test) 
    X_train=np.c_[X_train,Y_pred_m0]
    X_train=np.c_[X_train,Y_pred_m1]
    X_test=np.c_[X_test,Y_pred_test_m0]
    X_test=np.c_[X_test,Y_pred_test_m1]
    print("The shape of X_train:", X_train.shape)
    print("The shape of Y_train:", Y_train.shape)
    print("The shape of X_test:", X_test.shape)
    print("The shape of Y_train:", Y_test.shape)
    return X_train, Y_train, X_test, Y_test

X_train, Y_train, X_test, Y_test = split_m2(0) 
Y_test=Y_test.reshape(-1,1)


# In[61]:


X_train=sm.add_constant(X_train)
X_test=sm.add_constant(X_test)
model=sm.OLS(Y_train,X_train).fit()
Y_pred_test = model.predict(X_test)
Y_pred_test=Y_pred_test.reshape(-1,1)

acc_list = []
for i in range(1146):
    acc_list.append(acc_out(i))
acc_list = np.array(acc_list)
print(acc_list.mean(axis=0))

print_model = model.summary()
print(print_model)


# In[62]:


Y_pred_test_m2=Y_pred_test
Y_pred_m2 = model.predict(X_train)
Y_pred_m2=Y_pred_m1.reshape(-1,1)


# In[63]:


def split_m3(past_days):
#     split_ind = 38  # if original
    split_ind = 36  # if digonal
    X_train, Y_train = [], []
    X_test, Y_test = [], []

    for ind in range(past_days, split_ind):
        for item_ind in range(len(fnames)):
            X_train.append(data_full[item_ind][ind - past_days: ind + 1][:,[6,7]])
            Y_train.append(data_full[item_ind][ind][-1]) # change the number of y

    for ind in range(split_ind, split_ind + 6):
        for item_ind in range(len(fnames)):
            X_test.append(data_full[item_ind][ind - past_days: ind + 1][:,[6,7]])
            Y_test.append(data_full[item_ind][ind][-1]) # change the number of y
    X_train, Y_train = np.array(X_train), np.array(Y_train)
    X_test, Y_test = np.array(X_test), np.array(Y_test)
    if past_days == 0:
        X_train, Y_train, X_test, Y_test =         np.squeeze(X_train), np.squeeze(Y_train), np.squeeze(X_test), np.squeeze(Y_test) 
    X_train=np.c_[X_train,Y_pred_m0]
    X_train=np.c_[X_train,Y_pred_m1]
    X_train=np.c_[X_train,Y_pred_m2]
    X_test=np.c_[X_test,Y_pred_test_m0]
    X_test=np.c_[X_test,Y_pred_test_m1]
    X_test=np.c_[X_test,Y_pred_test_m2]
    print("The shape of X_train:", X_train.shape)
    print("The shape of Y_train:", Y_train.shape)
    print("The shape of X_test:", X_test.shape)
    print("The shape of Y_train:", Y_test.shape)
    return X_train, Y_train, X_test, Y_test

X_train, Y_train, X_test, Y_test = split_m3(0) 
Y_test=Y_test.reshape(-1,1)


# In[64]:


X_train=sm.add_constant(X_train)
X_test=sm.add_constant(X_test)
model=sm.OLS(Y_train,X_train).fit()
Y_pred_test = model.predict(X_test)
Y_pred_test=Y_pred_test.reshape(-1,1)

acc_list = []
for i in range(1146):
    acc_list.append(acc_out(i))
acc_list = np.array(acc_list)
print(acc_list.mean(axis=0))

print_model = model.summary()
print(print_model)


# # Add past 3days M0

# In[74]:


# data_full = process_original_with_last_2_m0()
data_full = process_diagonal(3)


# In[75]:


def split_m0(past_days):
#     split_ind = 38  # if original
    split_ind = 34  # if digonal
    X_train, Y_train = [], []
    X_test, Y_test = [], []

    for ind in range(past_days, split_ind):
        for item_ind in range(len(fnames)):
            X_train.append(data_full[item_ind][ind - past_days: ind + 1][:,[0,1,2,6,7,8]])
            Y_train.append(data_full[item_ind][ind][-4]) # change the number of y

    for ind in range(split_ind, split_ind + 6):
        for item_ind in range(len(fnames)):
            X_test.append(data_full[item_ind][ind - past_days: ind + 1][:,[0,1,2,6,7,8]])
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


# In[76]:


X_train=sm.add_constant(X_train)
X_test=sm.add_constant(X_test)
model=sm.OLS(Y_train,X_train).fit()
Y_pred_test = model.predict(X_test)
Y_pred_test=Y_pred_test.reshape(-1,1)

acc_list = []
for i in range(1146):
    acc_list.append(acc_out(i))
acc_list = np.array(acc_list)
print(acc_list.mean(axis=0))

print_model = model.summary()
print(print_model)


# In[77]:


Y_pred_test_m0=Y_pred_test
Y_pred_m0 = model.predict(X_train)
Y_pred_m0=Y_pred_m0.reshape(-1,1)


# In[78]:


def split_m1(past_days):
#     split_ind = 38  # if original
    split_ind = 34  # if digonal
    X_train, Y_train = [], []
    X_test, Y_test = [], []

    for ind in range(past_days, split_ind):
        for item_ind in range(len(fnames)):
            X_train.append(data_full[item_ind][ind - past_days: ind + 1][:,[3,4,6,7,8]])
            Y_train.append(data_full[item_ind][ind][-3]) # change the number of y

    for ind in range(split_ind, split_ind + 6):
        for item_ind in range(len(fnames)):
            X_test.append(data_full[item_ind][ind - past_days: ind + 1][:,[3,4,6,7,8]])
            Y_test.append(data_full[item_ind][ind][-3]) # change the number of y
    X_train, Y_train = np.array(X_train), np.array(Y_train)
    X_test, Y_test = np.array(X_test), np.array(Y_test)
    if past_days == 0:
        X_train, Y_train, X_test, Y_test =         np.squeeze(X_train), np.squeeze(Y_train), np.squeeze(X_test), np.squeeze(Y_test) 
    X_train=np.c_[X_train,Y_pred_m0]
    X_test=np.c_[X_test,Y_pred_test_m0]
    print("The shape of X_train:", X_train.shape)
    print("The shape of Y_train:", Y_train.shape)
    print("The shape of X_test:", X_test.shape)
    print("The shape of Y_train:", Y_test.shape)
    return X_train, Y_train, X_test, Y_test

X_train, Y_train, X_test, Y_test = split_m1(0) 
Y_test=Y_test.reshape(-1,1)


# In[79]:


X_train=sm.add_constant(X_train)
X_test=sm.add_constant(X_test)
model=sm.OLS(Y_train,X_train).fit()
Y_pred_test = model.predict(X_test)
Y_pred_test=Y_pred_test.reshape(-1,1)

acc_list = []
for i in range(1146):
    acc_list.append(acc_out(i))
acc_list = np.array(acc_list)
print(acc_list.mean(axis=0))

print_model = model.summary()
print(print_model)


# In[80]:


Y_pred_test_m1=Y_pred_test
Y_pred_m1 = model.predict(X_train)
Y_pred_m1=Y_pred_m1.reshape(-1,1)


# In[81]:


def split_m2(past_days):
#     split_ind = 38  # if original
    split_ind = 34  # if digonal
    X_train, Y_train = [], []
    X_test, Y_test = [], []

    for ind in range(past_days, split_ind):
        for item_ind in range(len(fnames)):
            X_train.append(data_full[item_ind][ind - past_days: ind + 1][:,[5,6,7,8]])
            Y_train.append(data_full[item_ind][ind][-2]) # change the number of y

    for ind in range(split_ind, split_ind + 6):
        for item_ind in range(len(fnames)):
            X_test.append(data_full[item_ind][ind - past_days: ind + 1][:,[5,6,7,8]])
            Y_test.append(data_full[item_ind][ind][-2]) # change the number of y
    X_train, Y_train = np.array(X_train), np.array(Y_train)
    X_test, Y_test = np.array(X_test), np.array(Y_test)
    if past_days == 0:
        X_train, Y_train, X_test, Y_test =         np.squeeze(X_train), np.squeeze(Y_train), np.squeeze(X_test), np.squeeze(Y_test) 
    X_train=np.c_[X_train,Y_pred_m0]
    X_train=np.c_[X_train,Y_pred_m1]
    X_test=np.c_[X_test,Y_pred_test_m0]
    X_test=np.c_[X_test,Y_pred_test_m1]
    print("The shape of X_train:", X_train.shape)
    print("The shape of Y_train:", Y_train.shape)
    print("The shape of X_test:", X_test.shape)
    print("The shape of Y_train:", Y_test.shape)
    return X_train, Y_train, X_test, Y_test

X_train, Y_train, X_test, Y_test = split_m2(0) 
Y_test=Y_test.reshape(-1,1)


# In[82]:


X_train=sm.add_constant(X_train)
X_test=sm.add_constant(X_test)
model=sm.OLS(Y_train,X_train).fit()
Y_pred_test = model.predict(X_test)
Y_pred_test=Y_pred_test.reshape(-1,1)

acc_list = []
for i in range(1146):
    acc_list.append(acc_out(i))
acc_list = np.array(acc_list)
print(acc_list.mean(axis=0))

print_model = model.summary()
print(print_model)


# In[83]:


Y_pred_test_m2=Y_pred_test
Y_pred_m2 = model.predict(X_train)
Y_pred_m2=Y_pred_m1.reshape(-1,1)


# In[84]:


def split_m3(past_days):
#     split_ind = 38  # if original
    split_ind = 34  # if digonal
    X_train, Y_train = [], []
    X_test, Y_test = [], []

    for ind in range(past_days, split_ind):
        for item_ind in range(len(fnames)):
            X_train.append(data_full[item_ind][ind - past_days: ind + 1][:,[6,7,8]])
            Y_train.append(data_full[item_ind][ind][-1]) # change the number of y

    for ind in range(split_ind, split_ind + 6):
        for item_ind in range(len(fnames)):
            X_test.append(data_full[item_ind][ind - past_days: ind + 1][:,[6,7,8]])
            Y_test.append(data_full[item_ind][ind][-1]) # change the number of y
    X_train, Y_train = np.array(X_train), np.array(Y_train)
    X_test, Y_test = np.array(X_test), np.array(Y_test)
    if past_days == 0:
        X_train, Y_train, X_test, Y_test =         np.squeeze(X_train), np.squeeze(Y_train), np.squeeze(X_test), np.squeeze(Y_test) 
    X_train=np.c_[X_train,Y_pred_m0]
    X_train=np.c_[X_train,Y_pred_m1]
    X_train=np.c_[X_train,Y_pred_m2]
    X_test=np.c_[X_test,Y_pred_test_m0]
    X_test=np.c_[X_test,Y_pred_test_m1]
    X_test=np.c_[X_test,Y_pred_test_m2]
    print("The shape of X_train:", X_train.shape)
    print("The shape of Y_train:", Y_train.shape)
    print("The shape of X_test:", X_test.shape)
    print("The shape of Y_train:", Y_test.shape)
    return X_train, Y_train, X_test, Y_test

X_train, Y_train, X_test, Y_test = split_m3(0) 
Y_test=Y_test.reshape(-1,1)


# In[85]:


X_train=sm.add_constant(X_train)
X_test=sm.add_constant(X_test)
model=sm.OLS(Y_train,X_train).fit()
Y_pred_test = model.predict(X_test)
Y_pred_test=Y_pred_test.reshape(-1,1)

acc_list = []
for i in range(1146):
    acc_list.append(acc_out(i))
acc_list = np.array(acc_list)
print(acc_list.mean(axis=0))

print_model = model.summary()
print(print_model)

