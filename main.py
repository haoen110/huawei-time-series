#!/usr/bin/env python
# coding: utf-8

# # Packages

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
gpu = tf.config.experimental.list_physical_devices('GPU')[0]
tf.config.experimental.set_memory_growth(gpu, True)
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")

# # Preprocessing

data_path = 'Time_Series'
fnames = [fname for fname in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, fname))]
fnames.sort()
print(len(fnames), "files")

def process_original(past_m0=0):
    data_full = []
    row = 44 - past_m0
    col = 7 + past_m0
    start_period = 1213 + past_m0
    for fname in fnames:
        data = pd.read_csv(os.path.join(data_path, fname), index_col=0)
        data = data.drop([' item'], axis=1)
        data.columns = ['qty', 'qty1', 'qty2', 'qty3']
        for i in range(past_m0):
            name = 'Last_M0' + str(i+1)
            data[name] = data['qty'].shift(i+1, axis=0)
        data['M0'] = data['qty']
        data['M1'] = data['qty'].shift(-1, axis=0)
        data['M2'] = data['qty'].shift(-2, axis=0)
        data['M3'] = data['qty'].shift(-3, axis=0)
        data = data.drop(['qty'], axis=1)
        data = data.dropna(how='any', axis=0)
        temp = np.zeros((row, col))  # this should be modified
        for i in range(start_period, 1257):
            try:
                temp[i-start_period] = data.loc[i].values
            except:
                continue
        data_full.append(temp)
    return data_full, ['original', past_m0]

def process_diagonal(past_m0=0):
    data_full = []
    if past_m0 <= 2:
        row = 42
        start_period = 1215
    else:
        row = 42 - (past_m0 -2)
        start_period = 1215 + (past_m0 - 2)
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
    return data_full, ['diagonal', past_m0]

def split(past_days, status, m):
    if status[0] == 'original':
        split_ind = 38 - status[1]
    if status[0] == 'diagonal':
        if status[1] <= 2:
            split_ind = 36
        else:
            split_ind = 36 - (status[1] - 2)
    X_train, Y_train = [], []
    X_test, Y_test = [], []
    for ind in range(past_days, split_ind):
        for item_ind in range(len(fnames)):
            X_train.append(data_full[item_ind][ind - past_days: ind + 1][:, :-4])
            Y_train.append(data_full[item_ind][ind][-4:])

    for ind in range(split_ind, split_ind + 6):
        for item_ind in range(len(fnames)):
            X_test.append(data_full[item_ind][ind - past_days: ind + 1][:, :-4])
            Y_test.append(data_full[item_ind][ind][-4:])
    X_train, Y_train = np.array(X_train), np.array(Y_train)
    X_test, Y_test = np.array(X_test), np.array(Y_test)
    if past_days == 0:
        X_train, Y_train, X_test, Y_test =         np.squeeze(X_train), np.squeeze(Y_train), np.squeeze(X_test), np.squeeze(Y_test) 
    if m == 0:
        Y_train = Y_train[:, 0].reshape(Y_train.shape[0], 1)
        Y_test = Y_test[:, 0].reshape(Y_test.shape[0], 1)
        # 筛选
        if status[0] == 'diagonal':
            X_train = np.concatenate((X_train[:, 0:3], X_train[:, 6:]), axis=1)
            X_test = np.concatenate((X_test[:, 0:3], X_test[:, 6:]), axis=1)
    if m == 1:
        Y_train = Y_train[:, 1].reshape(Y_train.shape[0], 1)
        Y_test = Y_test[:, 1].reshape(Y_test.shape[0], 1)
        if status[0] == 'diagonal':
            X_train = np.concatenate((X_train[:, 3:5], X_train[:, 6:]), axis=1)
            X_test = np.concatenate((X_test[:, 3:5], X_test[:, 6:]), axis=1)
    if m == 2:
        Y_train = Y_train[:, 2].reshape(Y_train.shape[0], 1)
        Y_test = Y_test[:, 2].reshape(Y_test.shape[0], 1)
        if status[0] == 'diagonal':
            X_train = X_train[:, 5:]
            X_test = X_test[:, 5:]
    if m == 3:
        Y_train = Y_train[:, 3].reshape(Y_train.shape[0], 1)
        Y_test = Y_test[:, 3].reshape(Y_test.shape[0], 1)
        if status[0] == 'diagonal':
            X_train = X_train[:, 6:]
            X_test = X_test[:, 6:]
    
    return X_train, Y_train, X_test, Y_test

def build_rnn(verbose=1):
    epochs, batch_size = 20, 30
    n_timesteps, n_features, n_outputs = X_train.shape[1], X_train.shape[2], Y_train.shape[1]
    np.random.seed(42)
    tf.random.set_seed(42)
    model = keras.models.Sequential()
    model.add(keras.layers.LSTM(75, activation='relu', input_shape=(n_timesteps, n_features), return_sequences=True))
    model.add(keras.layers.LSTM(30, return_sequences=True))
    model.add(keras.layers.LSTM(30))
    model.add(keras.layers.Dense(30, activation='relu'))
    model.add(keras.layers.Dense(n_outputs))
    model.compile(loss='mse', optimizer='adam')
    if verbose == 1:
        model.summary()
    history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, verbose=verbose, 
                        callbacks=[keras.callbacks.EarlyStopping(patience=10)],
                        validation_data=(X_test, Y_test))
    return model, history

def build_nn(verbose=1):
    epochs, batch_size = 20, 30
    np.random.seed(42)
    tf.random.set_seed(42)
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(20, activation='relu', input_dim=X_train.shape[1]))
    model.add(keras.layers.Dense(20))
    model.add(keras.layers.Dense(10))
    model.add(keras.layers.Dense(10))
    model.add(keras.layers.Dense(5))
    model.add(keras.layers.Dense(Y_train.shape[1]))
    model.compile(loss='mse', optimizer='adam')
    if verbose == 1:
        model.summary()
    history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, verbose=verbose, 
                        callbacks=[keras.callbacks.EarlyStopping(patience=10)],
                        validation_data=(X_test, Y_test))
    return model, history

def my_acc(Y_pred_test, Y_test):
    acc = []
    for index, value in enumerate(Y_pred_test):
        if max(Y_test[index], Y_pred_test[index]) == 0:
            acc.append(np.nan)
            continue
        acc.append((min(Y_test[index], Y_pred_test[index])/max(Y_test[index], Y_pred_test[index]))[0])
    return np.nanmean(acc)


# # Original Preprocessing and Diagonal Preprocessing
# 
# ## 1. MLP + Original
# 
# ### Precict M0

Past_M0 = []
MSE = []
ACC = []
for past_m0 in range(31):
    print("Preprocessing...PastM0 =", past_m0)
    data_full, status = process_original(past_m0)
    X_train, Y_train, X_test, Y_test = split(0, status)
    print("Training...PastM0 =", past_m0)
    model, history = build_nn(verbose=0)
    mse_test = model.evaluate(X_test, Y_test)
    Y_pred_test = model.predict(X_test)
    acc = my_acc(Y_pred_test, Y_test)
    print('MSE:', mse_test, 'ACC:', acc)
    Past_M0.append(past_m0)
    MSE.append(mse_test)
    ACC.append(acc)
mlp_original = pd.DataFrame({'Past M0': Past_M0,
                             'MSE': MSE,
                             'ACC': ACC})
mlp_original.to_csv('output/mlp_original.csv', index=False)

# ### Predict M1

Past_M0 = []
MSE_1 = []
ACC_1 = []
for past_m0 in range(31):
    print("Preprocessing...PastM0 =", past_m0)
    data_full, status = process_original(past_m0)
    X_train, Y_train, X_test, Y_test = split(0, status)
    print("Training...PastM0 =", past_m0)
    model, history = build_nn(verbose=0)
    mse_test = model.evaluate(X_test, Y_test)
    Y_pred_test = model.predict(X_test)
    acc = my_acc(Y_pred_test, Y_test)
    print('MSE:', mse_test, 'ACC:', acc)
    Past_M0.append(past_m0)
    MSE_1.append(mse_test)
    ACC_1.append(acc)
mlp_original_1 = pd.DataFrame({'Past M0': Past_M0,
                             'MSE': MSE_1,
                             'ACC': ACC_1})
mlp_original_1.to_csv('output/mlp_original_1.csv', index=False)

# ### Precict M2

Past_M0 = []
MSE_2 = []
ACC_2 = []
for past_m0 in range(31):
    print("Preprocessing...PastM0 =", past_m0)
    data_full, status = process_original(past_m0)
    X_train, Y_train, X_test, Y_test = split(0, status)
    print("Training...PastM0 =", past_m0)
    model, history = build_nn(verbose=0)
    mse_test = model.evaluate(X_test, Y_test)
    Y_pred_test = model.predict(X_test)
    acc = my_acc(Y_pred_test, Y_test)
    print('MSE:', mse_test, 'ACC:', acc)
    Past_M0.append(past_m0)
    MSE_2.append(mse_test)
    ACC_2.append(acc)
mlp_original_2 = pd.DataFrame({'Past M0': Past_M0,
                             'MSE': MSE_2,
                             'ACC': ACC_2})
mlp_original_2.to_csv('output/mlp_original_2.csv', index=False)

# ### Precict M3

Past_M0 = []
MSE_3 = []
ACC_3 = []
for past_m0 in range(31):
    print("Preprocessing...PastM0 =", past_m0)
    data_full, status = process_original(past_m0)
    X_train, Y_train, X_test, Y_test = split(0, status)
    print("Training...PastM0 =", past_m0)
    model, history = build_nn(verbose=0)
    mse_test = model.evaluate(X_test, Y_test)
    Y_pred_test = model.predict(X_test)
    acc = my_acc(Y_pred_test, Y_test)
    print('MSE:', mse_test, 'ACC:', acc)
    Past_M0.append(past_m0)
    MSE_3.append(mse_test)
    ACC_3.append(acc)
mlp_original_3 = pd.DataFrame({'Past M0': Past_M0,
                             'MSE': MSE_3,
                             'ACC': ACC_3})
mlp_original_3.to_csv('output/mlp_original_3.csv', index=False)

plt.figure(figsize=(12, 6))
mlp_original['ACC'].plot(label='Accuracy of M0')
mlp_original_1['ACC'].plot(label='Accuracy of M1')
mlp_original_2['ACC'].plot(label='Accuracy of M2')
mlp_original_3['ACC'].plot(label='Accuracy of M3')
plt.title('MLP (Original Process) with advanced periods of M0')
plt.xlabel('Extra M0')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('figures/mlp_original.png', dpi=300)


# ## 2. MLP + Diagnal

# ### Predict M0

D_Past_M0 = []
D_MSE = []
D_ACC = []
for past_m0 in range(31):
    print("Preprocessing...PastM0 =", past_m0)
    data_full, status = process_diagonal(past_m0)
    X_train, Y_train, X_test, Y_test = split(0, status, 0)
    print("Training...PastM0 =", past_m0)
    model, history = build_nn(verbose=0)
    mse_test = model.evaluate(X_test, Y_test)
    Y_pred_test = model.predict(X_test)
    acc = my_acc(Y_pred_test, Y_test)
    print('MSE:', mse_test, 'ACC:', acc)
    D_Past_M0.append(past_m0)
    D_MSE.append(mse_test)
    D_ACC.append(acc)

mlp_diagonal = pd.DataFrame({'Past M0': D_Past_M0,
                             'MSE': D_MSE,
                             'ACC': D_ACC})#### Predict M3
mlp_diagonal.to_csv('output/mlp_diagonal.csv', index=False)

# ### Predict M1

D_Past_M0 = []
D_MSE_1 = []
D_ACC_1 = []
for past_m0 in range(31):
    print("Preprocessing...PastM0 =", past_m0)
    data_full, status = process_diagonal(past_m0)
    X_train, Y_train, X_test, Y_test = split(0, status, 1)
    print("Training...PastM0 =", past_m0)
    model, history = build_nn(verbose=0)
    mse_test = model.evaluate(X_test, Y_test)
    Y_pred_test = model.predict(X_test)
    acc = my_acc(Y_pred_test, Y_test)
    print('MSE:', mse_test, 'ACC:', acc)
    D_Past_M0.append(past_m0)
    D_MSE_1.append(mse_test)
    D_ACC_1.append(acc)
mlp_diagonal_1 = pd.DataFrame({'Past M0': D_Past_M0,
                             'MSE': D_MSE_1,
                             'ACC': D_ACC_1})
mlp_diagonal_1.to_csv('output/mlp_diagonal_1.csv', index=False)

# ### Predict M2

D_Past_M0 = []
D_MSE_2 = []
D_ACC_2 = []
for past_m0 in range(31):
    print("Preprocessing...PastM0 =", past_m0)
    data_full, status = process_diagonal(past_m0)
    X_train, Y_train, X_test, Y_test = split(0, status, 2)
    print("Training...PastM0 =", past_m0)
    model, history = build_nn(verbose=0)
    mse_test = model.evaluate(X_test, Y_test)
    Y_pred_test = model.predict(X_test)
    acc = my_acc(Y_pred_test, Y_test)
    print('MSE:', mse_test, 'ACC:', acc)
    D_Past_M0.append(past_m0)
    D_MSE_2.append(mse_test)
    D_ACC_2.append(acc)
mlp_diagonal_2 = pd.DataFrame({'Past M0': D_Past_M0,
                             'MSE': D_MSE_2,
                             'ACC': D_ACC_2})
mlp_diagonal_2.to_csv('output/mlp_diagonal_2.csv', index=False)

D_Past_M0 = []
D_MSE_3 = []
D_ACC_3 = []
for past_m0 in range(31):
    print("Preprocessing...PastM0 =", past_m0)
    data_full, status = process_diagonal(past_m0)
    X_train, Y_train, X_test, Y_test = split(0, status, 3)
    print("Training...PastM0 =", past_m0)
    model, history = build_nn(verbose=0)
    mse_test = model.evaluate(X_test, Y_test)
    Y_pred_test = model.predict(X_test)
    acc = my_acc(Y_pred_test, Y_test)
    print('MSE:', mse_test, 'ACC:', acc)
    D_Past_M0.append(past_m0)
    D_MSE_3.append(mse_test)
    D_ACC_3.append(acc)
mlp_diagonal_3 = pd.DataFrame({'Past M0': D_Past_M0,
                             'MSE': D_MSE_3,
                             'ACC': D_ACC_3})
mlp_diagonal_3.to_csv('output/mlp_diagonal_3.csv', index=False)

plt.figure(figsize=(12, 6))
mlp_diagonal['ACC'].plot(label='Accuracy of M0')
mlp_diagonal_1['ACC'].plot(label='Accuracy of M1')
mlp_diagonal_2['ACC'].plot(label='Accuracy of M2')
mlp_diagonal_3['ACC'].plot(label='Accuracy of M3')
plt.title('MLP (Diagonal Process) with advanced periods of M0')
plt.xlabel('Extra M0')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('figures/mlp_diagonal.png', dpi=300)


# ## RNN + Original
# 
# ### Predict M0

Past_M0 = []
rnn_MSE = []
rnn_ACC = []
for past_m0 in range(31):
    print("Preprocessing...Past_M0:", past_m0)
    data_full, status = process_original(past_m0)
    X_train, Y_train, X_test, Y_test = split(1, status, 0)
    print("Training...Past_M0:", past_m0)
    model, history = build_rnn(verbose=0)
    mse_test = model.evaluate(X_test, Y_test)
    Y_pred_test = model.predict(X_test)
    acc = my_acc(Y_pred_test, Y_test)
    print('MSE:', mse_test, 'ACC:', acc)
    Past_M0.append(past_m0)
    rnn_MSE.append(mse_test)
    rnn_ACC.append(acc)
rnn_original = pd.DataFrame({'Past M0': Past_M0,
                             'MSE': rnn_MSE,
                             'ACC': rnn_ACC})
rnn_original.to_csv('output/rnn_original.csv', index=False)


# ### Predict M1

Past_M0 = []
rnn_MSE_1 = []
rnn_ACC_1 = []
for past_m0 in range(31):
    print("Preprocessing...Past_M0:", past_m0)
    data_full, status = process_original(past_m0)
    X_train, Y_train, X_test, Y_test = split(1, status, 1)
    print("Training...Past_M0:", past_m0)
    model, history = build_rnn(verbose=0)
    mse_test = model.evaluate(X_test, Y_test)
    Y_pred_test = model.predict(X_test)
    acc = my_acc(Y_pred_test, Y_test)
    print('MSE:', mse_test, 'ACC:', acc)
    Past_M0.append(past_m0)
    rnn_MSE_1.append(mse_test)
    rnn_ACC_1.append(acc)
rnn_original_1 = pd.DataFrame({'Past M0': Past_M0,
                             'MSE': rnn_MSE_1,
                             'ACC': rnn_ACC_1})
rnn_original_1.to_csv('output/rnn_original_1.csv', index=False)

# ### Predict M2

Past_M0 = []
rnn_MSE_2 = []
rnn_ACC_2 = []
for past_m0 in range(31):
    print("Preprocessing...Past_M0:", past_m0)
    data_full, status = process_original(past_m0)
    X_train, Y_train, X_test, Y_test = split(1, status, 2)
    print("Training...Past_M0:", past_m0)
    model, history = build_rnn(verbose=0)
    mse_test = model.evaluate(X_test, Y_test)
    Y_pred_test = model.predict(X_test)
    acc = my_acc(Y_pred_test, Y_test)
    print('MSE:', mse_test, 'ACC:', acc)
    Past_M0.append(past_m0)
    rnn_MSE_2.append(mse_test)
    rnn_ACC_2.append(acc)
rnn_original_2 = pd.DataFrame({'Past M0': Past_M0,
                             'MSE': rnn_MSE_2,
                             'ACC': rnn_ACC_2})
rnn_original_2.to_csv('output/rnn_original_2.csv', index=False)

# ### Predict M3

Past_M0 = []
rnn_MSE_3 = []
rnn_ACC_3 = []
for past_m0 in range(31):
    print("Preprocessing...Past_M0:", past_m0)
    data_full, status = process_original(past_m0)
    X_train, Y_train, X_test, Y_test = split(1, status, 3)
    print("Training...Past_M0:", past_m0)
    model, history = build_rnn(verbose=0)
    mse_test = model.evaluate(X_test, Y_test)
    Y_pred_test = model.predict(X_test)
    acc = my_acc(Y_pred_test, Y_test)
    print('MSE:', mse_test, 'ACC:', acc)
    Past_M0.append(past_m0)
    rnn_MSE_3.append(mse_test)
    rnn_ACC_3.append(acc)
rnn_original_3 = pd.DataFrame({'Past M0': Past_M0,
                             'MSE': rnn_MSE_3,
                             'ACC': rnn_ACC_3})
rnn_original_3.to_csv('output/rnn_original_3.csv', index=False)
plt.figure(figsize=(12, 6))
rnn_original['ACC'].plot(label='Accuracy of M0')
rnn_original_1['ACC'].plot(label='Accuracy of M1')
rnn_original_2['ACC'].plot(label='Accuracy of M2')
rnn_original_3['ACC'].plot(label='Accuracy of M3')
plt.title('RNN (LSTM) with advanced periods of M0')
plt.xlabel('Extra M0')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('figures/rnn_original.png', dpi=300)


# # MLP + Filtered Diagonal

# ## MLP Selection

# In[263]:


def run():
    Past_M0 = []
    MSE = []
    ACC = []
    for past_m0 in range(26):
        print("Preprocessing...PastM0 =", past_m0)
        global data_full, status
        data_full, status = process_diagonal(past_m0)
        global X_train, Y_train, X_test, Y_test
        X_train, Y_train, X_test, Y_test = split(0, status, 0)
        print("Training...PastM0 =", past_m0)
        model, history = build_nn(verbose=0)
        mse_test = model.evaluate(X_test, Y_test)
        Y_pred_test = model.predict(X_test)
        acc = my_acc(Y_pred_test, Y_test)
        print('MSE:', mse_test, 'ACC:', acc)
        Past_M0.append(past_m0)
        MSE.append(mse_test)
        ACC.append(acc)
    return Past_M0, MSE, ACC

FD_Past_M0, FD_MSE, FD_ACC = run()
Fmlp_diagonal = pd.DataFrame({'Past M0': FD_Past_M0,
                             'MSE': FD_MSE,
                             'ACC': FD_ACC})
Fmlp_diagonal.to_csv('output/Fmlp_diagonal.csv', index=False)
plt.figure(figsize=(12, 6))
# mlp_original['MSE'].plot()
Fmlp_diagonal['ACC'].plot(label='Accuracy of M0')
plt.title('MLP (Diagonal Process with Filter) with advanced periods of M0')
plt.xlabel('Extra M0')
plt.ylabel('Accuracy')
plt.xticks(np.arange(26))
plt.legend()
plt.savefig('figures/Fmlp_diagonal.png', dpi=300)
# plt.show()


# In[23]:


def process_diagonal_slection(past_m0=[1]):
    data_full = []
    if max(past_m0) <= 2:
        row = 42
        start_period = 1215
    else:
        row = 42 - (max(past_m0)-2)
        start_period = 1215 + (max(past_m0)-2)
    col = 10 + len(past_m0)
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
        for i in past_m0:
            name = 'Last_M0' + str(i)
            data[name] = data['qty'].shift(i, axis=0)
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
    return data_full, ['diagonal', past_m0], np.array(data.columns)[0:-4]

def split_selection(past_days, status, m):
    if status[0] == 'original':
        split_ind = 38 - max(status[1])
    if status[0] == 'diagonal':
        if max(status[1]) <= 2:
            split_ind = 36
        else:
            split_ind = 36 - (max(status[1]) - 2)
    X_train, Y_train = [], []
    X_test, Y_test = [], []
    for ind in range(past_days, split_ind):
        for item_ind in range(len(fnames)):
            X_train.append(data_full[item_ind][ind - past_days: ind + 1][:, :-4])
            Y_train.append(data_full[item_ind][ind][-4:]) # change the number of y

    for ind in range(split_ind, split_ind + 6):
        for item_ind in range(len(fnames)):
            X_test.append(data_full[item_ind][ind - past_days: ind + 1][:, :-4])
            Y_test.append(data_full[item_ind][ind][-4:]) # change the number of y
    X_train, Y_train = np.array(X_train), np.array(Y_train)
    X_test, Y_test = np.array(X_test), np.array(Y_test)
    if past_days == 0:
        X_train, Y_train, X_test, Y_test =         np.squeeze(X_train), np.squeeze(Y_train), np.squeeze(X_test), np.squeeze(Y_test)
    else: # for rnn
        if m == 0:
            X_train = np.concatenate((X_train[:, :, 0:3], X_train[:, :, 6:]), axis=2)
            X_test = np.concatenate((X_test[:, :, 0:3], X_test[:, :, 6:]), axis=2)
    if m == 0:
        Y_train = Y_train[:, 0].reshape(Y_train.shape[0], 1)
        Y_test = Y_test[:, 0].reshape(Y_test.shape[0], 1)
        # 筛选
        if status[0] == 'diagonal' and past_days == 0:
            X_train = np.concatenate((X_train[:, 0:3], X_train[:, 6:]), axis=1)
            X_test = np.concatenate((X_test[:, 0:3], X_test[:, 6:]), axis=1)
    if m == 1:
        Y_train = Y_train[:, 1].reshape(Y_train.shape[0], 1)
        Y_test = Y_test[:, 1].reshape(Y_test.shape[0], 1)
        if status[0] == 'diagonal' and past_days == 0:
            X_train = np.concatenate((X_train[:, 3:5], X_train[:, 6:]), axis=1)
            X_test = np.concatenate((X_test[:, 3:5], X_test[:, 6:]), axis=1)
    if m == 2:
        Y_train = Y_train[:, 2].reshape(Y_train.shape[0], 1)
        Y_test = Y_test[:, 2].reshape(Y_test.shape[0], 1)
        if status[0] == 'diagonal' and past_days == 0:
            X_train = X_train[:, 5:]
            X_test = X_test[:, 5:]
    if m == 3:
        Y_train = Y_train[:, 3].reshape(Y_train.shape[0], 1)
        Y_test = Y_test[:, 3].reshape(Y_test.shape[0], 1)
        if status[0] == 'diagonal' and past_days == 0:
            X_train = X_train[:, 6:]
            X_test = X_test[:, 6:]
    
    return X_train, Y_train, X_test, Y_test


# In[9]:


def run():
    Past_M0 = []
    MSE = []
    ACC = []
    test_list = 
    for i in range(11):
        past_m0 = [i]
        print("Preprocessing...PastM0 =", past_m0)
    #     data_full, status = process_original(past_m0)
        global data_full, status
        data_full, status = process_diagonal_slection(past_m0)
        global X_train, Y_train, X_test, Y_test
        X_train, Y_train, X_test, Y_test = split_selection(0, status, 0)
        print("Training...PastM0 =", past_m0)
        model, history = build_nn(verbose=0)
        mse_test = model.evaluate(X_test, Y_test)
        Y_pred_test = model.predict(X_test)
        acc = my_acc(Y_pred_test, Y_test)
        print('MSE:', mse_test, 'ACC:', acc)
        Past_M0.append(past_m0)
        MSE.append(mse_test)
        ACC.append(acc)
    return Past_M0, MSE, ACC


# In[240]:


data_full, status, feature_names = process_diagonal_slection(list(range(1, 31)))


# In[241]:


X_train, Y_train, X_test, Y_test = split_selection(0, status, 0)


# In[242]:


X_train.shape


# In[243]:


Y_train.shape


# ## Importance of M0

# In[324]:


from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV
data_full, status, feature_names = process_diagonal_slection(list(range(1, 31)))
X_train, Y_train, X_test, Y_test = split_selection(0, status, 0)
feature_names = np.delete(feature_names, [3, 4, 5])
print(feature_names)


# In[325]:


clf = LassoCV().fit(X_train, Y_train.reshape(-1))
importance = np.abs(clf.coef_)
print(importance)


# In[326]:


IP = pd.DataFrame({"Features": feature_names, "Importance": importance})
IP = IP.sort_values(by=['Importance'], ascending=False)
plt.figure()
sns.barplot(x='Importance', y='Features', data=IP[:8])
plt.savefig('figures/lasso_importance.png', dpi=300)


# ## Importance of M1

# In[329]:


data_full, status, feature_names = process_diagonal_slection(list(range(1, 31)))
X_train, Y_train, X_test, Y_test = split_selection(0, status, 1)
feature_names = np.delete(feature_names, [0, 1, 2, 5])
print(feature_names)


# In[330]:


clf = LassoCV().fit(X_train, Y_train.reshape(-1))
importance = np.abs(clf.coef_)
print(importance)


# In[331]:


IP = pd.DataFrame({"Features": feature_names, "Importance": importance})
IP = IP.sort_values(by=['Importance'], ascending=False)
plt.figure()
sns.barplot(x='Importance', y='Features', data=IP[:8])
plt.savefig('figures/lasso_importance1.png', dpi=300)


# ## Importance of M2

# In[309]:


data_full, status, feature_names = process_diagonal_slection(list(range(1, 31)))
X_train, Y_train, X_test, Y_test = split_selection(0, status, 2)
feature_names = np.delete(feature_names, [0, 1, 2, 3, 4])
print(feature_names)


# In[310]:


clf = LassoCV().fit(X_train, Y_train.reshape(-1))
importance = np.abs(clf.coef_)
print(importance)


# In[311]:


IP = pd.DataFrame({"Features": feature_names, "Importance": importance})
IP = IP.sort_values(by=['Importance'], ascending=False)
plt.figure()
sns.barplot(x='Importance', y='Features', data=IP[:8])
plt.savefig('figures/lasso_importance2.png', dpi=300)


# ## Importance of M3

# data_full, status, feature_names = process_diagonal_slection(list(range(1, 31)))
# X_train, Y_train, X_test, Y_test = split_selection(0, status, 3)
# feature_names = np.delete(feature_names, [0, 1, 2, 3, 4, 5])
# print(feature_names)

# In[333]:


clf = LassoCV().fit(X_train, Y_train.reshape(-1))
importance = np.abs(clf.coef_)
print(importance)


# In[334]:


IP = pd.DataFrame({"Features": feature_names, "Importance": importance})
IP = IP.sort_values(by=['Importance'], ascending=False)
plt.figure()
sns.barplot(x='Importance', y='Features', data=IP[:8])
plt.savefig('figures/lasso_importance3.png', dpi=300)


# ## Multi-Step

# In[249]:


data_full, status, feature_names = process_diagonal_slection([1, 12, 24])

X_train, Y_train, X_test, Y_test = split_selection(0, status, 0)
model, history = build_nn(verbose=0)

# X_train, Y_train, X_test, Y_test = split_selection(1, status, 0)
# model, history = build_rnn(verbose=0)

Y_pred_test = model.predict(X_test)
acc = my_acc(Y_pred_test, Y_test)
print('ACC:', acc)

m0_predict_train = model.predict(X_train)
m0_predict_test = model.predict(X_test)


# ### Pre-trained Model Predict M0
# 
# The best model is when past_m0 = 7.

# In[315]:


data_full, status, feature_names = process_diagonal_slection([1, 3, 6, 9, 12, 24])
X_train, Y_train, X_test, Y_test = split_selection(0, status, 0)
print("Training...PastM0 =", 1, 'and', 12)
model, history = build_nn(verbose=0)
Y_pred_test = model.predict(X_test)
acc = my_acc(Y_pred_test, Y_test)
print('ACC:', acc)

m0_predict_train = model.predict(X_train)
m0_predict_test = model.predict(X_test)


# ### Pre-trained Model Predict M1

# In[328]:


data_full, status, feature_names = process_diagonal_slection([1, 2, 5, 11, 23, 24])
X_train, Y_train, X_test, Y_test = split_selection(0, status, 1)
X_train = np.concatenate([X_train, m0_predict_train], axis=1)
X_test = np.concatenate([X_test, m0_predict_test], axis=1)

model, history = build_nn(verbose=0)
Y_pred_test = model.predict(X_test)
acc = my_acc(Y_pred_test, Y_test)
print('ACC:', acc)

m1_predict_train = model.predict(X_train)
m1_predict_test = model.predict(X_test)


# ### Pre-trained Model Predict M2

# In[335]:


data_full, status, feature_names = process_diagonal_slection([1, 2, 3, 4, 9, 10])
X_train, Y_train, X_test, Y_test = split_selection(0, status, 2)
# X_train = np.concatenate([X_train, m1_predict_train], axis=1)
# X_test = np.concatenate([X_test, m1_predict_test], axis=1)

model, history = build_nn(verbose=0)
Y_pred_test = model.predict(X_test)
acc = my_acc(Y_pred_test, Y_test)
print('ACC:', acc)

m2_predict_train = model.predict(X_train)
m2_predict_test = model.predict(X_test)


# ### Pre-trained Model Predict M3

# In[336]:


data_full, status, feature_names = process_diagonal_slection([3, 9, 1, 10, 2, 8])
X_train, Y_train, X_test, Y_test = split_selection(0, status, 3)
# X_train = np.concatenate([X_train, m2_predict_train], axis=1)
# X_test = np.concatenate([X_test, m2_predict_test], axis=1)
model, history = build_nn(verbose=0)
Y_pred_test = model.predict(X_test)
acc = my_acc(Y_pred_test, Y_test)
print('ACC:', acc)


# In[227]:


def acc_out(item_index, Y_test, Y_pred_test):
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
    acc = acc.reshape(6, 4)
    return acc


# In[228]:


acc_list = []
for i in range(1146):
    acc_list.append(acc_out(i, Y_test, Y_pred_test))
acc_list = np.array(acc_list)


# In[229]:


np.nanmean(acc_list, axis=0)


# In[230]:


result = pd.DataFrame(np.nanmean(acc_list, axis=0))
result.columns = ['Accuracy of M0', 'Accuracy of M1', 
                  'Accuracy of M2', 'Accuracy of M3']
result['Accuracy of M012'] = (result.iloc[:, 0] + result.iloc[:, 1] + result.iloc[:, 2])/3
result['Accuracy of M123'] = (result.iloc[:, 1] + result.iloc[:, 2] + result.iloc[:, 3])/3
result

