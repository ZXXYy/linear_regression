#!/usr/bin/env python

import assignment1 as a1
import numpy as np
import matplotlib.pyplot as plt

(countries, features, values) = a1.load_unicef_data()
# print(countries)
# print(features)

# initialize the variables
targets = values[:, 1]
train_err_bias = dict()
test_err_bias = dict()
train_err_no_bias = dict()
test_err_no_bias = dict()

for specific_feature in range(7, 15):
    # print(features[7])
    # choose a single feature
    x = values[:, specific_feature]
    # x = a1.normalize_data(x)

    N_TRAIN = 100
    x_train = x[0:N_TRAIN, :]
    x_test = x[N_TRAIN:, :]
    t_train = targets[0:N_TRAIN]
    t_test = targets[N_TRAIN:]
    # Complete the linear_regression and evaluate_regression functions of the assignment1.py
    # Pass the required parameters to these functions
    degree = 3
    # train the data with bias term
    (w, tr_err) = a1.linear_regression(x_train, t_train, 'polynomial', reg_lambda=0, degree=degree, bias=1)
    # evaluate the RMS error for test data
    (t_est, te_err) = a1.evaluate_regression(x_test, t_test, w, 'polynomial', degree, bias=1)
    train_err_bias[features[specific_feature]] = tr_err
    test_err_bias[features[specific_feature]] = te_err

    # train the data without bias term
    (w, tr_err) = a1.linear_regression(x_train, t_train, 'polynomial', reg_lambda=0, degree=degree, bias=0)
    (t_est, te_err) = a1.evaluate_regression(x_test, t_test, w, 'polynomial', degree, bias=0)
    train_err_no_bias[features[specific_feature]] = tr_err
    test_err_no_bias[features[specific_feature]] = te_err
    # print(tr_err)
    # print(te_err)


# Produce a plot of results.
plt.rcParams.update({'font.size': 15})
x = np.arange(len(list(train_err_bias.keys())))  # 首先用第一个的长度作为横坐标
width = 0.25    # 设置柱与柱之间的宽度
fig1 = plt.figure(num='fig1')
plt.figure(num='fig1')
plt.bar(x, list(train_err_bias.values()), width)
plt.bar(x+width, list(test_err_bias.values()), width, color='red')
plt.xticks(x + width/2, list(train_err_bias.keys()), rotation=10, horizontalalignment='right', fontsize=6)

plt.ylabel('RMS')
plt.legend(['Training error', 'Testing error'])
plt.xlabel('Features')
plt.title('Fit with polynomials, bias and no regularization ')

fig2 = plt.figure(num='fig2')
plt.figure(num='fig2')
plt.bar(x, list(train_err_no_bias.values()), width)
plt.bar(x+width, list(test_err_no_bias.values()), width, color='red')
plt.xticks(x + width/2, list(train_err_no_bias.keys()), rotation=10, horizontalalignment='right', fontsize=6)

plt.ylabel('RMS')
plt.legend(['Training error', 'Testing error'])
plt.xlabel('Features')
plt.title('Fit with polynomials, no bias and no regularization ')

plt.show()
