#!/usr/bin/env python

import assignment1 as a1
import numpy as np
import matplotlib.pyplot as plt

(countries, features, values) = a1.load_unicef_data()

# print(countries)
# print(features)
# ------------------get started questions--------------
# minMortIndex1990 = np.argmin(values[:, 1])
# minMort1990 = np.min(values[:, 1])
# minCountry1990 = countries[minMortIndex1990]
# print(minMort1990)
# print(minCountry1990)

targets = values[:, 1]
x = values[:, 7:]
x = a1.normalize_data(x)

N_TRAIN = 100
x_train = x[0:N_TRAIN, :]
x_test = x[N_TRAIN:, :]
t_train = targets[0:N_TRAIN]
t_test = targets[N_TRAIN:]
reg_lambdas = [0, 0.01, 0.1, 10, 100, 1000, 10000]
degree = 2
# Complete the linear_regression and evaluate_regression functions of the assignment1.py
# Pass the required parameters to these functions
valid_err = dict()
for reg_lambda in reg_lambdas:
    valid_err_total = 0
    for i in range(10, 100, 10):
        x_train2 = np.vstack((x_train[0:i-10, :], x_train[i:, :]))
        x_valid = x_train[i-10:i, :]
        t_train2 = np.vstack((t_train[0:i-10], t_train[i:]))
        t_valid = t_train[i-10:i]
        (w, tr_err) = a1.linear_regression(x_train2, t_train2, 'polynomial',
                                           reg_lambda=reg_lambda, degree=degree, bias=1)
        (t_est, v_err) = a1.evaluate_regression(x_valid, t_valid, w, 'polynomial', degree)
        valid_err_total = valid_err_total+v_err
    valid_err[reg_lambda] = valid_err_total/10
print(valid_err)
# Produce a plot of results.
plt.rcParams.update({'font.size': 15})
plt.semilogx(list(valid_err.keys()), list(valid_err.values()))
# plt.plot(list(valid_err.keys()), list(valid_err.values()))
plt.ylabel('RMS')
plt.legend(['validation error'])
plt.title('average validation set error versus lambda')
plt.xlabel('different lambda')
plt.show()
