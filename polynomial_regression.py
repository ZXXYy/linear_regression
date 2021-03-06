#!/usr/bin/env python

import assignment1 as a1
import numpy as np
import matplotlib.pyplot as plt

(countries, features, values) = a1.load_unicef_data()
# print(countries)
# print(features)

# ------------------get started questions--------------
# minMortIndex1990 = np.argmin(values[:, 0])
# minMort1990 = np.min(values[:, 0])
# minCountry1990 = countries[minMortIndex1990]
# minMortIndex2011 = np.argmin(values[:, 1])
# minMort2011 = np.min(values[:, 1])
# minCountry2011 = countries[minMortIndex2011]
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


# Complete the linear_regression and evaluate_regression functions of the assignment1.py
# Pass the required parameters to these functions
max_degree = 6
train_err = dict()
test_err = dict()
# fit a polynomial basis function for degree 1 to degree 6
for degree in range(1, max_degree+1):
    (w, tr_err) = a1.linear_regression(x_train, t_train, 'polynomial', 0, degree=degree, bias=1)
    # evaluate the RMS error for test data
    (t_est, te_err) = a1.evaluate_regression(x_test, t_test, w, 'polynomial', degree)
    train_err[degree] = tr_err
    test_err[degree] = te_err
print(train_err)
print(test_err)


# Produce a plot of results.
plt.rcParams.update({'font.size': 15})
plt.plot(list(train_err.keys()), list(train_err.values()))
plt.plot(list(test_err.keys()), list(test_err.values()))
plt.ylabel('RMS')
plt.legend(['Training error', 'Testing error'])
plt.title('Fit with normalized polynomials, no regularization ')
plt.xlabel('Polynomial degree')
plt.show()
