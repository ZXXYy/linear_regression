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

targets = values[:,1]
train_err_bias = dict()
test_err_bias = dict()

specific_feature = 10
# print(features[7])
x = values[:, specific_feature]
x = a1.normalize_data(x)
N_TRAIN = 100
x_train = x[0:N_TRAIN, :]
x_test = x[N_TRAIN:, :]
t_train = targets[0:N_TRAIN]
t_test = targets[N_TRAIN:]
# Complete the linear_regression and evaluate_regression functions of the assignment1.py
# Pass the required parameters to these functions
mu = [100, 10000]
s = 2000.0
(w, tr_err) = a1.linear_regression(x_train, t_train, 'sigmoid', reg_lambda=0, mu=mu, s=s, bias=1)
(t_est, te_err) = a1.evaluate_regression(x_test, t_test, w, 'sigmoid', mu=mu, s=s, bias=1)
train_err_bias[features[specific_feature]] = tr_err
test_err_bias[features[specific_feature]] = te_err
print(tr_err)
print(te_err)


# Produce a plot of results.
x_ev = np.linspace(np.ndarray.item(min(x_train)), np.ndarray.item(max(x_train)), num=500)
x_ev2 = np.linspace(np.ndarray.item(min(min(x_train), min(x_test))),
                   np.ndarray.item(max(max(x_train), max(x_test))), num=500)
x_ev = x_ev.reshape((500, 1))
x_ev2 = x_ev2.reshape((500, 1))

# Perform regression on the linspace samples.
phi = a1.design_matrix(x_ev, mu=mu, s=s, basis='sigmoid', bias=1)
phi2 = a1.design_matrix(x_ev2, mu=mu, s=s, basis='sigmoid', bias=1)
y_ev = np.dot(phi, w)
y_ev2 = np.dot(phi2, w)

plt.subplot(121)
plt.plot(x_ev, y_ev, 'r.-')
plt.plot(x_train, t_train, 'b.')
plt.plot(x_test, t_test, 'g.')
plt.legend(['fit polynomial', 'Training data', 'Testing data'])
plt.title('Visualization of linear regression function')
plt.xlabel(features[specific_feature]+" values \nfrom [min(x_train), max(x_train)]")
plt.ylabel("target values")

plt.subplot(122)
plt.plot(x_ev2, y_ev2, 'r.-')
plt.plot(x_train, t_train, 'b.')
plt.plot(x_test, t_test, 'g.')
plt.title('Visualization of linear regression function')
plt.legend(['fit polynomial', 'Training data', 'Testing data'])
plt.xlabel(features[specific_feature]+" values\n from [min(all data), max(all data)]")
plt.ylabel("target values")

plt.show()
