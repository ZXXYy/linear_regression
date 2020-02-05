#!/usr/bin/env python

import assignment1 as a1
import numpy as np
import matplotlib.pyplot as plt

# Plot a curve showing learned function.

(countries, features, values) = a1.load_unicef_data()
x = values[:, :]
targets = values[:, 1]
# x = a1.normalize_data(x)
N_TRAIN = 100
x_train = x[0:N_TRAIN, :]
x_test = x[N_TRAIN:, :]
t_train = targets[0:N_TRAIN]
t_test = targets[N_TRAIN:]
# change f to get degree 3 polynomials for different features
f = 10
# Use linspace to get a set of samples on which to evaluate
x_ev = np.linspace(np.ndarray.item(min(x_train[:, f])), np.ndarray.item(max(x_train[:, f])), num=500)
x_ev2 = np.linspace(np.ndarray.item(min(min(x_train[:, f]), min(x_test[:, f]))),
                   np.ndarray.item(max(max(x_train[:, f]), max(x_test[:, f]))), num=500)
x_ev = x_ev.reshape((500, 1))
x_ev2 = x_ev2.reshape((500, 1))

# TO DO::
# Perform regression on the linspace samples.
(w, tr_err) = a1.linear_regression(x_train[:, f], t_train, 'polynomial', reg_lambda=0, degree=3, bias=1)
# print(w)
phi = a1.design_matrix(x_ev, degree=3, basis='polynomial', bias=1)
phi2 = a1.design_matrix(x_ev2, degree=3, basis='polynomial', bias=1)
# Put your regression estimate here in place of y_ev.
y_ev = np.dot(phi, w)
y_ev2 = np.dot(phi2, w)

# Produce a plot of results.
plt.subplot(121)
plt.plot(x_ev, y_ev, 'r.-')
plt.plot(x_train[:, f], t_train, 'b.')
plt.plot(x_test[:, f], t_test, 'g.')
plt.legend(['fit polynomial', 'Training data', 'Testing data'])
plt.title('Visualization of linear regression function')
plt.xlabel(features[f]+"values \nfrom [min(x_train), max(x_train)]")
plt.ylabel("target values")

plt.subplot(122)
plt.plot(x_ev2, y_ev2, 'r.-')
plt.plot(x_train[:, f], t_train, 'b.')
plt.plot(x_test[:, f], t_test, 'g.')
plt.title('Visualization of linear regression function')
plt.legend(['fit polynomial', 'Training data', 'Testing data'])
plt.xlabel(features[f]+" values\n from [min(all data), max(all data)]")
plt.ylabel("target values")
plt.show()
