"""Basic code for assignment 1."""

import numpy as np
import pandas as pd
from scipy import nanmean
import math


def load_unicef_data():
    """Loads Unicef data from CSV file.

    Retrieves a matrix of all rows and columns from Unicef child mortality
    dataset.

    Args:
      none

    Returns:
      Country names, feature names, and matrix of values as a tuple (countries, features, values).

      countries: vector of N country names
      features: vector of F feature names
      values: matrix N-by-F
    """
    fname = 'SOWC_combined_simple.csv'

    # Uses pandas to help with string-NaN-numeric data.
    data = pd.read_csv(fname, na_values='_', encoding='latin1')
    # Strip countries title from feature names.
    features = data.axes[1][1:]
    # Separate country names from feature values.
    countries = data.values[:, 0]
    values = data.values[:, 1:]
    # Convert to numpy matrix for real.
    values = np.asmatrix(values, dtype='float64')

    # Modify NaN values (missing values).
    mean_vals = nanmean(values, axis=0)
    inds = np.where(np.isnan(values))
    values[inds] = np.take(mean_vals, inds[1])
    return (countries, features, values)


def normalize_data(x):
    """Normalize each column of x to have mean 0 and variance 1.
    Note that a better way to normalize the data is to whiten the data (decorrelate dimensions).  This can be done using PCA.

    Args:
      input matrix of data to be normalized

    Returns:
      normalized version of input matrix with each column with 0 mean and unit variance

    """
    mvec = x.mean(0)
    stdvec = x.std(axis=0)

    return (x - mvec) / stdvec


def linear_regression(x, t, basis, reg_lambda=0, degree=0, mu=0, s=1, bias=1):
    """Perform linear regression on a training set with specified regularizer lambda and basis

    Args:
      x is training inputs
      t is training targets
      reg_lambda is lambda to use for regularization tradeoff hyperparameter
      basis is string, name of basis to use
      degree is degree of polynomial to use (only for polynomial basis)
      mu,s are parameters of Gaussian basis

    Returns:
      w vector of learned coefficients
      train_err RMS error on training set
      """

    # Construct the design matrix.
    # Pass the required parameters to this function
    phi = design_matrix(x, degree=degree, mu=mu, s=s, basis=basis, bias=bias)
    # Learning Coefficients
    if reg_lambda > 0:
        # regularized regression
        i = reg_lambda * np.identity(phi.shape[1])
        regular = i + np.dot(phi.T, phi)
        w = np.dot(regular.I, phi.T)
        w = np.dot(w, t)
    else:
        # no regularization
        w = np.linalg.pinv(phi)
        w = np.dot(w, t)

    # Measure root mean squared error on training data.
    difference = np.dot(phi, w) - t
    square = np.multiply(difference, difference)
    train_err = square.sum() / x.shape[0]
    train_err = math.sqrt(train_err)

    return (w, train_err)


def design_matrix(x, degree=0, mu=list(), s=1, basis=None, bias=1):
    """ Compute a design matrix Phi from given input datapoints and basis.

    Args:
        x is training inputs
        degree is degree of polynomial to use (only for polynomial basis)
        mu,s are parameters of Gaussian basis
        bias is models bias term, 1 stands for models with bias term, 0 stands for models without bias term
    Returns:
      phi design matrix
    """
    if bias == 1:
        phi = np.ones((x.shape[0], 1))
    else:
        phi = np.empty((x.shape[0], 0))
    # design matrix for polynomial regression
    if basis == 'polynomial':
        x_change = x
        for i in range(degree):
            phi = np.hstack((phi, x_change))
            x_change = np.multiply(x, x_change)
    # design matrix for sigmoid regression
    elif basis == 'sigmoid':
        for i in range(len(mu)):
            mu_array = mu[i] * np.ones((x.shape[0], x.shape[1]))
            x_change = np.ones((x.shape[0], x.shape[1])) + np.exp((mu_array - x) / s)
            x_change = 1 / x_change
            phi = np.hstack((phi, x_change))
    else:
        assert (False), 'Unknown basis %s' % basis

    return phi


def evaluate_regression(x, t, w, basis, degree=0, mu=list(), s=1, bias=1):
    """Evaluate linear regression on a dataset.

    Args:
      x is training inputs
      t is training targets
      w is the trained coefficient
      basis is string, name of basis to use
      degree is degree of polynomial to use (only for polynomial basis)
      mu,s are parameters of Gaussian basis
      bias is models bias term, 1 stands for models with bias term, 0 stands for models without bias term
    Returns:
      t_est values of regression on inputs
      err RMS error on training set if t is not None
      """

    phi = design_matrix(x, degree=degree, mu=mu, s=s, basis=basis, bias=bias)
    # the estimated value for test data
    t_est = np.dot(phi, w)
    # compute for RMS
    difference = t_est - t
    square = np.multiply(difference, difference)
    err = square.sum() / x.shape[0]
    err = math.sqrt(err)

    return (t_est, err)
