'''
File: LogisticRegression.py
Author: Yutong Dai (rothdyt@gmail.com)
File Created: 2019-06-27
Last Modified: 2019-07-01 14:53
--------------------------------------------
Description:
'''

import numpy as np
from numpy import linalg as LA


def get_gradient(X, Y, beta):
    """
    @Input:
        X: np.array of shape n * d; X.T = [x1, x2, ..., xn]
        Y: np.array of shape n * 1; Y.T = [y1, y2, ..., yn]
        beta: np.array of shape d * 1
    @Output:
        The gradient of the logistic loss.
    """
    Z = Y * (np.matmul(X, beta))
    sigma = (-Y) * (1 / (1 + np.exp(Z)))
    return np.matmul(X.T, sigma)


def get_fval(X, Y, beta):
    """
    @Input:
        X: np.array of shape n * d; X.T = [x1, x2, ..., xn]
        Y: np.array of shape n * 1; Y.T = [y1, y2, ..., yn]
        beta: np.array of shape d * 1
    @Output:
        The value of the logistic loss.
    """
    Z = Y * (np.matmul(X, beta))
    temp = np.log(1 + np.exp(-Z))
    return np.sum(temp)


def LogisticRegression(X, Y, stepsize, maxiter=100):
    n, d = X.shape
    beta = np.zeros((d, 1))
    iteration = 0
    gradient_norm = [LA.norm(get_gradient(X, Y, beta)/n)]
    fval = [get_fval(X, Y, beta)/n]
    beta_norm = [LA.norm(beta)]
    while iteration < maxiter:
        gradient = get_gradient(X, Y, beta) / n
        beta -= stepsize * gradient
        gradient_norm.append(LA.norm(gradient))
        beta_norm.append(LA.norm(beta))
        fval.append(get_fval(X, Y, beta) / n)
        iteration += 1
    return beta, fval, gradient_norm, beta_norm


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    # non-separable
    X1 = np.array([[0.3, 0.9], [0.5, 1.5]])
    Y1 = np.array([[1], [-1]])
    stepsize = 1 / LA.norm(X1) ** 2
    beta1, fval1, gradient_norm1, beta_norm1 = LogisticRegression(X1, Y1, stepsize, maxiter=500)
    print("Estimated Beta is: {}".format(beta1))
    print("Estimated Slope is: {}".format(- beta1[0] / beta1[1]))

    # separable
    X2 = np.array([[2, 1], [-1, -1]])
    Y2 = np.array([[1], [-1]])
    stepsize = 1 / LA.norm(X2) ** 2
    beta2, fval2, gradient_norm2, beta_norm2 = LogisticRegression(X2, Y2, stepsize, maxiter=5000)
    print("Estimated Beta is {}".format(beta2))
    print("Estimated Slope is: {}".format(- beta2[0] / beta2[1]))

    fig = plt.figure()
    ax = fig.add_subplot(2, 2, 1)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    ax.plot(range(len(gradient_norm1)), gradient_norm1, color='blue', lw=2)
    ax.set_yscale('log')
    ax.set_ylabel(r'$||\nabla l(\beta)||$' + ' (log scale)')
    ax.set_xlabel('iteration')
    ax = fig.add_subplot(2, 2, 2)
    ax.plot(range(len(fval1)), fval1, color='red', lw=2)
    ax.set_ylabel(r'$l(\beta)$')
    ax.set_xlabel('iteration')

    ax = fig.add_subplot(2, 2, 3)
    ax.plot(range(len(gradient_norm2)), gradient_norm2, color='blue', lw=2)
    ax.set_yscale('log')
    ax.set_ylabel(r'$||\nabla l(\beta)||$' + ' (log scale)')
    ax.set_xlabel('iteration')
    ax = fig.add_subplot(2, 2, 4)
    ax.plot(range(len(fval2)), fval2, color='red', lw=2)
    ax.set_yscale('log')
    ax.set_ylabel(r'$l(\beta)$' + ' (log scale)')
    ax.set_xlabel('iteration')
    plt.savefig('../figs/LogisticRegression.png', dpi=300)

    fig = plt.figure()
    ax = fig.add_subplot(2, 1, 1)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    ax.plot(range(len(beta_norm1)), beta_norm1, lw=2)
    ax.set_ylabel(r'$||\beta||$')
    ax.set_xlabel('iteration')

    ax = fig.add_subplot(2, 1, 2)
    ax.plot(range(len(beta_norm2)), beta_norm2, lw=2)
    ax.set_ylabel(r'$||\beta\||$')
    ax.set_xlabel('iteration')
    plt.savefig('../figs/LogisticRegression-iterates.png', dpi=300)
