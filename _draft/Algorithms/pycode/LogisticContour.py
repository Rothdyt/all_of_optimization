'''
File: LogisticContour.py
Author: Yutong Dai (rothdyt@gmail.com)
File Created: 2019-07-01 09:04
Last Modified: 2019-07-01 15:21
--------------------------------------------
Description: Contour plot for logistic regression.
'''
import numpy as np
from numpy import linalg as LA
from copy import deepcopy


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
    beta = np.ones((d, 1))
    iteration = 0
    fval = [get_fval(X, Y, beta)/n]
    beta_seq = [deepcopy(beta)]
    while iteration < maxiter:
        gradient = get_gradient(X, Y, beta) / n
        beta -= stepsize * gradient
        fval.append(get_fval(X, Y, beta) / n)
        beta_seq.append(deepcopy(beta))
        iteration += 1
    return beta_seq, fval


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    # 1-d case
    X1 = np.array([[1], [2]])
    Y1 = np.array([[1], [-1]])
    betas = np.linspace(-10, 10, 1000).reshape(-1, 1)
    fval1 = [get_fval(X1, Y1, beta.reshape(-1, 1)) / X1.shape[0] for beta in betas]
    X2 = np.array([[1], [-1]])
    Y2 = np.array([[1], [-1]])
    fval2 = [get_fval(X2, Y2, beta.reshape(-1, 1)) / X2.shape[0] for beta in betas]

    fig = plt.figure(figsize=(5, 2))
    ax = fig.add_subplot(1, 2, 1)
    plt.subplots_adjust(wspace=0.4,)
    ax.plot(betas, fval1, color='blue', lw=2)
    ax.set_ylabel('fvalue')
    ax.set_xlabel(r'$\beta$')
    ax = fig.add_subplot(1, 2, 2)
    ax.plot(betas, fval2, color='red', lw=2)
    ax.set_ylabel('fvalue')
    ax.set_xlabel(r'$\beta$')
    plt.savefig('../figs/LogisticRegression-1dcontour.png', dpi=200)

    # 2-d case

    # non-separable
    fig = plt.figure(figsize=(8, 8))
    plt.subplots_adjust(wspace=0.4, hspace=0.4)

    X1 = np.array([[0.3, 0.9], [0.5, 1.5]])
    Y1 = np.array([[1], [-1]])
    stepsize = (1 / LA.norm(X1)) ** 2
    beta_seq1, fval1 = LogisticRegression(X1, Y1, stepsize, maxiter=100)
    beta_seq1 = np.array(beta_seq1)

    # (2,2,1)
    beta1_range = np.linspace(-2, 4, 30).reshape(-1, 1)
    beta2_range = np.linspace(-2, 4, 30).reshape(-1, 1)
    BETA1, BETA2 = np.meshgrid(beta1_range, beta2_range)
    FGRID = np.zeros_like(BETA1)
    for i in range(BETA1.shape[0]):
        for j in range(BETA1.shape[1]):
            beta_temp = np.array([BETA1[i, j], BETA2[i, j]]).reshape(-1, 1)
            FGRID[i, j] = get_fval(X1, Y1, beta_temp)
    ax = fig.add_subplot(2, 2, 1)
    contours = ax.contour(BETA1, BETA2, FGRID, 15, cmap='autumn')
    ax.set_xlabel(r'$\beta_1$')
    ax.set_ylabel(r'$\beta_2$')
    ax.plot(beta_seq1[:, 0], beta_seq1[:, 1], 'b.-')
    ax.clabel(contours, inline=True, fontsize=10)

    # (2,2,2)
    beta1_range = np.linspace(0.25, 1, 30).reshape(-1, 1)
    beta2_range = np.linspace(-0.5, 1, 30).reshape(-1, 1)
    BETA1, BETA2 = np.meshgrid(beta1_range, beta2_range)
    FGRID = np.zeros_like(BETA1)
    for i in range(BETA1.shape[0]):
        for j in range(BETA1.shape[1]):
            beta_temp = np.array([BETA1[i, j], BETA2[i, j]]).reshape(-1, 1)
            FGRID[i, j] = get_fval(X1, Y1, beta_temp)
    ax = fig.add_subplot(2, 2, 2)
    contours = ax.contour(BETA1, BETA2, FGRID, 15, cmap='autumn')
    ax.set_xlabel(r'$\beta_1$')
    ax.set_ylabel(r'$\beta_2$')
    ax.plot(beta_seq1[:, 0], beta_seq1[:, 1], 'b.-')
    ax.clabel(contours, inline=True, fontsize=10)

    # separable
    X2 = np.array([[1, 0], [-1, -1]])
    Y2 = np.array([[1], [-1]])
    stepsize = (1 / LA.norm(X2)) ** 2
    beta_seq2, fval2 = LogisticRegression(X2, Y2, stepsize, maxiter=100)
    beta_seq2 = np.array(beta_seq2)

    # (2,2,3)
    beta1_range = np.linspace(-2, 4, 30).reshape(-1, 1)
    beta2_range = np.linspace(-2, 4, 30).reshape(-1, 1)
    BETA1, BETA2 = np.meshgrid(beta1_range, beta2_range)
    FGRID = np.zeros_like(BETA1)
    for i in range(BETA1.shape[0]):
        for j in range(BETA1.shape[1]):
            beta_temp = np.array([BETA1[i, j], BETA2[i, j]]).reshape(-1, 1)
            FGRID[i, j] = get_fval(X2, Y2, beta_temp)
    ax = fig.add_subplot(2, 2, 3)
    contours = ax.contour(BETA1, BETA2, FGRID, 30, cmap='autumn')
    ax.set_xlabel(r'$\beta_1$')
    ax.set_ylabel(r'$\beta_2$')
    ax.plot(beta_seq2[:, 0], beta_seq2[:, 1], 'b.-')
    ax.clabel(contours, inline=True, fontsize=10)

    # (2,2,4)
    beta1_range = np.linspace(1, 4, 30).reshape(-1, 1)
    beta2_range = np.linspace(1, 1.5, 30).reshape(-1, 1)
    BETA1, BETA2 = np.meshgrid(beta1_range, beta2_range)
    FGRID = np.zeros_like(BETA1)
    for i in range(BETA1.shape[0]):
        for j in range(BETA1.shape[1]):
            beta_temp = np.array([BETA1[i, j], BETA2[i, j]]).reshape(-1, 1)
            FGRID[i, j] = get_fval(X2, Y2, beta_temp)
    ax = fig.add_subplot(2, 2, 4)
    contours = ax.contour(BETA1, BETA2, FGRID,
                          levels=[0, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5],
                          cmap='autumn')
    ax.set_xlabel(r'$\beta_1$')
    ax.set_ylabel(r'$\beta_2$')
    ax.plot(beta_seq2[:, 0], beta_seq2[:, 1], 'b.-')
    ax.clabel(contours, inline=True, fontsize=10)
    plt.savefig('../figs/LogisticRegression-2dcontour.png', dpi=300)
