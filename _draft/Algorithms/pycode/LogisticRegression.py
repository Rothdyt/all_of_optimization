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
    while iteration < maxiter:
        gradient = get_gradient(X, Y, beta) / n
        beta -= stepsize * gradient
        gradient_norm.append(LA.norm(gradient))
        fval.append(get_fval(X, Y, beta) / n)
        iteration += 1
    return beta, fval, gradient_norm


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    X1 = np.array([[1, 0], [2, 0]])
    Y1 = np.array([[1], [-1]])
    stepsize = 1 / LA.norm(X1) ** 2
    beta1, fval1, gradient_norm1 = LogisticRegression(X1, Y1, stepsize, maxiter=1000)
    print("Estimated Beta is {}".format(beta1))
    X2 = np.array([[1, 0], [-1, 0]])
    Y2 = np.array([[1], [-1]])
    stepsize = 1 / LA.norm(X2) ** 2
    beta2, fval2, gradient_norm2 = LogisticRegression(X2, Y2, stepsize, maxiter=1000)
    print("Estimated Beta is {}".format(beta2))

    fig = plt.figure()
    ax = fig.add_subplot(2, 2, 1)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    ax.plot(range(len(gradient_norm1)), gradient_norm1, color='blue', lw=2)
    ax.set_yscale('log')
    ax.set_ylabel(r'$\nabla l(\beta)$' + ' (log scale)')
    ax.set_xlabel('iteration')
    ax = fig.add_subplot(2, 2, 2)
    ax.plot(range(len(fval1)), fval1, color='red', lw=2)
    ax.set_ylabel(r'$l(\beta)$')
    ax.set_xlabel('iteration')

    ax = fig.add_subplot(2, 2, 3)
    ax.plot(range(len(gradient_norm2)), gradient_norm2, color='blue', lw=2)
    ax.set_yscale('log')
    ax.set_ylabel(r'$\nabla l(\beta)$' + ' (log scale)')
    ax.set_xlabel('iteration')
    ax = fig.add_subplot(2, 2, 4)
    ax.plot(range(len(fval2)), fval2, color='red', lw=2)
    ax.set_ylabel(r'$l(\beta)$')
    ax.set_xlabel('iteration')
    plt.savefig('../figs/LogisticRegression.jpg', dpi=300)
    plt.show()
