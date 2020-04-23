import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


class Bayesian:

    def __init__(self):
        pass

    @staticmethod
    def plot_data(x, t):
        plt.scatter(x, t, marker='o', c="k", s=20)

    @staticmethod
    def plot_truth(x, y, label='Truth'):
        plt.plot(x, y, 'k--', label=label)

    @staticmethod
    def plot_predictive(x, y, std, y_label='Prediction', std_label='Uncertainty', plot_xy_labels=True):
        y = y.ravel()
        std = std.ravel()

        plt.plot(x, y, label=y_label)
        plt.fill_between(x.ravel(), y + std, y - std, alpha = 0.5, label=std_label)

        if plot_xy_labels:
            plt.xlabel('x')
            plt.ylabel('y')

    @staticmethod
    def plot_posterior_samples(x, ys, plot_xy_labels=True):
        plt.plot(x, ys[:, 0], 'r-', alpha=0.5, label='Post. samples')
        for i in range(1, ys.shape[1]):
            plt.plot(x, ys[:, i], 'r-', alpha=0.5)

        if plot_xy_labels:
            plt.xlabel('x')
            plt.ylabel('y')

    @staticmethod
    def plot_posterior(mean, cov, w0, w1):
        resolution = 100

        grid_x = grid_y = np.linspace(-1, 1, resolution)
        grid_flat = np.dstack(np.meshgrid(grid_x, grid_y)).reshape(-1, 2)

        densities = stats.multivariate_normal.pdf(grid_flat, mean=mean.ravel(), cov=cov).reshape(resolution, resolution)
        plt.imshow(densities, origin='lower', extent=(-1, 1, -1, 1))
        plt.scatter(w0, w1, marker='x', c="r", s=20, label='Truth')

        plt.xlabel('w0')
        plt.ylabel('w1')

    @staticmethod
    def print_comparison(title, a, b, a_prefix='np', b_prefix='br'):
        print(title)
        print('-' * len(title))
        print(f'{a_prefix}:', a)
        print(f'{b_prefix}:', b)
        print()

    @staticmethod
    def posterior(phi, y, alpha, beta):
        """
            Computes the mean and covariance matrix of the posterior distribution by imposing
            an isotropic Gaussian prior with mean vector zero, covariance matrix (alpha^-1)*I
            and a Gaussian likelihood distribution with mean (w^T)*phi and variance beta^-1

            params:
                - phi: design matrix [numpy array of shape (m x k)]
                - y: vector of labels [numpy array of shape (m x 1)]
                - alpha: scalar value representing the precision of the prior
                - beta: scalar value representing the precision of the likelihood

            returns
                - m_N: mean of the posterior.
                -S_N: precision matrix of the posterior
        """
        S_N_inv = alpha * np.eye(phi.shape[1]) + beta * (phi.T.dot(phi))
        S_N = np.linalg.inv(S_N_inv)
        m_N = beta * S_N.dot(phi.T).dot(y)
        return m_N, S_N

    @staticmethod
    def posterior_predictive(phi, beta, m_N, S_N):
        """
            Computes the mean and variance of the posterior predictive distribution using
            the posterior distribution of the weights that was previously calculated.

            params:
                - phi: design matrix [numpy array of shape (m x k)]
                - m_N: mean of the posterior distribution of the weights
                - S_N: precision matrix of the posterior distribution of the weights
                - beta: scalar value representing the precision of the likelihood

            returns
                - u: mean of the posterior predictive
                -lambda_inv: variance of the posterior predictive
        """
        u = phi.dot(m_N)
        # Only compute variances (diagonal elements of covariance matrix)
        lambda_inv = 1 / beta + np.sum(phi.dot(S_N) * phi, axis=1)
        return u, lambda_inv