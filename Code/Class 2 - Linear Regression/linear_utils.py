import numpy as np


class LR(object):

    def __init__(self):
        pass

    @staticmethod
    def norm(arr):
        return (arr - min(arr)) / (max(arr) - min(arr))

    @staticmethod
    def denormalize(arr, minimum, maximum):
        return arr * (maximum - minimum) + minimum

    @staticmethod
    def h(w, x):
        """
            params:
                w: [np_array] a vector of weights with dimensions (nx1), where n represents the number of weights.
                x: [np_array] a vector of feature variables with dimensions (nxm),
                    where n represents the number of feature variables and m the number of training examples
            returns:
                h: [double] the estimation performed by the linear model h=w'*x
        """
        return w.T.dot(x)

    def cost(self, w, x, y, l2=0):
        """
            params:
                w: [np_array] a vector of weights with dimensions (nx1), where n represents the number of weights.
                x: [np_array] a vector of feature variables with dimensions (nxm),
                    where n represents the number of feature variables and m the number of training examples
                y: [np_array] a vector of feature variables with dimensions (mx1),
                    where m represents the number of target variables
                l2: [double] regularization constant for Ridge Regression

            returns:
                cost: [double] the mean squared error
        """
        return (1/(2*x.shape[1])) * (np.sum(np.square(self.h(w, x).T-y))) + l2*(np.sum(np.square(w)))

    def grad(self, w, x, y, l2=0):
        """
            params:
                w: [np_array] a vector of weights with dimensions (nx1), where n represents the number of weights.
                x: [np_array] a vector of feature variables with dimensions (nxm),
                    where n represents the number of feature variables and m the number of training examples
                y: [np_array] a vector of feature variables with dimensions (mx1),
                    where m represents the number of target variables
                l2: [double] regularization constant for Ridge Regression

            returns:
                grad: [double] the derivative of the mean squared error
        """
        e = self.h(w, x).T - y
        return (1 / (x.shape[1])) * np.dot(x, e) + l2*w

    def fit_gd(self, x_train, y_train, x_test, y_test, num_iter, alpha, w=None, l2=0):
        """
            We calculate gradient descent for minimizing the MSE to obtain the best linear hypothesis.

            params:
                x_train: [np_array] a vector of feature variables with dimensions (nxm),
                          where n represents the number of feature variables and m the number of training examples
                x_test: [np_array] a vector of feature variables with dimensions (nxm),
                          where n represents the number of feature variables and m the number of validation examples
                y_train: [np_array] a vector of feature variables with dimensions (mx1),
                    where m represents the number of target training variables
                y_test: [np_array] a vector of feature variables with dimensions (mx1),
                    where m represents the number of target validation variables
                num_iter: [int] an integer indicating the number of iterations of the Gradient Descent algorithm
                alpha: [double] learning rate constant specifying the magnitude update step
                w: [np_array] vector that contains the initial weights to optimize the model with dimensions (n x 1)
                l2: [double] regularization constant for Ridge Regression

            return:
                j_train: [np_array] a vector (num_iter x 1) containing all cost function evaluations during training
                j_test: [np_array] a vector (num_iter x 1) containing all cost function evaluations during evaluation
                w: [np_array] a vector of the final optimized weights with dimensions (nx1)
        """

        if w is None:
            # Random weight initialization
            w = np.random.randn(x_train.shape[0], 1)

        # Vectors to save all costs at each iteration
        j_train = np.zeros(num_iter)
        j_test = np.zeros(num_iter)

        # Iterate over the specified range
        for i in range(num_iter):

            # Update weights using Gradient Descent
            w = w - alpha * self.grad(w, x_train, y_train, l2=l2)

            # Save training cost
            j_train[i] = self.cost(w, x_train, y_train, l2=l2)

            # Save validation cost
            j_test[i] = self.cost(w, x_test, y_test, l2=l2)

        return w, j_train, j_test

    @staticmethod
    def fit_ne(phi, y, l2=0):
        """
            params:
                x: [np_array] a vector of feature variables with dimensions (mxn),
                    where n represents the number of feature variables and m the number of training examples
                y: [np_array] a vector of feature variables with dimensions (nx1),
                    where m represents the number of target variables

            return:
                w: [np_array] a vector of the final optimized weights with dimensions (nx1)
        """
        return np.linalg.inv(phi.T.dot(phi) + l2*np.identity(phi.shape[1])).dot(phi.T).dot(y)

    @staticmethod
    def identity_basis_function(x):
        return x

    @staticmethod
    def gaussian_basis_function(x, mu, sigma=0.1):
        return np.exp(-0.5 * (x - mu) ** 2 / sigma ** 2)

    @staticmethod
    def polynomial_basis_function(x, degree):
        return x ** degree

    @staticmethod
    def expand(x, bf, bf_args=None):
        if bf_args is None:
            return np.concatenate([np.ones(x.shape), bf(x)], axis=1)
        else:
            return np.concatenate([np.ones(x.shape)] + [bf(x, bf_arg) for bf_arg in bf_args], axis=1)