import numpy as np

class LR(object):

    def __init__(self):
        pass

    def h(self, w, x):
        """
            params:
                w: [np_array] a vector of weights with dimensions (nx1), where n represents the number of weights.
                x: [np_array] a vector of feature variables with dimensions (nxm),
                    where n represents the number of feature variables and m the number of training examples
            returns:
                h: [double] the estimation performed by the linear model h=w'*x
        """
        return w.T.dot(x)

    def J(self, w, x, y):
        """
            params:
                w: [np_array] a vector of weights with dimensions (nx1), where n represents the number of weights.
                x: [np_array] a vector of feature variables with dimensions (nxm),
                    where n represents the number of feature variables and m the number of training examples
                y: [np_array] a vector of feature variables with dimensions (mx1),
                    where m represents the number of target variables
            returns:
                cost: [double] the mean squared error
        """
        return (1/(2*x.shape[1])) * (np.sum(np.square(self.h(w, x).T-y)))

    def dJ(self, w, x, y):
        """
            params:
                w: [np_array] a vector of weights with dimensions (nx1), where n represents the number of weights.
                x: [np_array] a vector of feature variables with dimensions (nxm),
                    where n represents the number of feature variables and m the number of training examples
                y: [np_array] a vector of feature variables with dimensions (mx1),
                    where m represents the number of target variables
            returns:
                dJ: [double] the derivative of the mean squared error
        """
        e = self.h(w, x).T - y
        return (1 / (x.shape[1])) * np.dot(x, e)

    def optimizar_LMS(self, x, y, num_iter, alpha, w=None):
        """
        We calculate gradient descent for minimizing the MSE to obtain the best linear hypothesis.
            params:
                x: [np_array] a vector of feature variables with dimensions (nxm),
                    where n represents the number of feature variables and m the number of training examples
                y: [np_array] a vector of feature variables with dimensions (mx1),
                    where m represents the number of target variables
                num_iter: [int] an integer indicating the number of iterations of the Gradient Descent algorithm
                alpha: [double] learning rate constant specifying the magnitude update step
                w: [np_array] vector that contains the initial weights to start optimzing the model with dimensions (n x 1)

            return:
                j: [np_array] a vector (num_iter x 1) containing all cost function evaluations during training
                w: [np_array] a vector of the final optimized weights with dimensions (nx1)
        """

        if w is None:
            # Inicializamos los pesos aleatoriamente
            w = np.random.randn(x.shape[0], 1)

        # se generan los vectores
        it = np.arange(0, num_iter)
        j = np.zeros(num_iter)

        # Se optimiza el modelo por el numero de iteraciones
        for i in range(num_iter):

            # Actualizamos los pesos
            w = w - alpha * self.dJ(w, x, y)

            # Guardamos el costo
            j[i] = self.J(w, x, y)

        return w, j

    def optimizar_NE(self, x, y):
        """
            params:
                x: [np_array] a vector of feature variables with dimensions (mxn),
                    where n represents the number of feature variables and m the number of training examples
                y: [np_array] a vector of feature variables with dimensions (nx1),
                    where m represents the number of target variables

            return:
                w: [np_array] a vector of the final optimized weights with dimensions (nx1)
        """
        return np.linalg.inv(x.T.dot(x)).dot(x.T).dot(y)