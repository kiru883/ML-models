import numpy as np


class LinearRegression:
    def __init__(self, epochs='default', batch_size='default', learning_rate='default', l2=0.1, optimizer="ORIG"):
        # original optimizers with default settings
        self.__OPTIMIZERS = {
            "SGD": {"epochs": 10, "batch_size": 10, "learning_rate": 0.1, "object": self.__SGD},
            "ORIG": {"epochs": 1, "batch_size": "all", "learning_rate": "full", "object": self.__ORIG}
        }

        # set oprimizer parameters
        if optimizer not in self.__OPTIMIZERS.keys():
            raise("Bad optimizer type!")
        else:
            self.optimizer = self.__OPTIMIZERS[optimizer]
            if epochs != 'default':
                self.optimizer['epochs'] = epochs
            if batch_size != 'default':
                self.optimizer['batch_size'] = batch_size
            if learning_rate != 'default':
                self.optimizer['learning_rate'] = learning_rate

        # weights is none if model is untrained
        self.weights = None

        # set l2 regularizator coeficient
        self.__l2 = l2

    # train method
    def fit(self, X, y):
        # column for bias
        X = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
        # set weights
        self.weights = np.random.random((1, X.shape[1]))

        batch_size = X.shape[0] if self.optimizer['batch_size'] == 'all' else self.optimizer['batch_size']
        for epoch in range(self.optimizer['epochs']):
            X_batch, y_batch = X[epoch*batch_size:(epoch+1)*batch_size], y[epoch*batch_size:(epoch+1)*batch_size]

            # if solve is optimization method
            if self.optimizer['learning_rate'] != 'full':
                self.weights += self.optimizer['learning_rate'] * self.optimizer['object'](X_batch, y_batch) \
                                + self.__l2 * self.__L2_regularizator()
            # if analytical solve
            else:
                self.weights = self.optimizer['object'](X_batch, y_batch)

    # predict method
    def predict(self, X):
        # column for bias
        X = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
        return np.matmul(X, np.transpose(self.weights))

    # optimizers
    # analytical solve of linear regression
    def __ORIG(self, X, y):
        xtx = np.linalg.inv(np.matmul(np.transpose(X), X))
        weights = np.matmul(np.matmul(xtx, np.transpose(X)), y)
        return np.transpose(weights)

    # gradient descent
    def __SGD(self, X, y):
        dldy = np.matmul(X, np.transpose(self.weights)) - y
        grad_weights_avg = np.sum((X * dldy), axis=0) / X.shape[0]
        return -grad_weights_avg

    def __L2_regularizator(self):
        return

