from __future__ import division
from __future__ import print_function

import numpy as np
import scipy.sparse as sp
from lightfm import LightFM
from lightfm.evaluation import precision_at_k
from sklearn.base import BaseEstimator, ClassifierMixin


class LightWrapper(BaseEstimator, ClassifierMixin):
    """
    lightFM model wrapped as an sklearn classifier 
    """
    def __init__(self, no_components=10, k=5, n=10, learning_schedule='adagrad', loss='wrap', learning_rate=0.05,
                 rho=0.95, epsilon=1e-06, item_alpha=1e-05, user_alpha=1e-05, max_sampled=10, random_state=None,
                 epochs=10, user_features=None, item_features=None, verbose=False, num_threads=1, users=None, items=None):

        self.usr_ftrs = user_features
        self.itm_ftrs = item_features
        self.epochs = epochs
        self.verbose = verbose
        self.num_threads = num_threads
        self.users = users
        self.items = items

        # sklearn hacks
        self.random_state = random_state
        self.max_sampled = max_sampled
        self.user_alpha = user_alpha
        self.item_alpha = item_alpha
        self.epsilon = epsilon
        self.rho = rho
        self.learning_rate = learning_rate
        self.loss = loss
        self.n = n
        self.k = k
        self.learning_schedule = learning_schedule
        self.no_components = no_components

        self.model = LightFM(no_components, k, n, learning_schedule, loss, learning_rate, rho, epsilon,
                             user_alpha, user_alpha, max_sampled, random_state)

    def fit(self, X):

        X = self.convert(X)
        self.model.fit(X, epochs=self.epochs, num_threads=self.num_threads, verbose=self.verbose)

        # hacking sklearn score interfaces
        self.train_interactions = X

    def predict(self, X):

        user_ids = X.ip.astype('category', categories=self.users).cat.codes
        item_ids = X.vid.astype('category', categories=self.items).cat.codes

        return self.model.predict(user_ids, item_ids, num_threads=self.num_threads)

    def score(self, X, **kwargs):

        test_interactions = self.convert(X)
        test_precision = precision_at_k(self.model, test_interactions, self.train_interactions, k=10, num_threads=self.num_threads).mean()
        return test_precision

    def convert(self, frame):

        """
        converts pandas dataframe to coo sparse matrix
        :param frame: 
        :return: 
        """
        row = frame.ip.astype('category', categories=self.users).cat.codes
        col = frame.vid.astype('category', categories=self.items).cat.codes

        mat = sp.lil_matrix((len(self.users), len(self.items)), dtype=np.int32)
        mat[row, col] = 1

        return mat.tocoo()
