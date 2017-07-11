from time import time

import logging
import numpy as np
from sklearn.model_selection import RandomizedSearchCV

logger = logging.getLogger(__name__)
log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_fmt)

def report(results, n_top=3):
    """
     Utility function to report best scores
    :param results: 
    :param n_top: 
    :return: 
    """
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            logger.info("Model with rank: {0}".format(i))
            logger.info("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            logger.info("Parameters: {0}".format(results['params'][candidate]))
            logger.info("")

def grid_search(clf, X, cv, **kwargs):
    pass

def random_search(clf, X, cv, **kwargs):

    # run randomized search
    random_search = RandomizedSearchCV(clf, cv=cv, param_distributions=kwargs.get('param_dist'),
                                       n_iter=kwargs.get('n_iter_search'), n_jobs=kwargs.get('n_jobs'), verbose=kwargs.get('verbose'))

    start = time()
    random_search.fit(X)
    logger.info("RandomizedSearchCV took %.2f seconds for %d candidates"
          " parameter settings." % ((time() - start), kwargs.get('n_iter_search')))
    report(random_search.cv_results_)