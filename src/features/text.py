import logging
import six
from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import VectorizerMixin

import spacy
import pandas as pd
import numpy as np
import scipy.sparse as sp
import multiprocessing




log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_fmt)
logger = logging.getLogger(__name__)

class GloveVectorizer(BaseEstimator, VectorizerMixin):
    """Convert a collection of text documents to a matrix of averaged token glove vectors
        This implementation produces a sparse representation of the vectors using
        scipy.sparse.coo_matrix and spacy NLP toolkit.
        """
    def __init__(self, rows, embedding_size = 300):
        self.shape = (rows, embedding_size)
        self.embedding_size = embedding_size
        self.nlp = spacy.load('en')

    def fit(self, raw_documents, y=None):
        """Learn a vocabulary dictionary of all tokens in the raw documents.
        Parameters
        ----------
        raw_documents : iterable
            An iterable which yields either str, unicode or file objects.
        Returns
        -------
        self
        """
        self.fit_transform(raw_documents)
        return self

    def fit_transform(self, raw_documents, y=None): #TODO - work in progress - need to chenge to iterable
        """Learn the vocabulary dictionary and return term-document matrix.
        This is equivalent to fit followed by transform, but more efficiently
        implemented.
        Parameters
        ----------
        raw_documents : iterable
            An iterable which yields either iterable.
        Returns
        -------
        X : array, [n_samples, n_features]
            Document-term matrix.
        """
        # We intentionally don't call the transform method to make
        # fit_transform overridable without unwanted side effects in
        # TfidfVectorizer.
        if isinstance(raw_documents, six.string_types):
            raise ValueError(
                "Iterable over raw text documents expected, "
                "string object received.")

        fields = []
        for field, field_name, indices in raw_documents:
            logger.info('transforming field %s' % field_name)
            data = np.zeros(self.shape, dtype=np.float32)
            vectors = self.transform(field)
            data[indices] = vectors
            fields.append(data)

        data = np.hstack(fields)
        mat = sp.csr_matrix(data)
        logger.info('generated mat: %s' % str(mat.shape))

        return mat

    def transform(self, raw_documents):
        """Transform documents to document-term matrix.
        Extract token vectors out of raw text documents using the Glove vocabulary.
        Parameters
        ----------
        raw_documents : iterable
            An iterable which yields either str, unicode or file objects.
        Returns
        -------
        X : sparse matrix, [n_samples, n_features]
            Document-term matrix.
        """
        if isinstance(raw_documents, six.string_types):
            raise ValueError(
                "Iterable over raw text documents expected, "
                "string object received.")

        return [doc.vector for doc in self.nlp.pipe(raw_documents, parse=False, entity=False, batch_size=10000, n_threads=multiprocessing.cpu_count())]

    def inverse_transform(self, X):
        """Return terms per document with nonzero entries in X.
        Parameters
        ----------
        X : {array, sparse matrix}, shape = [n_samples, n_features]
        Returns
        -------
        X_inv : list of arrays, len = n_samples
            List of arrays of terms.
        """
        raise Exception('Not Implemented')


