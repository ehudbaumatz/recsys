from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import VectorizerMixin


class GloveVectorizer(BaseEstimator, VectorizerMixin):
    pass