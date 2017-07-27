from __future__ import unicode_literals # support py2
import numpy as np
import pandas as pd
import os
import scipy.sparse as sp
from features.text import GloveVectorizer


def load_interactions_dataset(path, format='coo', shuffle=True, usecols=None):

    if usecols is None:
        usecols = ['user_id', 'item_id']
    df = pd.read_csv(path, usecols=usecols)
    if 'rating' not in df.columns.values: df.loc[:, 'rating'] = 1
    shape = (df.user_id.unique().shape[0], df.item_id.unique().shape[0])
    if shuffle: df = df.sample(frac=1)
    train, test = split(df)

    if format == 'coo':
        train = to_sparse_matrix(train.user_id.values, train.item_id.values, train.rating.values, shape)
        test = to_sparse_matrix(test.user_id.values, test.item_id.values, test.rating.values, shape)
        return train, test
    elif format == 'ndarray':
        return df.values, train.index.values, test.index.values
    else:
        return df, train, test


def load_items_features(path, usecols=None, vectorization='glove', embedding_size=300):

    if usecols is None:
        usecols = ['item_id', 'video_title', 'description', 'keywords', 'iab_category_one', 'iab_category_two']
    df = pd.read_csv(path, usecols=usecols, index_col='item_id')  # hard coded - should throw since this is a must

    # prepare iterators
    itr = [([item[1] for item in field.dropna().iteritems()], name, field.dropna().index) for name, field in df.iteritems()]

    if vectorization == 'glove':
        vectorizer = GloveVectorizer(df.shape[0])

    return vectorizer.fit_transform(itr)

def split(df, test_samples=10):
    test = df.groupby('user_id', sort=False, as_index=False).head(test_samples)
    train = df[~df.isin(test)].dropna()

    return train, test


def load_movielens(folder):
    """
    load movielens onto dataframe
    :param folder: 
    :return: 
    """

    def transform(df, is_train):
        df.loc[:, 'user_id'] = df['user_id'] - 1
        df.loc[:, 'item_id'] = df['item_id'] - 1
        df.loc[:, 'is_train'] = is_train
        return df

    train = pd.read_csv(os.path.join(folder, 'ua.base'), sep='\t', header=None,
                        names=['user_id', 'item_id', 'rating', 'timestamp'], index_col=False)
    test = pd.read_csv(os.path.join(folder, 'ua.test'), sep='\t', header=None,
                       names=['user_id', 'item_id', 'rating', 'timestamp'], index_col=False)

    train = transform(train, True)
    test = transform(test, False)

    return pd.concat([train, test], ignore_index=True)


def to_sparse_matrix(row, col, label, shape, mat_format='coo'):
    """
    convert data frame to sparse matrix
    :param array: 
    :param shape: 
    :param mat_format: 
    :return: 
    """

    mat = sp.lil_matrix(shape, dtype=np.int32)
    mat[row, col] = label

    return mat.tocoo() if mat_format == "coo" else mat.tocsr()
