import numpy as np
from scipy.stats import randint as sp_randint
from sklearn.model_selection import RandomizedSearchCV
from time import time
from lightfm_model import LightWrapper


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
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")

def tune(df, param_dist):

    print('convert to sparse matrix')
    # specify parameters and distributions to sample from
    param_dist = {"epochs": [20, 30, 40, 50, 60],
                  "learning_rate": [0.005, 0.001, 0.01, 0.1, 0.2, 0.3],
                  "no_components": sp_randint(20, 200),
                  "user_alpha": [0.005, 0.001, 0.01, 0.1, 0.2, 0.3],
                  "max_sampled": [10, 20, 30, 40, 50]}

    users = list(sorted(df.ip.unique()))
    items = list(sorted(df.vid.unique()))

    clf = LightWrapper(loss='warp', num_threads=1, users=users, items=items)
    # run randomized search
    n_iter_search = 101
    random_search = RandomizedSearchCV(clf, param_distributions=param_dist, cv=cv,
                                       n_iter=n_iter_search, n_jobs=1, verbose=1000)

    start = time()
    random_search.fit(df)
    print("RandomizedSearchCV took %.2f seconds for %d candidates"
          " parameter settings." % ((time() - start), n_iter_search))
    report(random_search.cv_results_)

def convert(frame, users, items):

    row = frame.ip.astype('category', categories=users).cat.codes
    col = frame.vid.astype('category', categories=items).cat.codes

    mat = sp.lil_matrix((len(users), len(items)), dtype=np.int32)
    mat[row, col] = 1

    return mat.tocoo()


df = pd.read_csv('small_data.csv', usecols=['ip', 'vid'])
df['ranks'] = 1
# df.drop_duplicates(inplace=True)

test = df.groupby('ip', sort=False, as_index=False).head(5)
train = df[~df.isin(test)].dropna()

users = list(sorted(df.ip.unique()))
items = list(sorted(df.vid.unique()))

mat = convert(test, users, items)
print (mat)

# test_mat = convert(test, users, items)
#
#
# sdf = pd.SparseDataFrame(test_mat)
# arr = sdf.values.resize(mat.shape, refcheck=False)
# matrix = sp.coo_matrix(arr)
print (mat)



# clf = LightWrapper(loss='warp', num_threads=1, users=users, items=items)
#
# clf.fit(train)
# clf.score(test)


# users = list(sorted(df.ip.unique()))
# items = list(sorted(df.vid.unique()))
#
# row = df.ip.astype('category', categories=users).cat.codes.values
# col = df.vid.astype('category', categories=items).cat.codes.values
#
# mat = sp.csr_matrix((df.ranks.values.tolist(), (row, col)), dtype=np.int32)

# print('Users: {}, items: {}, shape: {}, dense: {}'.format(len(users), len(items), str(mat.shape), mat.getnnz()))


# mat, mat_row, mat_col = convert(df, users, items)
# train, train_row, train_col  = convert(train, users, items)
# test, test_row, test_col  = convert(test, users, items)

# tune(mat, [[(train_row, train_col), (test_row, test_col)]])
tune(df, [[train.index, test.index]])
# df.reset_index(drop=True).set_index(0, inplace=True)

# split train/val/test
# train, x = train_test_split(mat, test_size=0.1, random_state=0)
# y = sp.csr_matrix((x.data, x.indices, x.indptr), shape=(9, 9))
#
# clf = LightWrapper(loss='warp', shape=df.shape)
# print(df.shape)
# scores = cross_val_score(clf, mat, n_jobs=1)
# zeros = sp.csr_matrix((5,5), dtype=np.int32)
