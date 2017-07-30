import pandas as pd


def get_internal_data(interactions, items_raw_features=None, user_raw_features=None):

    df = pd.read_csv(interactions) if interactions else _sample_interactions()
    t, user_mapper, item_mapper = _transform(df)


    items = pd.read_csv(items_raw) if items_raw else _fetch_items_data(df.vid.unique())



def _fetch_items_data(ids):
    """
    fetch sample interactions data from memsql 
    :return: Dataframe
    """
    pass

def _sample_interactions(site='huffingtonpost.com'):
    """
    fetch sample interactions data from memsql 
    :return: Dataframe
    """
    pass

def _transform(df):
    users_u = list(sorted(df.ip.unique()))
    items_u = list(sorted(df.vid.unique()))

    row = df.ip.astype('category', categories=users_u).cat.codes.rename('user_id')
    user_mapper = pd.concat([df.ip, row],axis=1).drop_duplicates().sort_values(by='user_id')

    col = df.vid.astype('category', categories=items_u).cat.codes.rename('item_id')
    item_mapper = pd.concat([df.vid, col], axis=1).drop_duplicates().sort_values(by='item_id')

    t = pd.concat([row, col], axis=1)
    return t, user_mapper, item_mapper

df = pd.read_csv('/Users/baumatz/Documents/python/recsys/data/intermediate/2017.06.50.100.5000.csv')
t, user_mapper, item_mapper = _transform(df)
t.to_csv('/Users/baumatz/Documents/python/recsys/data/intermediate/2017.06.50.100.5000.transformed.csv', index=False)
item_mapper.to_csv('/Users/baumatz/Documents/python/recsys/data/raw/items_mapper.csv', index=False)
user_mapper.to_csv('/Users/baumatz/Documents/python/recsys/data/raw/user_mapper.csv', index=False)

# get_internal_data('/Users/baumatz/Documents/python/recsys/data/raw/data.csv')