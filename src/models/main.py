import logging

import click
import numpy as np
import scipy.sparse as sp
import yaml
from dotenv import find_dotenv, load_dotenv

from lightfm import LightFM
from lightfm.evaluation import precision_at_k, auc_score

from lightfm_model import LightWrapper
from tune_model import random_search

from lightfm.datasets import fetch_movielens, fetch_stackexchange
import sys, os

sys.path.insert(0, os.path.abspath('..'))
from data.datasets import load_movielens, to_sparse_matrix, load_interactions_dataset

logger = logging.getLogger(__name__)

def evaluate_model(model, train, test, item_fetures=None, user_features=None):

    train_precision = precision_at_k(model, train, k=10, user_features=user_features, item_features=item_fetures).mean()
    test_precision = precision_at_k(model, test, train_interactions=train, k=10, user_features=user_features, item_features=item_fetures).mean()

    train_auc = auc_score(model, train, user_features=user_features, item_features=item_fetures).mean()
    test_auc = auc_score(model, test, train_interactions=train, user_features=user_features, item_features=item_fetures).mean()

    print('Precision: train %.2f, test %.2f.' % (train_precision, test_precision))
    print('AUC: train %.2f, test %.2f.' % (train_auc, test_auc))

    return train_precision, test_precision, train_auc, test_auc


##### CLI ######
@click.group()
@click.option('-cf', '--config', type=click.File('r'), default='../../docs/config.yaml', help='config path')
@click.pass_context
def cli(ctx, config):
    ctx.obj = yaml.load(config)


@cli.command(help='validate stuff')
@click.pass_context
@click.argument('data_home', type=click.Path(exists=True), default='../../data/raw/stackexchange')
def validate_item_features(ctx, data_home):

    data = fetch_stackexchange('crossvalidated',
                               test_set_fraction=0.1,
                               indicator_features=False,
                               tag_features=True, data_home=data_home)

    train = data['train']
    test = data['test']

    # Set the number of threads; you can increase this
    # ify you have more physical cores available.
    NUM_COMPONENTS = 30
    NUM_EPOCHS = 3
    ITEM_ALPHA = 1e-6

    # Let's fit a WARP model: these generally have the best performance.
    model = LightFM(loss='warp',
                    item_alpha=ITEM_ALPHA,
                    no_components=NUM_COMPONENTS)

    # Run 3 epochs and time it.
    model = model.fit(train, epochs=NUM_EPOCHS)

    train_auc = auc_score(model, train).mean()
    print('Collaborative filtering train AUC: %s' % train_auc)

    test_auc = auc_score(model, test, train_interactions=train).mean()
    print('Collaborative filtering test AUC: %s' % test_auc)

    # Set biases to zero
    model.item_biases *= 0.0

    test_auc = auc_score(model, test, train_interactions=train).mean()
    print('Collaborative filtering test AUC: %s' % test_auc)

    item_features = data['item_features']
    tag_labels = data['item_feature_labels']

    print('There are %s distinct tags, with values like %s.' % (item_features.shape[1], tag_labels[:3].tolist()))

    # Define a new model instance
    model = LightFM(loss='warp',
                    item_alpha=ITEM_ALPHA,
                    no_components=NUM_COMPONENTS)

    # Fit the hybrid model. Note that this time, we pass
    # in the item features matrix.
    model = model.fit(train,
                      item_features=item_features,
                      epochs=NUM_EPOCHS)

    # Don't forget the pass in the item features again!
    train_auc = auc_score(model,  train, item_features=item_features).mean()
    print('Hybrid training set AUC: %s' % train_auc)

    test_auc = auc_score(model,  test, train_interactions=train, item_features=item_features).mean()
    print('Hybrid test set AUC: %s' % test_auc)

@cli.command(help='validate stuff')
@click.pass_context
@click.argument('data_home', type=click.Path(exists=True), default='../../data/raw/stackexchange')
def validate_item_features(ctx, data_home):

    data = fetch_stackexchange('crossvalidated',
                               test_set_fraction=0.1,
                               indicator_features=False,
                               tag_features=True, data_home=data_home)

    train = data['train']
    test = data['test']

    # Set the number of threads; you can increase this
    # ify you have more physical cores available.
    NUM_COMPONENTS = 30
    NUM_EPOCHS = 3
    ITEM_ALPHA = 1e-6

    # Let's fit a WARP model: these generally have the best performance.
    model = LightFM(loss='warp',
                    item_alpha=ITEM_ALPHA,
                    no_components=NUM_COMPONENTS)

    # Run 3 epochs and time it.
    model = model.fit(train, epochs=NUM_EPOCHS)

    train_auc = auc_score(model, train).mean()
    print('Collaborative filtering train AUC: %s' % train_auc)

    test_auc = auc_score(model, test, train_interactions=train).mean()
    print('Collaborative filtering test AUC: %s' % test_auc)

    # Set biases to zero
    model.item_biases *= 0.0

    test_auc = auc_score(model, test, train_interactions=train).mean()
    print('Collaborative filtering test AUC: %s' % test_auc)

    item_features = data['item_features']
    tag_labels = data['item_feature_labels']

    print('There are %s distinct tags, with values like %s.' % (item_features.shape[1], tag_labels[:3].tolist()))

    # Define a new model instance
    model = LightFM(loss='warp',
                    item_alpha=ITEM_ALPHA,
                    no_components=NUM_COMPONENTS)

    # Fit the hybrid model. Note that this time, we pass
    # in the item features matrix.
    model = model.fit(train,
                      item_features=item_features,
                      epochs=NUM_EPOCHS)

    # Don't forget the pass in the item features again!
    train_auc = auc_score(model,  train, item_features=item_features).mean()
    print('Hybrid training set AUC: %s' % train_auc)

    test_auc = auc_score(model,  test, train_interactions=train, item_features=item_features).mean()
    print('Hybrid test set AUC: %s' % test_auc)


@cli.command(help='validate stuff')
@click.pass_context
@click.argument('data_home', type=click.Path(exists=True), default='../../data/raw/movielens')
def validate(ctx, data_home):

    # matrix creation validation
    df = load_movielens(data_home)
    dic = fetch_movielens(data_home, download_if_missing=True)

    train_o = dic['train']
    test_o = dic['test']

    train_df = df[df['is_train']]
    test_df = df[~df['is_train']]

    shape = (df.user_id.unique().shape[0], df.item_id.unique().shape[0])

    train_t = to_sparse_matrix(train_df.user_id.values, train_df.item_id.values, train_df.rating.values, shape)
    test_t = to_sparse_matrix(test_df.user_id.values, test_df.item_id.values, test_df.rating.values, shape)

    assert (train_o.shape == train_t.shape)
    assert (np.array_equal(test_o.diagonal(), test_t.diagonal()))

    model = LightFM(loss='warp')

    model.fit(train_o, epochs=10)
    train_precision, test_precision, train_auc, test_auc = evaluate_model(model, train_o, test_o)

    model.fit(train_t, epochs=10)
    train_precision_t, test_precision_t, train_auc_t, test_auc_t = evaluate_model(model, train_t, test_t)

    assert (abs(train_precision - train_precision_t) < 2)
    assert (abs(test_precision - test_precision_t) < 2)
    assert (abs(train_auc - train_auc_t) < 2)
    assert (abs(test_auc - test_auc_t) < 2)

    clf = LightWrapper(loss='warp', shape=shape)
    clf.fit(train_df[['user_id', 'item_id']].values, train_df.rating.values)
    train_precision_t, test_precision_t, train_auc_t, test_auc_t = clf.evaluate(test_df[['user_id', 'item_id']].values, test_df.rating.values)

    assert (abs(train_precision - train_precision_t) < 2)
    assert (abs(test_precision - test_precision_t) < 2)
    assert (abs(train_auc - train_auc_t) < 2)
    assert (abs(test_auc - test_auc_t) < 2)

    random_search(clf, df[['user_id', 'item_id', 'rating']].values, [[train_df.index.values, test_df.index.values]], param_dist={"epochs": [10], "learning_rate": [0.005]})


@cli.command(help='hyper-parameter tuning on a sklearn model')
@click.pass_context
@click.argument('input_file', type=click.Path(exists=True), default='../../data/raw/sample.csv')
@click.option('-m', '--model', default='lightfm', help='model to tune')
@click.option('-l', '--loss', default='warp', help='loss to optimize')
@click.option('-j', '--jobs', default=-1, help='number of threads')
@click.option('-i', '--iter', default=10, help='number of hyper parameter search iterations')
@click.option('-v', '--verbose', default=10, help='level of verbosity')
@click.option('-t', '--threads', default=1, help='model threads')
def tune(ctx, input_file, model, loss, jobs, iter, verbose, threads):

    df, train, test = load_interactions_dataset(input_file, format='pandas')
    users_count = df.user_id.unique().shape[0]
    items_count = df.item_id.unique().shape[0]
    logger.info('Users: {}, items: {}, train{}, test{}, model: {}, loss: {}, jobs: {}, iter: {}'.format(users_count, items_count, model, loss,
                                                                               jobs, iter, train.shape, test.shape))
    if model == 'lightfm':

        # for tuning we utilize sklearn parallelism, so using default one thread
        clf = LightWrapper(loss=loss, shape=(users_count, items_count), num_threads=threads)
        cfg = ctx.obj.get('tune_group')[model]
        random_search(clf, df.values, [[train.index.values, test.index.values]], param_dist=cfg.get('param_dist'),
                      n_iter_search=iter, n_jobs=jobs, verbose=verbose)
    else:
        raise Exception('only lightfm supported currently')


@cli.command(help='training recommender systems')
@click.pass_context
@click.argument('input_file', type=click.Path(exists=True), default='../../data/processed/aol_com/06/50.100.5000')
@click.option('-m', '--model', default='lightfm', help='model to train')
@click.option('-l', '--loss', default='warp', help='loss to optimize')
@click.option('-v', '--verbose', default=10, help='level of verbosity')
@click.option('-t', '--threads', default=1, help='model threads')
def train(ctx, input_file, model, loss, verbose, threads):

    train = sp.load_npz(os.path.join(input_file, 'train.npz'))
    test = sp.load_npz(os.path.join(input_file, 'test.npz'))
    items = sp.load_npz(os.path.join(input_file, 'items.npz'))
    segments = input_file.split('/')
    conditions = segments[-1].split('.')

    logger.info('Dataset, site: {}, months: {}, min_uv: {}, max_uv: {}, min_vv: {}, train: {}, test: {}'.format(
        segments[-3].replace('_', '.'), segments[-2].split('.'), conditions[0], conditions[1], conditions[2], train.shape, test.shape))

    logger.info(
        'model: {}, loss: {}, item_alpha(L2): {} no_components: {}, learning_rate: {}'.format(model, loss, 1e-6, 30, 0.05))

    model = LightFM(loss='warp', learning_rate=0.05, item_alpha=1e-6, no_components=30)
    model = model.fit(train, epochs=10, verbose=True)
    logger.info('A pure collaborative filtering model')
    evaluate_model(model, train, test)


    model = LightFM(loss='warp', learning_rate=0.05, item_alpha=1e-6, no_components=30)
    model = model.fit(train, item_features=items, epochs=10, verbose=True)
    logger.info('A hybrid model')
    evaluate_model(model, train, test,item_fetures=items)



@cli.command(help='testing recommender systems')
@click.pass_context
@click.argument('input_filepath', type=click.Path(exists=True), default='../../data/raw/latest.train.tbl')
@click.argument('db_filepath', type=click.Path(exists=True), default='../../data/db/latest.context.tbl')
@click.argument('output_directory', type=click.Path(), default='../../data/demo/')
def test(ctx, input_filepath, db_filepath, output_directory):
    pass


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())
    cli()
