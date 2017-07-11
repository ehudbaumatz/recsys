import click
import logging

import pandas as pd
import yaml
from dotenv import find_dotenv, load_dotenv
from lightfm_model import LightWrapper
from tune_model import random_search

logger = logging.getLogger(__name__)


@click.group()
@click.option('-cf', '--config', type=click.File('r'), default='../../docs/config.yaml', help='config path')
@click.pass_context
def cli(ctx, config):
    ctx.obj = yaml.load(config)

@cli.command(help='hyper-parameter tuning on a sklearn model')
@click.pass_context
@click.argument('input_file', type=click.Path(exists=True), default='../../data/processed/train.csv')
@click.option('-m', '--model', default='lightfm', help='model to tune')
@click.option('-l', '--loss', default='warp', help='loss to optimize')
@click.option('-j', '--jobs', default=-1, help='number of threads')
@click.option('-i', '--iter', default=10, help='number of hyper parameter search iterations')
@click.option('-v', '--verbose', default=10, help='level of verbosity')
def tune(ctx, input_file, model, loss, jobs, iter, verbose):

    df = pd.read_csv(input_file)
    if model == 'lightfm':

        # users and items ids
        users = list(sorted(df.ip.unique()))
        items = list(sorted(df.vid.unique()))

        # for tuning we utilize sklearn parallelism, so using default one thread
        clf = LightWrapper(loss=loss, users=users, items=items)

        # take first N items (shuffled by sort) for test and rest for training TODO - this is really bad ... ned CV
        test = df.groupby('ip', sort=True, as_index=False).head(5)
        train = df[~df.isin(test)].dropna()
        cfg = ctx.obj.get('tune_group')[model]
        cfg['n_iter_search'] = iter; cfg['n_jobs'] = jobs; cfg['verbose'] = verbose

        random_search(clf, df, [[train.index, test.index]], **cfg)
    else: raise Exception('only lightfm supported currently')

@cli.command(help='training recommender systems')
@click.pass_context
@click.argument('input_file', type=click.Path(exists=True), default='../../data/db/latest.context.tbl')
def train(ctx, input_file):
    pass


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
    # tune()

