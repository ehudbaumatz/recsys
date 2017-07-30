import glob
import logging
import tarfile

import click
import gzip
import numpy as np
import pandas as pd
import yaml
from dotenv import find_dotenv, load_dotenv
import scipy.sparse as sp
from datasets import load_interactions_dataset, load_items_features

# from lightfm import LightFM
# from lightfm.evaluation import precision_at_k, auc_score
#
# from lightfm_model import LightWrapper
# from tune_model import random_search
#
# from lightfm.datasets import fetch_movielens, fetch_stackexchange
import sys, os
sys.path.insert(0, os.path.abspath('..'))


logger = logging.getLogger(__name__)

def split_path(path):

    name = os.path.basename(path)
    condition = name[name.find('.')+1:name.rfind('.')]
    name = name[:name.find('.')]
    return name, condition


##### CLI ######
@click.group()
@click.option('-cf', '--config', type=click.File('r'), default='../../docs/config.yaml', help='config path')
@click.pass_context
def cli(ctx, config):
    ctx.obj = yaml.load(config)


@cli.command(help='generate dataset')
@click.argument('input_file', type=click.Path(exists=True), default='../../data/intermediate/aol_com/06/')
@click.argument('output_file', default='../../data/processed/aol_com/06/')
@click.option('-f', '--format', default='coo', help='format to save data')
@click.option('-t', '--text', default='glove', help='text vectorization')
def generate(input_file, output_file, format, text):

    interactions = glob.glob(input_file + '/interactions*')
    items = glob.glob(input_file + '/items_data*')
    users = glob.glob(input_file + '/users_data*')
    mapper = {}

    for interaction in interactions:

        name, condition = split_path(interaction)
        logger.info('loading interactions %s dataset, transforming to: %s format, condition: %s' % (name, format, condition))
        train, test = load_interactions_dataset(interaction, format)
        mapper[condition] = {'train':train, 'test':test}

    for item_path in items:

        name, condition = split_path(item_path)
        assert (condition in mapper)
        logger.info( 'loading items %s dataset, vectorization method: %s, condition: %s' % (name, format, output_file))

        items_features = load_items_features(item_path, vectorization=text)
        mapper[condition]['items'] = items_features

    for key, dic in mapper.items():

        path = os.path.join(output_file, key)
        os.makedirs(path)
        sp.save_npz(os.path.join(path, 'train.npz') , dic['train'])
        sp.save_npz(os.path.join(path, 'test.npz'), dic['test'])
        sp.save_npz(os.path.join(path, 'items.npz'), dic['items'])


if __name__ == '__main__':




    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())
    cli()
