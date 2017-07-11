import timeit
import click
import logging

import pandas as pd
import yaml
from dotenv import find_dotenv, load_dotenv

logger = logging.getLogger(__name__)


# @click.group()
# @click.option('-cf', '--config', type=click.File('r'), default='../../config.yaml', help='config path')
@click.pass_context
def cli(ctx, config):
    pass


# @cli.command(
#     help='Runs data processing scripts to turn raw data from input_file into cleaned data ready to be analyzed (saved in output_file)')
# @click.pass_context
# @click.argument('input_file', type=click.Path(exists=True), default='../../data/raw/latest.train.tbl')
# @click.argument('execute_output_file', type=click.File('a'), default='../../data/processed/latest_exec.train.dat')
# @click.argument('load_output_file', type=click.File('a'), default='../../data/processed/latest_load.train.dat')
# def tune(ctx, input_file, execute_output_file, load_output_file):





@cli.command(help='Runs hyper-parameter tuning on a sklearn model')
@click.pass_context
@click.argument('input_file', type=click.Path(exists=True), default='../../data/processed/train.csv')
def tune(ctx, input_file, execute_output_file, load_output_file):
    client = SqlClient(os.environ.get('CONNECTION_STRING'))
    query = ctx.obj['context_query']

    augmenter = feature_augmenter.FeaturesAugmentor()

    for idx, df in enumerate(load_raw_features_file(input_file, ctx['names'])):
        context = client.select(query, df)

        df_with_features = augmenter.transform(df, context)

        # embedding.eval(df_with_features)

        # append
        dump_vw_cb_adf_file(df_with_features, execute_output_file, load_output_file)


@cli.command(help='update database with input_file')
@click.pass_context
@click.argument('input_file', type=click.Path(exists=True), default='../../data/db/latest.context.tbl')
def insert(ctx, input_file):
    t = timeit.Timer(setup="client = SqlClient(os.environ.get('CONNECTION_STRING'))")

    for idx, df in enumerate(load_context_file(input_file, ctx['context_names'])):
        logger.info('inserting %d rows from %s, passed line %d, operation taken %.3f' % (
            df.shape[0], input_file, idx * df.shape[0], t.timeit('client.insert(df)')))


@cli.command(help='creates sample raw files')
@click.pass_context
@click.argument('input_filepath', type=click.Path(exists=True), default='../../data/raw/latest.train.tbl')
@click.argument('db_filepath', type=click.Path(exists=True), default='../../data/db/latest.context.tbl')
@click.argument('output_directory', type=click.Path(), default='../../data/demo/')
def demo(ctx, input_filepath, db_filepath, output_directory):
    logger.info('making sample raw files')

    df = iter(load_raw_features_file(input_filepath, ctx.obj['names'], 4)).next()
    keys = df.reset_index()[['ad_campaign_id', 'country_code', 'domain']].drop_duplicates()
    logger.info('loaded first chunk from %s, df=%s' % (df.shape, input_filepath))

    logger.info('filtering %s, this can take time' % db_filepath)
    context = pd.concat(
        [frame.reset_index().merge(keys) for frame in load_context_file(db_filepath, ctx['short_names'])])

    logger.info('saving to %s' % output_directory)
    if not os.path.exists(output_directory): os.mkdir(output_directory)
    df.to_csv(os.path.join(output_directory, os.path.basename(input_filepath)), index=False, header=None, sep='\x01')
    context.to_csv(os.path.join(db_filepath, os.path.basename(db_filepath)), index=False, header=None, sep='\x01')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())
    cli()
