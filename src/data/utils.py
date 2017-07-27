import json
import csv

import logging
import requests
import numpy as np

logger = logging.getLogger(__name__)
log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_fmt)


def get_data_from_elasticsearch(ids, fields='category,lifecycle,localized', path='../../data/raw/additional_video_data1.csv'):

    grace = lambda dic, col: dic[col] if dic.has_key(col) else 0

    with open(path, 'w') as f:

        writer = csv.DictWriter(f, fieldnames=['id', 'category', 'lifecycle', 'localized'])
        writer.writeheader()

        url = 'http://elasticsearch-master.vidible.aolcloud.net:9200/video/video/_search?q=(%s)&fields=' + fields
        step = 20
        chunks = [ids[i:i+step] for i in range(0, len(ids), step)]
        for idx, chunk in enumerate(chunks):
            if idx % 10 == 0: logger.info('passed %d out of %d' % (idx, len(chunks)))
            try:
                response = requests.get(url=url % ' OR '.join(chunk))
                res = json.loads(response.content)
                for hit in res[u'hits'][u'hits']:

                    fields = {'id': hit[u'_id']}
                    fields.update(hit[u'fields'])

                    writer.writerow(fields)

            except Exception as ex:
                logger.error(ex)


def get_indices_indptr(df, embedding_size):

    n_features = df.shape[1] * embedding_size
    s = df.isnull().sum(axis=1)
    s = n_features - s * embedding_size
    s = s.cumsum()
    indices = np.arange(s.values[-1])
    indptrs = np.insert(s.values, 0, 0)
    return indices, indptrs








