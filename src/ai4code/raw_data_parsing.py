import logging
import os
from pathlib import Path

import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)

def get_ranks(base, derived):
    return [base.index(d) for d in derived]


#class RawDataParser:
#    def __init__(self, data_dir, data_size=100):
#        self.df, self.df_ranks = self._get_raw_data(data_dir, data_size)

def get_raw_data(data_dir, is_train, data_size):
    if not os.path.exists(data_dir):
        raise Exception(f'Dir {data_dir} does not exist')

    if is_train:
        dataset_str = 'train'
    else:
        dataset_str = 'test'
    data_dir = Path(data_dir)
    paths_train = list((data_dir / dataset_str).glob('*.json'))
    if dataset_str == 'train':
        paths_train = paths_train[:data_size]
    notebooks_train = [
        _read_notebook(path) for path in tqdm(paths_train, desc=f'{dataset_str} NBs')
    ]

    df = (
        pd.concat(notebooks_train)
            .set_index('id', append=True)
            .swaplevel()
            .sort_index(level='id', sort_remaining=False)
    )

    if is_train:
        df_orders = pd.read_csv(
            data_dir / 'train_orders.csv',
            index_col='id',
            squeeze=True,
        ).str.split()  # Split the string representation of cell_ids into a list

        df_orders_ = df_orders.to_frame().join(
            df.reset_index('cell_id').groupby('id')['cell_id'].apply(list),
            how='right',
        )

        ranks = {}
        for id_, cell_order, cell_id in df_orders_.itertuples():
            ranks[id_] = {'cell_id': cell_id, 'rank': get_ranks(cell_order, cell_id)}

        df_ranks = (
            pd.DataFrame
                .from_dict(ranks, orient='index')
                .rename_axis('id')
                .apply(pd.Series.explode)
                .set_index('cell_id', append=True)
        )

        return df, df_ranks
    else:
        return df

def _read_notebook(path):
    return (
        pd.read_json(
            path,
            dtype={'cell_type': 'category', 'source': 'str'})
            .assign(id=path.stem)
            .rename_axis('cell_id')
    )


