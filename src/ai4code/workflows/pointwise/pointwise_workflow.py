import logging
import os
import random
import sys
from bisect import bisect
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import wandb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import DataCollatorWithPadding

from ai4code.text_cleaning_utils import preprocess_text
from ai4code.workflows.pointwise.dataset import MarkdownDataset
from ai4code.workflows.pointwise.train import train, validate
from ai4code.workflows.workflow import Workflow, CVWorkflow

logger = logging.getLogger(__name__)


#pd.options.display.width = 180
#pd.options.display.max_colwidth = 120

#BERT_PATH = "distilbert-base-uncased"

#data_dir = Path('../input/AI4Code')


NVALID = 0.1  # size of validation set





def read_notebook(path):
    return (
        pd.read_json(
            path,
            dtype={'cell_type': 'category', 'source': 'str'})
        .assign(id=path.stem)
        .rename_axis('cell_id')
    )


def get_ranks(base, derived):
    return [base.index(d) for d in derived]


def remove_imports(row):
    lines = row.split('\n')
    lines = [x for x in lines if not x.startswith('import') and not x.startswith('from ')]
    return '\n'.join(lines)


class OrigPointwiseWorkflow(Workflow):
    def __init__(self, cfg, env):
        self.env = env
        self.cfg = cfg.args


    def train(self, input_data_path, artifacts_path):
        paths_train = list((Path(input_data_path) / 'train').glob('*.json'))[:self.cfg.data.max_data_size]
        notebooks_train = [
            read_notebook(path) for path in tqdm(paths_train, desc='Train NBs')
        ]

        df = (
            pd.concat(notebooks_train)
                .set_index('id', append=True)
                .swaplevel()
                .sort_index(level='id', sort_remaining=False)
        )

        if self.cfg.data.remove_imports:
            df.loc[df.cell_type == 'code', 'source'] = df.loc[df.cell_type == 'code', 'source'].apply(remove_imports)

        df_orders = pd.read_csv(
            Path(input_data_path) / 'train_orders.csv',
            index_col='id',
            squeeze=True,
        ).str.split()  # Split the string representation of cell_ids into a list

        df_orders_ = df_orders.to_frame().join(
            df.reset_index('cell_id').groupby('id')['cell_id'].apply(list),
            how='right',
        )

        ranks = {}
        for id_, cell_order, cell_id in df_orders_.itertuples():
            ranks[id_] = {'cell_id': cell_id, 'rank': get_ranks(cell_order, cell_id)}  # not same

        df_ranks = (
            pd.DataFrame
                .from_dict(ranks, orient='index')
                .rename_axis('id')
                .apply(pd.Series.explode)
                .set_index('cell_id', append=True)
        )

        tokenizer = AutoTokenizer.from_pretrained(self.cfg.model_name, do_lower_case=True)

        df_ancestors = pd.read_csv(Path(input_data_path) / 'train_ancestors.csv', index_col='id')
        df = df.reset_index().merge(df_ranks, on=["id", "cell_id"]).merge(df_ancestors, on=[
            "id"])  # not same but don't think it matters
        df["pct_rank"] = df["rank"] / df.groupby("id")["cell_id"].transform("count")

        if self.cfg.data.clean_text:
            df.source = df.source.apply(preprocess_text)

        splitter = GroupShuffleSplit(n_splits=1, test_size=NVALID, random_state=0)

        train_ind, val_ind = next(splitter.split(df, groups=df["ancestor_id"]))

        train_df = df.loc[train_ind].reset_index(drop=True)
        val_df = df.loc[val_ind].reset_index(drop=True)

        train_df_mark = train_df[train_df["cell_type"] == "markdown"].reset_index(drop=True)
        val_df_mark = val_df[val_df["cell_type"] == "markdown"].reset_index(drop=True)

        if self.cfg.data.add_markdown_context:
            train_df_mark = self.add_context(train_df_mark, train_df[train_df["cell_type"] == "code"], tokenizer)
            val_df_mark = self.add_context(val_df_mark, val_df[val_df["cell_type"] == "code"], tokenizer)

        train_ds = MarkdownDataset(train_df_mark, max_len=self.cfg.tokenizer.max_len, tokenizer=tokenizer, add_markdown_context=self.cfg.data.add_markdown_context)
        val_ds = MarkdownDataset(val_df_mark, max_len=self.cfg.tokenizer.max_len, tokenizer=tokenizer, add_markdown_context=self.cfg.data.add_markdown_context)

        train_loader = DataLoader(train_ds, batch_size=self.cfg.training.per_device_train_batch_size, shuffle=True, num_workers=self.cfg.training.num_workers,
                                  pin_memory=False, drop_last=True, collate_fn=DataCollatorWithPadding(tokenizer))
        val_loader = DataLoader(val_ds, batch_size=self.cfg.training.per_device_eval_batch_size, shuffle=False, num_workers=self.cfg.training.num_workers,
                                pin_memory=False, drop_last=False, collate_fn=DataCollatorWithPadding(tokenizer))

        model = AutoModelForSequenceClassification.from_pretrained(
           self.cfg.model_name, num_labels=1)
        
        model = model.to(self.env.device)
        model, y_pred, val_kd = train(model, train_loader, val_loader, epochs=self.cfg.training.num_train_epochs,
                              device=self.env.device, df_orders=df_orders, val_df=val_df,
                              lr=self.cfg.training.learning_rate,
                              lr_decay_rate=self.cfg.training.lr_decay_rate)

        wandb.log({'kendall_tau': val_kd})

        logger.info(f'validation kendall tau={val_kd}')
        model.save_pretrained(os.path.join(artifacts_path, 'model'))
        tokenizer.save_pretrained(os.path.join(artifacts_path, 'model'))

    def subsample_context(self, row, num_to_sample):
        row_len = len(row)
        row_sample_idx = random.sample(range(row_len), min(row_len, num_to_sample))
        row_sample_idx = sorted(row_sample_idx)
        return [row[idx] for idx in row_sample_idx]

    def add_context(self, df, df_code, tokenizer):
        num_to_sample = self.cfg.data.markdown_context.num_to_sample
        sample_max_len = self.cfg.data.markdown_context.max_len
        sep = tokenizer.sep_token

        df_code = df_code.copy()
        df_code.source = df_code.source.apply(lambda x: x[:min(len(x), 500)])
        train_context_markdown = df_code.groupby('id').source.apply(
            lambda x: list(x)).to_frame()
        df = pd.merge(df, train_context_markdown, on='id', how='left', suffixes=('', '_all'))
        #df = df.apply(filter_markdown_context, axis=1)
        df.source_all = df.source_all.apply(lambda x: self.subsample_context(x, num_to_sample))
        df.source_all = df.source_all.apply(
            lambda x: sep.join(['\n'.join(z.split('\n')[:sample_max_len]) for z in x]))

        if self.cfg.data.markdown_context.clean:
            df.source_all = df.source_all.apply(preprocess_text)

        return df

    def predict(self, input_data_path, artifacts_path):
        checkpoint_dir = os.path.join(artifacts_path, 'model')
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, do_lower_case=True) # TODO: replace with autotokenizer
        
        paths_test = list((Path(input_data_path) / 'test').glob('*.json'))
        notebooks_test = [
            read_notebook(path) for path in tqdm(paths_test, desc='Test NBs')
        ]
        test_df = (
            pd.concat(notebooks_test)
                .set_index('id', append=True)
                .swaplevel()
                .sort_index(level='id', sort_remaining=False)
        ).reset_index()

        if self.cfg.data.remove_imports:
            test_df.loc[test_df.cell_type == 'code', 'source'] = test_df.loc[test_df.cell_type == 'code', 'source'].apply(remove_imports)


        test_df["rank"] = test_df.groupby(["id", "cell_type"]).cumcount()
        test_df["pred"] = test_df.groupby(["id", "cell_type"])["rank"].rank(pct=True)

        if self.cfg.data.clean_text:
            test_df.source = test_df.source.apply(preprocess_text)

        test_df["pct_rank"] = 0

        test_df_markdown = test_df[test_df["cell_type"] == "markdown"].reset_index(drop=True)

        if self.cfg.data.add_markdown_context:
            test_df_markdown = self.add_context(test_df_markdown, test_df[test_df["cell_type"] == "code"], tokenizer)

        test_ds = MarkdownDataset(test_df_markdown, max_len=self.cfg.tokenizer.max_len,
                                  tokenizer=tokenizer, add_markdown_context=self.cfg.data.add_markdown_context)
        test_loader = DataLoader(test_ds, batch_size=self.cfg.training.per_device_eval_batch_size, shuffle=False, num_workers=self.cfg.training.num_workers,
                                 pin_memory=False, drop_last=False, collate_fn=DataCollatorWithPadding(tokenizer))

        model = AutoModelForSequenceClassification.from_pretrained(os.path.join(artifacts_path, 'model'))

        _, y_test = validate(model, test_loader, device=self.env.device)
        test_df.loc[test_df["cell_type"] == "markdown", "pred"] = y_test
        return test_df

    def create_submission(self, test_preds, output_dir):
        sub_df = test_preds.sort_values("pred").groupby("id")["cell_id"].apply(lambda x: " ".join(x)).reset_index()
        sub_df.rename(columns={"cell_id": "cell_order"}, inplace=True)
        sub_df.to_csv(os.path.join(output_dir, "submission.csv"), index=False)


def filter_markdown_context(row):
    row['source_all'] = [x for x in row['source_all'] if x != row['source']]
    return row