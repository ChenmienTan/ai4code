import logging
import os
import pickle
import random
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import wandb
from scipy.special import softmax
from sentence_transformers import SentenceTransformer, datasets, losses, InputExample, models
from sentence_transformers.evaluation import BinaryClassificationEvaluator, SentenceEvaluator
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader
from tqdm import tqdm

from ai4code.text_cleaning_utils import preprocess_text
from ai4code.workflows.pointwise.evaluation import kendall_tau
from ai4code.workflows.workflow import Workflow

logger = logging.getLogger(__name__)


NVALID = 0.1  # size of validation set


class KTEvaluator(SentenceEvaluator):
    def __init__(self, val_df, order_df, mode, softmax_temp, add_boundary_code_cells, asym_pool, name: str = '', batch_size: int = 32, show_progress_bar: bool = False):
        self.order_df = order_df
        self.val_df = val_df
        self.name = name
        self.batch_size = batch_size
        self.mode = mode
        self.softmax_temp = softmax_temp
        self.add_boundary_code_cells = add_boundary_code_cells
        self.asym_pool = asym_pool
        if show_progress_bar is None:
            show_progress_bar = (logger.getEffectiveLevel() == logging.INFO or logger.getEffectiveLevel() == logging.DEBUG)
        self.show_progress_bar = show_progress_bar

    @classmethod
    def from_input_examples(cls, val_df, order_df, **kwargs):
        return cls(val_df, order_df, **kwargs)

    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:

        if epoch != -1:
            if steps == -1:
                out_txt = f" after epoch {epoch}:"
            else:
                out_txt = f" in epoch {epoch} after {steps} steps:"
        else:
            out_txt = ":"

        logger.info("K-T Accuracy Evaluation of the model on " + self.name + " dataset" + out_txt)

        main_score = self.compute_metrices(model)

        return main_score


    def compute_metrices(self, model):
        preds = get_preds_from_m2c_emb(self.val_df, model, self.mode, self.softmax_temp, self.asym_pool)
        val_df2 = self.val_df.copy()
        val_df2['pred'] = preds
        if self.add_boundary_code_cells:
            val_df2 = val_df2[val_df2.cell_id != "the_end"]  # TODO: handle in inference too
            val_df2 = val_df2[val_df2.cell_id != "the_begin"]

        y_dummy = val_df2.sort_values("pred").groupby('id')['cell_id'].apply(list)
        val_kd = kendall_tau(self.order_df.loc[y_dummy.index], y_dummy)
        return val_kd




def logging_callback(score, epoch, steps):
    logger.info(f'score={score} epoch={epoch} steps={steps}')
    wandb.log({'validation_kd': score})

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

def add_boundary_code_cells_func(group, boundary_start_text, boundary_end_text, is_inference=False):
    output = group
    max_rank_idx = group['rank'].astype('int').idxmax()
    if group.loc[max_rank_idx, :].cell_type == 'markdown' or is_inference:
        group_rank = group.loc[group.cell_type == 'code', 'rank'].max() + 1 if is_inference else group['rank'].max() + 1
        last_row_dict = {'id': [group['id'].values[0]], 'cell_id': ["the_end"], 'cell_type': ['code'],
                   'source': [boundary_end_text], 'rank': [group_rank]}

        if 'ancestor_id' in group.columns:
            last_row_dict['ancestor_id'] = [group['ancestor_id'].values[0]]
        if 'parent_id' in group.columns:
            last_row_dict['parent_id'] = [group['parent_id'].values[0]]
        if 'parent_id' in group.columns:
            last_row_dict['pct_rank'] = 1.0
        last_row_df = pd.DataFrame.from_dict(last_row_dict)
        output = pd.concat([output, last_row_df], axis=0)
    min_rank_idx = group['rank'].astype('int').idxmin()
    if group.loc[min_rank_idx, :].cell_type == 'markdown' or is_inference:
        group_rank = group.loc[group.cell_type == 'code', 'rank'].min() -1 if is_inference else group['rank'].min() + 1

        first_row_dict = {'id': [group['id'].values[0]], 'cell_id': ["the_begin"], 'cell_type': ['code'],
                 'source': [boundary_start_text], 'rank': [group_rank]
                 }
        if 'ancestor_id' in group.columns:
            first_row_dict['ancestor_id'] = [group['ancestor_id'].values[0]]
        if 'parent_id' in group.columns:
            first_row_dict['parent_id'] = [group['parent_id'].values[0]]
        if 'parent_id' in group.columns:
            first_row_dict['pct_rank'] = 0.0
        first_row_df = pd.DataFrame.from_dict(
            first_row_dict)
        output = pd.concat([first_row_df, output], axis=0)
    return output


def get_input_examples(df):
    df = df.copy()
    df_mark = df[df["cell_type"] == "markdown"].reset_index(drop=True)
    df_code = df[df["cell_type"] == "code"].reset_index(drop=True)
    df_code['rank'] = df_code['rank'] - 1
    df_cmb = pd.merge(df_mark, df_code[['id', 'source', 'rank', 'cell_id']], on=['id', 'rank'], suffixes=('', '_code'))

    if 'orig_rank' in df_cmb.columns:
        filter_mask = (df_cmb['rank'] - df_cmb['orig_rank']).abs() <= 1
        df_cmb.loc[filter_mask, 'source_code'] = 'DELETE_ME'

    return df_cmb.apply(lambda x: InputExample(texts=[x['source'], x['source_code']]), axis=1).values.tolist()


def get_input_examples_hard(df):
    samples = get_input_examples(df)

    df_shuffled = df.copy().groupby(['id']).apply(shuffle_group).reset_index()

    begin_text = df_shuffled[df_shuffled.cell_id == 'the_begin'].source.values[0]
    end_text = df_shuffled[df_shuffled.cell_id == 'the_end'].source.values[0]

    neg_samples = get_input_examples(df_shuffled)

    for p,n in zip(samples, neg_samples):
        if p.texts[1] == begin_text:
            n.texts[1] = end_text
        elif p.texts[1] == end_text:
            n.texts[1] = begin_text

    # hacky way to deal with bad hard negatives: otherwise we run into collation issues
    return [InputExample(texts=x.texts + [y.texts[1]]) if y.texts[1] != 'DELETE_ME'
            else InputExample(texts=x.texts + [random.choice(neg_samples).texts[1]])
            for x, y in zip(samples, neg_samples)]


def get_pred_group_func(group, model, mode, temp, asym_pool):
    group["rank4pred"] = group.groupby(["id", "cell_type"])["rank"].rank(pct=False)

    df_mark = group[group.cell_type == 'markdown']
    df_code = group[group.cell_type == 'code']

    if asym_pool:
        embs_mark = model.encode([{'markdown': x} for x in df_mark.source.values], batch_size=128, show_progress_bar=False,
                                 normalize_embeddings=True)
        embs_code = model.encode([{'code': x} for x in df_code.source.values], batch_size=128, show_progress_bar=False,
                                 normalize_embeddings=True)
    else:
        all_embs = model.encode(df_mark.source.values.tolist() + df_code.source.values.tolist(), batch_size=128, show_progress_bar=False,
                                 normalize_embeddings=True)
        embs_mark = all_embs[:len(df_mark)]
        embs_code = all_embs[len(df_mark):]

    mark_code_sim = np.dot(embs_mark, np.transpose(embs_code))

    if mode == 'argmax':
        markdown_argmax = np.argmax(mark_code_sim, axis=1)
        argmax_code_ranks = df_code.iloc[markdown_argmax]['rank4pred']
        group['pred'] = group['rank4pred']
        group.loc[group["cell_type"] == "markdown", 'pred'] = argmax_code_ranks.values - 0.5
    elif mode == 'softmax':
        code_ranks = df_code['rank4pred'].values.tolist()
        markdown_softmax = softmax(temp * mark_code_sim, axis=1)
        rep_code_ranks = np.transpose(np.repeat(np.expand_dims(code_ranks, -1), markdown_softmax.shape[0], axis=1))
        group['pred'] = group['rank4pred']
        preds = np.average(rep_code_ranks, weights=markdown_softmax, axis=1)
        group.loc[group["cell_type"] == "markdown", 'pred'] = preds - 0.5
    elif mode == 'hungary':
        try:
            row_ind, col_ind = scipy.optimize.linear_sum_assignment(mark_code_sim, maximize=True)
            argmax_code_ranks = df_code.iloc[col_ind]['rank4pred']
            group['pred'] = group['rank4pred']
            group.loc[group["cell_type"] == "markdown", 'pred'] = argmax_code_ranks.values - 0.5
        except:
            logger.warning('Failed to hungary')
            markdown_argmax = np.argmax(mark_code_sim, axis=1)
            argmax_code_ranks = df_code.iloc[markdown_argmax]['rank4pred']
            group['pred'] = group['rank4pred']
            group.loc[group["cell_type"] == "markdown", 'pred'] = argmax_code_ranks.values - 0.5
    else:
        raise NotImplementedError

    return group

def get_preds_from_m2c_emb(df_, model, mode, temp, asym_pool):
    df = df_.groupby('id').apply(lambda x: get_pred_group_func(x, model, mode, temp, asym_pool)).reset_index()
    #df = get_pred_group_func(df_, model, mode)
    return df['pred']




def shuffle_group(group):
    group['orig_rank'] = group['rank']
    group['rank'] = np.random.permutation(group['rank'].values)
    return group

def get_val_examples(df):
    samples = get_input_examples(df)
    for sample in samples:
        sample.label = 1

    df_shuffled = df.copy().groupby(['id']).apply(shuffle_group).reset_index()
    neg_samples = get_input_examples(df_shuffled)
    for sample in neg_samples:
        sample.label = 0

    return samples + neg_samples
    #code_texts = shuffle([x.texts[1] for x in samples])
    #neg_samples = [InputExample(texts=[x.texts[0], y], label=0) for x,y in zip(samples, code_texts)]



class M2CWorkflow(Workflow):
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

        if self.cfg.model_head == 'mean_pooling':
            bert = models.Transformer(self.cfg.model_name)
            pooler = models.Pooling(
                bert.get_word_embedding_dimension(),
                pooling_mode_mean_tokens=True
            )
            model = SentenceTransformer(modules=[bert, pooler])
        elif self.cfg.model_head == 'default':
            model = SentenceTransformer(self.cfg.model_name)
        elif self.cfg.model_head == 'asym_pool':
            bert = models.Transformer(self.cfg.model_name)
            pooler = models.Pooling(
                bert.get_word_embedding_dimension(),
                pooling_mode_mean_tokens=True
            )

            emb_dim = self.cfg.asym_emb_size
            logger.info(f'asym emb_dim={emb_dim}')

            asym_model = models.Asym({'markdown': [models.Dense(bert.get_word_embedding_dimension(), emb_dim)],
                                      'code': [ models.Dense(bert.get_word_embedding_dimension(), emb_dim)]},
                                     allow_empty_key=False)
            model = SentenceTransformer(modules=[bert, pooler, asym_model])
        else:
            raise NotImplementedError

        model.max_seq_length = self.cfg.tokenizer.max_len

        if self.cfg.data.add_boundary_code_cells:
            boundary_func = lambda x: add_boundary_code_cells_func(x, self.cfg.data.boundary_start_text,
                                                                   self.cfg.data.boundary_end_text)
            train_df = train_df.groupby('id').apply(boundary_func).reset_index(drop=True)
            val_df = val_df.groupby('id').apply(boundary_func).reset_index(drop=True)

        logger.info('Preparing training samples')
        if not self.cfg.data.hard_negatives:
            train_samples = get_input_examples(train_df)
        else:
            train_samples = get_input_examples_hard(train_df)
            num_hard = len([x for x in train_samples if len(x.texts) == 3])
            num_easy = len([x for x in train_samples if len(x.texts) == 2])
            logger.info('{} samples with hard negs, {} without'.format(num_hard, num_easy))
        logger.info('Preparing val samples')
        val_samples = get_val_examples(val_df)

        if self.cfg.model_head == 'asym_pool':
            train_samples = [InputExample(texts=[{'markdown': x.texts[0]}] + [{'code': y} for y in x.texts[1:]]) for x in train_samples]
            val_samples = [InputExample(texts=[{'markdown': x.texts[0]}] + [{'code': y} for y in x.texts[1:]]) for x
                             in val_samples]
            train_loader = DataLoader(
                train_samples, batch_size=self.cfg.training.per_device_train_batch_size)
        else:
            train_loader = datasets.NoDuplicatesDataLoader(
                train_samples, batch_size=self.cfg.training.per_device_train_batch_size)
        loss = losses.MultipleNegativesRankingLoss(model)

        checkpoint_dir = os.path.join(artifacts_path, 'checkpoints')
        fit_output_path = os.path.join(artifacts_path, 'fit_output')

        warmup_steps = int(len(train_loader) * self.cfg.training.num_train_epochs * 0.1)

        if self.cfg.training.evaluator == 'none':
            evaluator = None
        elif self.cfg.training.evaluator == 'binary':
            evaluator = BinaryClassificationEvaluator.from_input_examples(
                val_samples,
                batch_size=self.cfg.training.per_device_eval_batch_size)
        elif self.cfg.training.evaluator == 'kendalltau':
            evaluator = KTEvaluator.from_input_examples(val_df, df_orders,
                                                        mode=self.cfg.inference.mode,
                                                        softmax_temp=self.cfg.inference.softmax_temp,
                                                        add_boundary_code_cells=self.cfg.data.add_boundary_code_cells,
                                                        batch_size=self.cfg.training.per_device_eval_batch_size,
                                                        asym_pool=self.cfg.model_head == 'asym_pool')
        else:
            raise NotImplementedError

        if True:
            model.fit(
                train_objectives=[(train_loader, loss)],
                epochs=self.cfg.training.num_train_epochs,
                evaluator=evaluator,
                warmup_steps=warmup_steps,
                checkpoint_path=checkpoint_dir,
                show_progress_bar=self.cfg.training.show_progress_bar,
                use_amp=self.env.device != 'cpu',
                checkpoint_save_total_limit=1,
                output_path=fit_output_path,
                evaluation_steps=self.cfg.training.eval_steps,
                callback=logging_callback
            )

        model = SentenceTransformer(fit_output_path)

        val_df['pred'] = get_preds_from_m2c_emb(val_df, model, self.cfg.inference.mode, self.cfg.inference.softmax_temp, self.cfg.model_head == 'asym_pool')

        if self.cfg.data.add_boundary_code_cells:
            val_df = val_df[val_df.cell_id != "the_end"] # TODO: handle in inference too
            val_df = val_df[val_df.cell_id != "the_begin"]

        y_dummy = val_df.sort_values("pred").groupby('id')['cell_id'].apply(list)
        val_kd = kendall_tau(df_orders.loc[y_dummy.index], y_dummy)

        wandb.log({'kendall_tau': val_kd})

        logger.info(f'validation kendall tau={val_kd}')
        model.save(os.path.join(artifacts_path, 'model'))

        val_df.to_pickle(os.path.join(artifacts_path, 'oof.pkl'))


        #model.save_pretrained(os.path.join(artifacts_path, 'model'))


    def predict(self, input_data_path, artifacts_path):
        checkpoint_dir = os.path.join(artifacts_path, 'model')
        model = SentenceTransformer(checkpoint_dir)

        paths_test = list((Path(input_data_path) / 'test').glob('*.json'))
        notebooks_test = [
            read_notebook(path) for path in tqdm(paths_test, desc='Test NBs')
        ]

        logger.info('num notebooks {}'.format(len(notebooks_test)))

        test_df = (
            pd.concat(notebooks_test)
                .set_index('id', append=True)
                .swaplevel()
                .sort_index(level='id', sort_remaining=False)
        ).reset_index()

        logger.info('df size {}'.format(len(test_df)))

        if self.cfg.data.remove_imports:
            test_df.loc[test_df.cell_type == 'code', 'source'] = test_df.loc[test_df.cell_type == 'code', 'source'].apply(remove_imports)

        test_df["rank"] = test_df.groupby(["id", "cell_type"]).cumcount()

        if self.cfg.data.add_boundary_code_cells:
            boundary_func = lambda x: add_boundary_code_cells_func(x, self.cfg.data.boundary_start_text, self.cfg.data.boundary_end_text, True)
            test_df = test_df.groupby('id').apply(boundary_func).reset_index(drop=True)

        if self.cfg.data.clean_text:
            test_df.source = test_df.source.apply(preprocess_text)

        test_df['pred'] = get_preds_from_m2c_emb(test_df, model, self.cfg.inference.mode, self.cfg.inference.softmax_temp, self.cfg.model_head == 'asym_pool')

        if self.cfg.inference.produce_embs:
            logger.info('Creating output embs')
            logger.info('df size at emb produce {}'.format(len(test_df)))
            embs = model.encode(test_df.source.values, batch_size=self.cfg.training.per_device_eval_batch_size, show_progress_bar=False,
                                 normalize_embeddings=True)
            logger.info('embs size {}'.format(embs.shape))
            with open(os.path.join(self.env.submission_path, 'embs.pkl'), 'wb') as fout:
                pickle.dump(embs, fout)

        if self.cfg.data.add_boundary_code_cells:
            test_df = test_df[test_df.cell_id != "the_end"] #
            test_df = test_df[test_df.cell_id != "the_begin"]

        return test_df

    def create_submission(self, test_preds, output_dir):
        sub_df = test_preds.sort_values("pred").groupby("id")["cell_id"].apply(lambda x: " ".join(x)).reset_index()
        sub_df.rename(columns={"cell_id": "cell_order"}, inplace=True)
        sub_df.to_csv(os.path.join(output_dir, "submission.csv"), index=False)


def filter_markdown_context(row):
    row['source_all'] = [x for x in row['source_all'] if x != row['source']]
    return row