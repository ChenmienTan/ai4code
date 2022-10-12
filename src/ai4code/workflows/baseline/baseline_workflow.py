import os
import pickle
from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse
import numpy as np
from xgboost import XGBRanker

from ai4code.data_splitter import create_binary_split
from ai4code.evaluation import kendall_tau
from ai4code.raw_data_parsing import get_raw_data
from ai4code.workflows.workflow import StandardWorkflow, Preprocessor, Postprocessor


class BaselinePreprocessor(Preprocessor):
    def __init__(self):
        self.save_file_name = 'preprocessor.p'
        self.tfidf = TfidfVectorizer(min_df=0.01)

    def save(self, save_dir):
        file_path = os.path.join(save_dir, self.save_file_name)
        with open(file_path, 'wb') as fout:
            pickle.dump(self.tfidf, fout)

    def load(self, save_dir):
        file_path = os.path.join(save_dir, self.save_file_name)
        with open(file_path, 'rb') as fin:
            self.tfidf = pickle.load(fin)

    def fit_transform(self, data):
        self.tfidf.fit(data['source'].astype(str))
        return self.transform(data)

    def transform(self, data):
        X_train = self.tfidf.transform(data['source'].astype(str))

        # Add code cell ordering
        X_train = sparse.hstack((
            X_train,
            np.where(
                data['cell_type'] == 'code',
                data.groupby(['id', 'cell_type']).cumcount().to_numpy() + 1,
                0,
            ).reshape(-1, 1)
        ))

        if 'label' in data.columns:
            y_train = data.label.to_numpy()
            # Number of cells in each notebook
            groups = data.groupby('id').size().to_numpy()

            return X_train, y_train, groups
        else:
            return X_train


class BaselinePostProcessor(Postprocessor):
    def __init__(self):
        self.save_file_name = 'postprocessor.p'

    def fit_transform(self, raw_data, prep_data, preds):
        pass


    def transform(self, raw_data, prep_data, preds):
        y_pred = pd.DataFrame({'rank': preds}, index=raw_data.index)
        y_pred = (
            y_pred
                .sort_values(['id', 'rank'])  # Sort the cells in each notebook by their rank.
                # The cell_ids are now in the order the model predicted.
                .reset_index('cell_id')  # Convert the cell_id index into a column.
                .groupby('id')['cell_id'].apply(list)  # Group the cell_ids for each notebook into a list.
        )
        return y_pred

    def save(self, save_dir):
        pass

    def load(self, save_dir):
        pass


class BaselineWorkflow(StandardWorkflow):
    MODEL_FILENAME = 'model.txt'

    def __init__(self, cfg, env):
        self.env = env
        self.cfg = cfg

    def get_raw_data(self, input_data_path, is_train):
        return get_raw_data(self.env.raw_data_path, is_train, self.cfg.data.max_data_size)

    def get_folds(self, features, labels):
        df_ancestors = pd.read_csv(Path(self.env.raw_data_path) / 'train_ancestors.csv', index_col='id')
        return create_binary_split(features, df_ancestors, self.cfg.data.test_ratio)

    def do_train(self, X_train, y_train, X_val, y_val, fold_idx, *kwargs):
        # Training set
        groups = kwargs[0]

        model = XGBRanker(
            min_child_weight=10,
            subsample=0.5,
            tree_method='hist',
        )
        model.fit(X_train, y_train, group=groups)
        return model

    def get_evaluation_score(self, preds, labels):
        return kendall_tau(labels, preds)

    def make_preprocessor(self):
        return BaselinePreprocessor()

    def make_postprocessor(self):
        return BaselinePostProcessor()

    def save_model(self, model, model_dir):
        model.save_model(os.path.join(model_dir, BaselineWorkflow.MODEL_FILENAME))

    def load_model(self, model_dir):
        model_xgb = XGBRanker()
        model_xgb.load_model(os.path.join(model_dir, BaselineWorkflow.MODEL_FILENAME))
        return model_xgb

    def ensemble_preds(self, preds, postprocessed_preds):
        return postprocessed_preds[0]

    def create_submission(self, test_preds, output_dir):
        y_submit = (
            test_preds
                .apply(' '.join)  # list of ids -> string of ids
                .rename_axis('id')
                .rename('cell_order')
        )
        y_submit.to_csv(os.path.join(output_dir, 'submission.csv'))

