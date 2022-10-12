import logging
import os
from pathlib import Path

import pandas as pd
from datasets import concatenate_datasets
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from ai4code.data_splitter import create_binary_split
from ai4code.evaluation import kendall_tau
from ai4code.raw_data_parsing import get_raw_data
from ai4code.transformer_utils import train_loop, get_model_checkpoint
from ai4code.workflows.workflow import StandardWorkflow
from ai4code.workflows.finetune.postprocessing import FineTunePostProcessor
from ai4code.workflows.finetune.preprocessing import FineTunePreprocessor
from ai4code.workflows.finetune.utils import ModelWrapper

from ai4code.workflows.finetune.utils import ranks_to_cell_ids


logger = logging.getLogger(__name__)

class FineTuneWorkflow(StandardWorkflow):
    MODEL_FILENAME = 'model.txt'

    def __init__(self, cfg, env):
        self.env = env
        self.cfg = cfg.args

        if cfg.do_train:
            logger.info('Creating tokenizer from hub')
            self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model_name)
        else: # TODO: currently not supporting a separate tokenizer in each fold
            fold_dirs = [x for x in os.listdir(self.env.artifacts_path) if x.startswith('fold')]
            checkpoint_dir = get_model_checkpoint(os.path.join(self.env.artifacts_path, fold_dirs[0]))
            logger.info(f'Creating tokenizer from checkpoint {checkpoint_dir}')
            self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)

    def get_raw_data(self, input_data_path, is_train):
        return get_raw_data(self.env.raw_data_path, is_train, self.cfg.data.max_data_size)

    def get_folds(self, features, labels):
        df_ancestors = pd.read_csv(Path(self.env.raw_data_path) / 'train_ancestors.csv', index_col='id')
        return create_binary_split(features, df_ancestors, self.cfg.data.test_ratio)

    def do_train(self, X_train, y_train, X_val, y_val, fold_idx, *kwargs):
        model = AutoModelForSequenceClassification.from_pretrained(self.cfg.model_name, num_labels=1)

        train_ds = concatenate_datasets([X_train, y_train], axis=1)
        val_ds = concatenate_datasets([X_val, y_val], axis=1)
        compute_metrics = None
        eval_metric_name = None
        output_dir = os.path.join(self.env.artifacts_path, f'fold_{fold_idx}')
        trainer = train_loop(self.cfg.training, model, self.tokenizer, train_ds, val_ds,
                                     compute_metrics, eval_metric_name, output_dir)
        fold_path = Path(trainer.state.best_model_checkpoint).parent.absolute()

        return self.load_model(fold_path)

    def get_evaluation_score(self, preds, labels):
        _preds = ranks_to_cell_ids(preds)

        labels_df = pd.DataFrame(labels)
        labels_df['rank'] = labels_df.label

        _labels = ranks_to_cell_ids(labels_df)

        return kendall_tau(_labels, _preds)

    def make_preprocessor(self):
        return FineTunePreprocessor(self.tokenizer, max_len=self.cfg.tokenizer.max_len)

    def make_postprocessor(self):
        return FineTunePostProcessor()

    def save_model(self, model, model_dir):
        pass
        #model.save_model(os.path.join(model_dir, BaselineWorkflow.MODEL_FILENAME))

    def load_model(self, model_dir):
        checkpoint_dir = get_model_checkpoint(model_dir)
        return ModelWrapper(model=AutoModelForSequenceClassification.from_pretrained(checkpoint_dir),
                     tokenizer=self.tokenizer,
                     batch_size=self.cfg.inference.batch_size,
                     target_device=self.env.device)

    def ensemble_preds(self, preds, postprocessed_preds):
        return postprocessed_preds[0]

    def create_submission(self, test_preds, output_dir):
        y_submit = (
            ranks_to_cell_ids(test_preds)
                .apply(' '.join)  # list of ids -> string of ids
                .rename_axis('id')
                .rename('cell_order')
        )
        y_submit.to_csv(os.path.join(output_dir, 'submission.csv'))