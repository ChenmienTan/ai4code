import logging
import os
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

class Workflow(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def train(self, input_data_path, artifacts_path):
        pass

    @abstractmethod
    def predict(self, input_data_path, artifacts_path):
        pass

    @abstractmethod
    def create_submission(self, test_preds, output_dir):
        pass


class CVWorkflow(Workflow):
    OOF_DF_FILENAME = '../oof_df.pkl'
    FOLD_DIR_TEMPLATE = 'fold_{}'

    @abstractmethod
    def get_fold_data(self, fold_idx):
        pass

    @abstractmethod
    def get_num_folds(self):
        pass

    @abstractmethod
    def get_train_fold_idx(self):
        pass

    @abstractmethod
    def do_train(self, train_data, val_data, fold_dir):
        pass

    def train(self, input_data_path, artifacts_path):
        logger.info('Running preprocessing')

        folds_to_train = self.get_train_fold_idx()
        all_folds = list(range(self.get_num_folds()))

        oof_dfs = []

        for fold_idx in all_folds:
            train_data, val_data = self.get_fold_data(fold_idx)

            logger.info('train data size={}, validation data size={}'.format(len(train_data), len(val_data)))

            oof_df = val_data.copy()
            oof_df['oof_pred'] = np.nan
            if fold_idx not in folds_to_train:
                oof_dfs.append(oof_df)
                logger.info(f'Skipping fold {fold_idx}')
                continue
            logger.info(f'Training fold {fold_idx}')

            fold_dir = os.path.join(artifacts_path, CVWorkflow.FOLD_DIR_TEMPLATE.format(fold_idx))
            if not os.path.exists(fold_dir):
                os.makedirs(fold_dir)

            model, y_pred = self.do_train(train_data, val_data, fold_dir)

            oof_df['oof_pred'] = y_pred
            oof_dfs.append(oof_df)

            logger.info(f'Done training fold {fold_idx} ')

        combined_oof_df = pd.concat(oof_dfs)
        combined_oof_df.to_pickle(os.path.join(artifacts_path, StandardWorkflow.OOF_DF_FILENAME))


class StandardWorkflow(Workflow):
    OOF_DF_FILENAME = '../oof_df.pkl'
    FOLD_DIR_TEMPLATE = 'fold_{}'

    def predict(self, input_data_path, artifacts_path):
        logger.info('Running inference')
        test_data = self.get_raw_data(input_data_path, is_train=False)
        all_preds = []
        all_postprocessed_preds = []

        for d in os.listdir(artifacts_path):
            fold_dir = os.path.join(artifacts_path, d)
            if os.path.isdir(fold_dir) and d.startswith('fold_'):
                preprocessor = self.make_preprocessor()
                preprocessor.load(fold_dir)
                X_test = preprocessor.transform(test_data)

                model = self.load_model(fold_dir)
                fold_preds = model.predict(X_test)

                all_preds.append(fold_preds)

                postprocessor = self.make_postprocessor()
                postprocessor.load(d)

                postprocessed_preds = postprocessor.transform(test_data, X_test, fold_preds)

                all_postprocessed_preds.append(postprocessed_preds)
        return self.ensemble_preds(all_preds, all_postprocessed_preds)

    def train(self, input_data_path, artifacts_path):
        logger.info('Running preprocessing')
        raw_features, raw_labels = self.get_raw_data(input_data_path, is_train=True)
        raw_features['fold_idx'], raw_features['to_train'] = self.get_folds(raw_features, raw_labels)
        raw_features['label'] = raw_labels

        folds_to_train = raw_features[raw_features.to_train == 1].fold_idx.unique().astype('int').tolist()
        all_folds = raw_features.fold_idx.unique().astype('int').tolist()

        oof_dfs = []

        for fold_idx in all_folds:
            train_data = raw_features[raw_features.fold_idx == fold_idx]
            val_data = raw_features[raw_features.fold_idx != fold_idx]

            oof_df = val_data.copy()
            oof_df['oof_pred'] = np.nan
            if fold_idx not in folds_to_train:
                oof_dfs.append(oof_df)
                logger.info(f'Skipping fold {fold_idx}')
                continue
            logger.info(f'Training fold {fold_idx}')
            preprocessor = self.make_preprocessor()
            X_train, y_train, train_extra = preprocessor.fit_transform(train_data)
            assert len(X_train) == len(y_train)
            X_val, y_val, _ = preprocessor.transform(val_data)
            assert len(X_val) == len(y_val)

            model = self.do_train(X_train, y_train, X_val, y_val, fold_idx, train_extra)
            y_pred = model.predict(X_val) # TODO: refactor

            post_processor = self.make_postprocessor()
            post_processor.data = val_data # TODO: refactor
            processed_preds = post_processor.transform(val_data, X_val, y_pred)
            #processed_labels = post_processor.transform(val_data, X_val, y_val)

            fold_score = self.get_evaluation_score(processed_preds, val_data.label)
            oof_df['oof_pred'] = processed_preds
            oof_dfs.append(oof_df)

            fold_dir = os.path.join(artifacts_path, StandardWorkflow.FOLD_DIR_TEMPLATE.format(fold_idx))
            if not os.path.exists(fold_dir):
                os.makedirs(fold_dir)
            preprocessor.save(fold_dir)
            self.make_postprocessor().save(fold_dir)

            self.save_model(model, fold_dir)
            logger.info(f'Done training fold {fold_idx} with score {fold_score}')

        combined_oof_df = pd.concat(oof_dfs)
        combined_oof_df.to_pickle(os.path.join(artifacts_path, StandardWorkflow.OOF_DF_FILENAME))
        #for fold_idx in folds_to_train

    @abstractmethod
    def save_model(self, model, model_dir):
        pass

    @abstractmethod
    def load_model(self, model_dir):
        pass

    @abstractmethod
    def make_preprocessor(self):
        pass

    @abstractmethod
    def make_postprocessor(self):
        pass

    @abstractmethod
    def get_raw_data(self, input_data_path, is_train):
        pass

    @abstractmethod
    def get_folds(self, features, labels):
        pass

    @abstractmethod
    def do_train(self, X_train, y_train, X_val, y_val, fold_idx, *kwargs):
        pass

    @abstractmethod
    def get_evaluation_score(self, preds, labels):
        pass

    @abstractmethod
    def ensemble_preds(self, preds, postprocessed_preds):
        pass


class Preprocessor(ABC):
    @abstractmethod
    def fit_transform(self, data):
        pass

    @abstractmethod
    def transform(self, data):
        pass

    @abstractmethod
    def save(self, save_dir):
        pass

    @abstractmethod
    def load(self, save_dir):
        pass


class Postprocessor(ABC):
    @abstractmethod
    def fit_transform(self, raw_data, prep_data, preds):
        pass

    @abstractmethod
    def transform(self, raw_data, prep_data, preds):
        pass

    @abstractmethod
    def save(self, save_dir):
        pass

    @abstractmethod
    def load(self, save_dir):
        pass

