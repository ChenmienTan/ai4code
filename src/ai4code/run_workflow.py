import warnings

import wandb

from ai4code.workflows.pointwise.m2c_workflow import M2CWorkflow
from ai4code.workflows.pointwise.pointwise_workflow import OrigPointwiseWorkflow

warnings.simplefilter(action='ignore', category=FutureWarning)
import logging.config
import os
import pathlib

import hydra
import yaml
from omegaconf import OmegaConf

from ai4code.workflows.finetune.fine_tune_workflow import FineTuneWorkflow
from ai4code.workflows.baseline.baseline_workflow import BaselineWorkflow
from ai4code.logging_utils import register_tqdm_logger, flatten

HERE = pathlib.Path(__file__).parent.resolve()

# code cleaning: https://www.kaggle.com/code/haithamaliryan/ai4code-extract-all-functions-variables-names
# --train_file /Users/victormay/Documents/data/AI4Code/rebart_input/train.jsonl --eval_data_file /Users/victormay/Documents/data/AI4Code/rebart_input/dev.jsonl --out_dir . --model_type facebook/bart-large --model_name_or_path facebook/bart-large --device 1 --do_train --do_eval --save_total_limit 1 --num_train_epochs 1 --logging_steps 3000 --gradient_accumulation_steps 8 --train_batch_size 4 --eval_batch_size 8 --overwrite_out_dir --max_input_length 1024 --max_output_length 40 --task index_with_sep --overwrite_cache

logging.config.dictConfig(yaml.safe_load(
    pathlib.Path(os.path.join(HERE, 'config', 'logging.yaml')).read_text()))

logger = logging.getLogger(__name__)


@hydra.main(config_path="config", config_name="config", version_base=None)
def main(cfg: OmegaConf):
    register_tqdm_logger()
    logger.info('cwd={}'.format(os.getcwd()))

    wandb_mode = 'online' if cfg.wandb.enabled else 'disabled'
    wandb.init(project=cfg.wandb.project, config=flatten(dict(cfg)), mode=wandb_mode)

    if cfg.workflow.name == 'baseline':
        workflow = BaselineWorkflow(cfg.workflow.args, cfg.env)
    elif cfg.workflow.name == 'finetune':
        workflow = FineTuneWorkflow(cfg.workflow, cfg.env)
    elif cfg.workflow.name == 'pointwise':
        workflow = OrigPointwiseWorkflow(cfg.workflow, cfg.env)
    elif cfg.workflow.name == 'm2c':
        workflow = M2CWorkflow(cfg.workflow, cfg.env)
    else:
        raise NotImplementedError

    if cfg.workflow.do_train:
        workflow.train(cfg.env.raw_data_path, cfg.env.artifacts_path)
    else:
        logger.info('Skipping training due to config')

    if cfg.workflow.do_predict:
        if os.path.exists(cfg.env.artifacts_path):
            preds = workflow.predict(cfg.env.raw_data_path, cfg.env.artifacts_path)
            preds.to_pickle(os.path.join(cfg.env.submission_path, 'preds.pkl'))
            workflow.create_submission(preds, cfg.env.submission_path)
        else:
            logger.error('Artifacts dir not found')
    else:
        logger.info('Skipping inference due to config')

    logger.info('Finished successfully')

if __name__ == '__main__':
    main()

