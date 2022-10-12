import gc
import logging
import os
import shutil
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import Trainer, EarlyStoppingCallback, AutoModelForSequenceClassification, TrainingArguments, \
    DataCollatorWithPadding

from ai4code.os_utils import log_disk_usage

LOGGER = logging.getLogger(__name__)


def model_predict(model, tokenizer, test_ds, eval_batch_size, target_device):
    LOGGER.info(f'Predicting with target device {target_device}')
    model = model.to(target_device)
    test_data_loader = DataLoader(test_ds, batch_size=eval_batch_size, collate_fn=DataCollatorWithPadding(tokenizer))
    data_iter = iter(test_data_loader)

    all_preds = []
    for inputs in data_iter:
        inputs = {k:v.to(target_device) for k,v in inputs.items()}
        with torch.no_grad():
            batch_preds = model(**inputs).logits.detach().cpu().tolist()
            all_preds += batch_preds

    all_preds = np.array(all_preds).reshape(-1)
    return all_preds

def get_model_checkpoint(model_dir):
    fold_checkpoint_dirs = [x for x in os.listdir(model_dir) if x.startswith('checkpoint')]
    if len(fold_checkpoint_dirs) == 1:
        return os.path.join(model_dir, fold_checkpoint_dirs[0])
    else:
        raise IOError('wrong number of checkpoint folders!')

def get_fold_model_checkpoints(dataset_dir):
    all_fold_checkpoint_dirs = []
    for fold_dir in os.listdir(dataset_dir):
        if fold_dir.startswith('model') and os.path.isdir(os.path.join(dataset_dir, fold_dir)):
            all_fold_checkpoint_dirs.append(get_model_checkpoint(os.path.join(dataset_dir, fold_dir)))
    return all_fold_checkpoint_dirs

def make_compute_metrics(eval_pred, metric_name, metric_func):
    predictions, labels = eval_pred

    if isinstance(predictions, tuple):
        predictions = predictions[0]

    predictions = predictions.reshape(len(predictions))
    return {
        metric_name: metric_func(labels, predictions)
    }


def del_old_checkpoints(best_model_checkpoint_path):    
    parent_dir = Path(best_model_checkpoint_path).parent.absolute()     
    basename = os.path.basename(best_model_checkpoint_path)
    for subdir in os.listdir(parent_dir):
        if subdir.startswith('checkpoint') and not subdir == basename:
            LOGGER.info(f'deleting dir {subdir} in {parent_dir}')
            shutil.rmtree(os.path.join(parent_dir, subdir))

def train_loop(cfg, model, tokenizer, train_ds, val_ds, compute_metrics, eval_metric_name, output_dir):
    #LOGGER.info(f"========== fold: {fold_idx} ==========")
    #log_disk_usage()

    #tokenizer = AutoTokenizer.from_pretrained(cfg.model_path)
    #model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path, num_labels=1)

    #raw_datasets = load_train_ds(fold_idx)
    #datasets = transform_datasets_for_train(raw_datasets, tokenizer)

    train_args = TrainingArguments(
        output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        num_train_epochs=cfg.num_train_epochs,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        load_best_model_at_end=True,
        metric_for_best_model=eval_metric_name,
        greater_is_better=True,
        save_total_limit=1,
        log_level='error',
        optim="adamw_torch",
        fp16=cfg.fp16 if torch.cuda.is_available() else False,
        gradient_checkpointing=cfg.gradient_checkpointing
    )

    trainer = Trainer(
        model,
        train_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=cfg.early_stopping_patience)]
    )

    trainer.evaluate()
    trainer.train()

    #oof_preds = trainer.predict(val_ds.predictions.reshape(-1)

    # Cleaning up
    os.remove(os.path.join(trainer.state.best_model_checkpoint, 'optimizer.pt'))
    del_old_checkpoints(trainer.state.best_model_checkpoint)

    torch.cuda.empty_cache()
    gc.collect()

    LOGGER.info(f'Finished training model {output_dir}')
    log_disk_usage()

    #checkpoint_path = trainer.state.best_model_checkpoint

    return trainer