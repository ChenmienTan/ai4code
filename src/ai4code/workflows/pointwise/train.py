import logging
import os
import sys

import torch
import wandb
from sklearn.metrics import mean_squared_error
from torch import autocast
from torch.cuda.amp import GradScaler
from tqdm import tqdm
import numpy as np
from transformers import get_linear_schedule_with_warmup, AutoModelForSequenceClassification

from ai4code.workflows.pointwise.evaluation import get_kendall_tau

WARMUP_STEPS_RATIO = 0.1

MAX_NORM = 1.0

logger = logging.getLogger(__name__)


def read_data(data, device):
    return (data['input_ids'].to(device), data['attention_mask'].to(device)), data['labels'].to(device)


def validate(model, val_loader, device):
    model = model.to(device)
    model.eval()

    tbar = tqdm(val_loader, file=sys.stdout)

    preds = []
    labels = []

    with torch.no_grad():
        for idx, data in enumerate(tbar):
            inputs, target = read_data(data, device)

            pred = model(inputs[0], inputs[1])[0]

            preds.append(pred.detach().cpu().numpy().ravel())
            labels.append(target.detach().cpu().numpy().ravel())

    return np.concatenate(labels), np.concatenate(preds)


def adjust_lr(optimizer, lr_decay_rate):
    for p in optimizer.param_groups:
        p['lr'] *= lr_decay_rate


def train(model, train_loader, val_loader, epochs, device, df_orders, val_df, lr, lr_decay_rate):
    np.random.seed(0)

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)  # get_optimizer(model)
    num_train_optimization_steps = int(epochs * len(train_loader))
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=WARMUP_STEPS_RATIO * num_train_optimization_steps,
                                                num_training_steps=num_train_optimization_steps)

    criterion = torch.nn.MSELoss()

    scaler = GradScaler()

    best_checkpoint_dir = None
    best_val_kd = -1
    checkpoint_idx = 0

    for e in range(epochs):
        model.train()
        tbar = tqdm(train_loader, file=sys.stdout)

        #adjust_lr(optimizer, lr_decay_rate)

        loss_list = []
        preds = []
        labels = []

        for idx, data in enumerate(tbar):
            inputs, target = read_data(data, device)

            with autocast('cpu' if device == 'cpu' else 'cuda'):
                pred = model(inputs[0], inputs[1])[0]
                loss = criterion(pred, target)

            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_NORM)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

            loss_list.append(loss.detach().cpu().item())
            preds.append(pred.detach().cpu().numpy().ravel())
            labels.append(target.detach().cpu().numpy().ravel())

            avg_loss = np.round(np.mean(loss_list), 4)

            tbar.set_description(f"Epoch {e + 1} Loss: {avg_loss}")

        y_val, y_pred = validate(model, val_loader, device)

        val_kd = get_kendall_tau(df_orders, val_df, y_pred)

        val_mse = np.round(mean_squared_error(y_val, y_pred), 4)
        logger.info(f"Validation MSE: {val_mse}, Kendall Tau: {val_kd}")
        wandb.log({'validation_mse': val_mse, 'validation_kd': val_kd})

        if val_kd > best_val_kd:
            best_val_kd = val_kd
            logger.info(f'New best Kendall Tau: {val_kd}')
            best_checkpoint_dir = f'checkpoint_{checkpoint_idx}'
            logger.info(f'Saving checkpoint to {best_checkpoint_dir}')
            model.save_pretrained(best_checkpoint_dir)
            checkpoint_idx += 1
        else:
            logger.info('Early stopping')
            break



    logger.info(f'Using model from checkpoint {best_checkpoint_dir}')
    model = AutoModelForSequenceClassification.from_pretrained(best_checkpoint_dir)

    return model, y_pred, val_kd