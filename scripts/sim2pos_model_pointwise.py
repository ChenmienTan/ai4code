from pathlib import Path

import pandas as pd
import scipy.signal
import sklearn

import torch
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pylab as plt
import numpy as np
from scipy.special import softmax
from sklearn.metrics import mean_squared_error
from torch.utils.data import DataLoader, TensorDataset
from transformers import get_linear_schedule_with_warmup

from ai4code.evaluation import kendall_tau

df_train = pd.read_pickle('/data/rnn/train.pkl')
df_valid = pd.read_pickle('/data/rnn/val.pkl')

raw_data_path = '/Users/victormay/Documents/data/AI4Code/data'

batch_size = 16
num_samples = 100
num_epochs = 5000
learning_rate = 0.001  # 0.001 lr


class Model(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        super(Model, self).__init__()

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.fc = []
        self.fc.append(nn.Linear(input_size, hidden_dim[0]))
        for idx, h in enumerate(hidden_dim[1:]):
            self.fc.append(nn.Linear(hidden_dim[idx], h))
        self.fc_out = nn.Linear(hidden_dim[-1], output_size)

        self.dropout = nn.Dropout(p=0.1)
        self.activation = nn.ReLU()

        self.conv = nn.Conv1d(1, 1, 3)

    def forward(self, x):
        out = x
        #out = self.conv(torch.unsqueeze(out, 1))

        for layer in self.fc:
            out = layer(out)
            out = self.dropout(out)
            out = self.activation(out)

        return self.fc_out(out)

def train_loop(train_dataloader, X_test_tensors, y_test_tensors):
    model = Model(input_size=num_samples, output_size=1, hidden_dim=[200], n_layers=1)

    criterion = torch.nn.MSELoss()  # mean-squared error for regression
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.1)

    train_losses = []
    test_losses = []

    best_loss = 100000
    num_dec = 0

    num_train_optimization_steps = int(num_epochs * len(train_dataloader))
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0.1 * num_train_optimization_steps,
                                                num_training_steps=num_train_optimization_steps)

    for epoch in range(num_epochs):
        for i, data in enumerate(train_dataloader):
            model.train()
            optimizer.zero_grad()  # caluclate the gradient, manually setting to 0
            outputs = model(X_train_tensors)  # forward pass

            # obtain the loss function
            loss = criterion(outputs, y_train_tensors)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            loss.backward()  # calculates the loss of the loss function

            optimizer.step()  # improve from loss, i.e backprop
            scheduler.step()

        if epoch % 1 == 0:
            model.eval()
            with torch.no_grad():
                test_loss = criterion(model(X_test_tensors), y_test_tensors)
                #print("Epoch: %d, train loss: %1.5f, test loss %1.5f" % (epoch, loss.item(), test_loss.item()))
            train_losses.append(loss.item())
            test_losses.append(test_loss.item())

            if test_loss.item() > best_loss:
                num_dec += 1
                if num_dec >= 10:
                    print('early stopping')
                    break
            else:
                num_dec = 0
                best_loss = test_loss.item()

    #plt.plot(np.transpose([train_losses, test_losses]))
    #plt.legend(['Train', 'Test'])
    return model

def get_feat_tensors(X, num_samples):
    return [ scipy.signal.resample(x, num_samples) for x in X]




X_train = df_train[df_train.cell_type == 'markdown'].softmax.values
y_train = df_train[df_train.cell_type == 'markdown']['rank'].rank(pct=True)

X_test = df_valid[df_valid.cell_type == 'markdown'].softmax.values
y_test = df_valid[df_valid.cell_type == 'markdown']['rank'].rank(pct=True)

y_train = [[x] for x in y_train]
y_test = [[x] for x in y_test]

df_ancestors = pd.read_csv(Path(raw_data_path) / 'train_ancestors.csv', index_col='id')

X_train_tensors = (torch.Tensor(get_feat_tensors(X_train, num_samples)))  # Variable
X_test_tensors = (torch.Tensor(get_feat_tensors(X_test, num_samples)))

y_train_tensors = (torch.Tensor(y_train))
y_test_tensors = (torch.Tensor(y_test))


#splitter = sklearn.model_selection.KFold(n_splits=5, shuffle=False, random_state=None)

#for X_train_tensors_, y_train_tensors_, in splitter.split(X_train_tensors, y_train_tensors):

train_dataloader = DataLoader(TensorDataset(X_train_tensors, y_train_tensors), batch_size=batch_size, shuffle=True)

models = []
for idx in range(1):
    print('training model {}'.format(idx))
    model = train_loop(train_dataloader, X_test_tensors, y_test_tensors)
    models.append(model)


all_preds = [model(X_test_tensors).data.numpy() for model in models]
data_predict = np.mean(all_preds, axis=0) # numpy conversion
#preds_and_y = list(zip(data_predict.squeeze().tolist(), [x[0] for x in y_test]))
#print(preds_and_y)
plt.figure()
plt.scatter([x[0] for x in y_test], data_predict.squeeze().tolist())
print('MSE={}'.format(mean_squared_error(data_predict, y_test)))
plt.show()


df_orders = pd.read_csv(
            Path(raw_data_path) / 'train_orders.csv',
            index_col='id',
            squeeze=True,
        ).str.split()  # Split the string representation of cell_ids into a list


def get_rank_preds(group):
    temp = 40
    group["rank4pred"] = group.groupby(["id", "cell_type"])["rank"].rank(pct=False)

    df_mark = group[group.cell_type == 'markdown']
    df_code = group[group.cell_type == 'code']

    code_ranks = df_code['rank4pred'].values.tolist()
    markdown_softmax = softmax(temp * mark_code_sim, axis=1)
    rep_code_ranks = np.transpose(np.repeat(np.expand_dims(code_ranks, -1), markdown_softmax.shape[0], axis=1))
    group['pred'] = group['rank4pred']
    preds = np.average(rep_code_ranks, weights=markdown_softmax, axis=1)
    group.loc[group["cell_type"] == "markdown", 'pred'] = preds - 0.5

y_dummy = df_valid.sort_values("pred").groupby('id')['cell_id'].apply(list)
val_kd = kendall_tau(df_orders.loc[y_dummy.index], y_dummy)
print(val_kd)
pass
# TODO: batching, kfold, kendall tau eval, schedule