import pandas as pd

import torch
import torch.nn as nn
from torch.autograd import Variable

import numpy as np
from scipy.special import softmax


df_train = pd.read_pickle('/data/rnn/train.pkl')
df_valid = pd.read_pickle('/data/rnn/val.pkl')


class Model(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        super(Model, self).__init__()

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.rnn = nn.GRU(input_size, hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_size)

    def forward(self, x):
        batch_size = x.size(0)

        hidden = self.init_hidden(batch_size)

        out, hidden = self.rnn(x, hidden)

        out = out.contiguous().view(-1, self.hidden_dim)
        out = self.fc(out)

        return out, hidden

    def init_hidden(self, batch_size):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        # We'll send the tensor holding the hidden state to the device we specified earlier as well
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        return hidden


X_train = df_train[df_train.cell_type == 'markdown'].softmax.values
y_train = df_train[df_train.cell_type == 'markdown']['rank'].rank(pct=True)

X_test = df_valid[df_valid.cell_type == 'markdown'].softmax.values
y_test = df_valid[df_valid.cell_type == 'markdown']['rank'].rank(pct=True)

y_train = [[x] for x in y_train]
y_test = [[x] for x in y_test]

def pad_to_fixed(X, max_len, padding_value):
    seq_lens = [len(x) for x in X]

    # pad first seq to desired length
    X[0] = nn.ConstantPad1d((0, max_len - X[0].shape[0]), 0)(X[0])

    # pad all seqs to desired length
    return nn.utils.rnn.pad_sequence(X, padding_value=padding_value)


def get_feat_tensors(X, max_len):
    return pad_to_fixed([torch.Tensor(x) for x in X.tolist()], max_len=max_len, padding_value=-100).transpose(0, 1)




max_len = 150


X_train_tensors = (torch.Tensor(get_feat_tensors(X_train, max_len)))  # Variable
X_test_tensors = (torch.Tensor(get_feat_tensors(X_test, max_len)))

y_train_tensors = (torch.Tensor(y_train))
y_test_tensors = (torch.Tensor(y_test))

# reshaping to rows, timestamps, features

X_train_tensors_final = torch.reshape(X_train_tensors, (X_train_tensors.shape[0], 1, X_train_tensors.shape[1]))
X_test_tensors_final = torch.reshape(X_test_tensors, (X_test_tensors.shape[0], 1, X_test_tensors.shape[1]))

num_epochs = 1000
learning_rate = 0.1  # 0.001 lr

model = Model(input_size=X_train_tensors_final.shape[-1], output_size=1, hidden_dim=5, n_layers=1)

criterion = torch.nn.MSELoss()  # mean-squared error for regression
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    optimizer.zero_grad()  # caluclate the gradient, manually setting to 0
    outputs = model(X_train_tensors_final)  # forward pass

    # obtain the loss function
    loss = criterion(outputs, y_train_tensors)

    loss.backward()  # calculates the loss of the loss function

    optimizer.step()  # improve from loss, i.e backprop
    if epoch % 5 == 0:
        print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))

train_predict, _ = model(X_test_tensors_final)  # forward pass
data_predict = train_predict.data.numpy()  # numpy conversion
preds_and_y = list(zip(data_predict.squeeze().tolist(), [x[0] for x in y_test]))
preds_and_y