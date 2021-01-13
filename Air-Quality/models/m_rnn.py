import torch
torch.manual_seed(1)
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.autograd import Variable
from torch.nn.parameter import Parameter

import math
import utils
import argparse
import data_loader

#from ipdb import set_trace
from sklearn import metrics

SEQ_LEN = 36


class FeatureRegression(nn.Module):
    def __init__(self, input_size):
        super(FeatureRegression, self).__init__()
        self.build(input_size)

    def build(self, input_size):
        self.W = Parameter(torch.Tensor(input_size, input_size))
        self.b = Parameter(torch.Tensor(input_size))

        m = torch.ones(input_size, input_size) - torch.eye(input_size, input_size)
        self.register_buffer('m', m)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(0))
        self.W.data.uniform_(-stdv, stdv)
        if self.b is not None:
            self.b.data.uniform_(-stdv, stdv)

    def forward(self, x):
        z_h = F.linear(x, self.W * Variable(self.m), self.b)
        return z_h

class TemporalDecay(nn.Module):
    def __init__(self, input_size, output_size, diag = False):
        super(TemporalDecay, self).__init__()
        self.diag = diag

        self.build(input_size, output_size)

    def build(self, input_size, output_size):
        self.W = Parameter(torch.Tensor(output_size, input_size))
        self.b = Parameter(torch.Tensor(output_size))

        if self.diag == True:
            assert(input_size == output_size)
            m = torch.eye(input_size, input_size)
            self.register_buffer('m', m)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(0))
        self.W.data.uniform_(-stdv, stdv)
        if self.b is not None:
            self.b.data.uniform_(-stdv, stdv)

    def forward(self, d):
        if self.diag == True:
            gamma = F.relu(F.linear(d, self.W * Variable(self.m), self.b))
        else:
            gamma = F.relu(F.linear(d, self.W, self.b))
        gamma = torch.exp(-gamma)
        return gamma

class Model(nn.Module):
    def __init__(self, rnn_hid_size, impute_weight):
        super(Model, self).__init__()

        self.rnn_hid_size = rnn_hid_size
        self.impute_weight = impute_weight

        self.build()

    def build(self):
        self.rnn_cell = nn.LSTMCell(36 * 3, self.rnn_hid_size)
        self.pred_rnn = nn.LSTM(36, self.rnn_hid_size, batch_first = True)

        self.temp_decay_h = TemporalDecay(input_size = 36, output_size = self.rnn_hid_size, diag = False)
        self.temp_decay_x = TemporalDecay(input_size = 36, output_size = 36, diag = True)

        self.hist_reg = nn.Linear(self.rnn_hid_size * 2, 36)
        self.feat_reg = FeatureRegression(36)

        self.weight_combine = nn.Linear(36 * 2, 36)

        self.dropout = nn.Dropout(p = 0.25)
        self.out = nn.Linear(self.rnn_hid_size, 1)

    def get_hidden(self, data, direct):
        values = data[direct]['values']
        masks = data[direct]['masks']
        deltas = data[direct]['deltas']

        hiddens = []

        h = Variable(torch.zeros((values.size()[0], self.rnn_hid_size)))
        c = Variable(torch.zeros((values.size()[0], self.rnn_hid_size)))

        if torch.cuda.is_available():
            h, c = h.cuda(), c.cuda()

        for t in range(SEQ_LEN):
            hiddens.append(h)

            x = values[:, t, :]
            m = masks[:, t, :]
            d = deltas[:, t, :]

            inputs = torch.cat([x, m, d], dim = 1)

            h, c = self.rnn_cell(inputs, (h, c))

        return hiddens


    def forward(self, data, direct):
        # Original sequence with 24 time steps
        hidden_forward = self.get_hidden(data, 'forward')
        hidden_backward = self.get_hidden(data, 'backward')[::-1]

        values = data[direct]['values']
        masks = data[direct]['masks']
        deltas = data[direct]['deltas']

        x_loss = 0.0
        y_loss = 0.0

        imputations = []

        for t in range(SEQ_LEN):
            x = values[:, t, :]
            m = masks[:, t, :]
            d = deltas[:, t, :]

            hf = hidden_forward[t]
            hb = hidden_backward[t]
            h = torch.cat([hf, hb], dim = 1)

            x_h = self.hist_reg(h)
            x_f = self.feat_reg(x)

            alpha = torch.sigmoid(self.weight_combine(torch.cat([m, d], dim = 1)))

            x_c = alpha * x_h + (1 - alpha)

            x_loss += torch.sum(torch.abs(x - x_c) * m) / (torch.sum(m) + 1e-5)

            imputations.append(x_c.unsqueeze(dim = 1))

        imputations = torch.cat(imputations, dim = 1)

        evals = data[direct]['evals']
        eval_masks = data[direct]['eval_masks']

        h = Variable(torch.zeros((values.size()[0], self.rnn_hid_size)))
        c = Variable(torch.zeros((values.size()[0], self.rnn_hid_size)))

        if torch.cuda.is_available():
            h, c = h.cuda(), c.cuda()

        imputations = Variable(imputations.data, requires_grad = False)
        out, (h, c) = self.pred_rnn(imputations)



        return {'loss': x_loss * self.impute_weight,\
                'imputations': imputations,\
                'evals': evals, 'eval_masks': eval_masks}

    def run_on_batch(self, data, optimizer, epoch=None):
        ret = self(data, direct = 'forward')

        if optimizer is not None:
            optimizer.zero_grad()
            ret['loss'].backward()
            optimizer.step()

        return ret
