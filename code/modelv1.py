import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.autograd import Variable


import numpy as np

###############################################################################
###  Inputs in this model are different than those in other model versions  ###
###  other models are adapted to pytorch embedding function, while this     ###
###  model expects input in the form of a list of tensors, which can be     ###
###  found in the embedding matrix of the other models, but adding a        ###
###  processing step within the training writeup to get all of the tensors  ###
###  into list form.                                                        ###
###############################################################################

class BasicLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Note: since b_ii and b_hi are the same dimension, they can 
        # simply be written as b_i
        self.W_ii = Parameter(torch.Tensor(input_dim, hidden_dim))
        self.W_hi = Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.b_i = Parameter(torch.Tensor(hidden_dim))

        self.W_if = Parameter(torch.Tensor(input_dim, hidden_dim))
        self.W_hf = Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.b_f = Parameter(torch.Tensor(hidden_dim))

        self.W_io = Parameter(torch.Tensor(input_dim, hidden_dim))
        self.W_ho = Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.b_o = Parameter(torch.Tensor(hidden_dim))

        self.W_ig = Parameter(torch.Tensor(input_dim, hidden_dim))
        self.W_hg = Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.b_g = Parameter(torch.Tensor(hidden_dim))

        self.init_random_weights()

    def init_random_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                nn.init.zeros_(p.data)

    def initHidden(self):
        rand_hid = Variable(torch.randn(self.num_layers, self.batch_size, self.hidden_size))
        rand_cell = Variable(torch.randn(self.num_layers, self.batch_size, self.hidden_size))
        return rand_hid, rand_cell


    def forward(self, input, input_states):
        """Assumes input is a list of tensors"""
        hidden_seq = []
        if input_states is None:
            h_t, c_t = self.initHidden()
        else:
            h_t, c_t = input_states
        for t in range(input): # iterate over the time steps
            x_t = input[t]
            i_t = torch.sigmoid(x_t @ self.W_ii + h_t @ self.W_hi + self.b_i)
            f_t = torch.sigmoid(x_t @ self.W_if + h_t @ self.W_hf + self.b_f)
            g_t = torch.tanh(x_t @ self.W_ig + h_t @ self.W_hg + self.b_g)
            o_t = torch.sigmoid(x_t @ self.W_io + h_t @ self.W_ho + self.b_o)
            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * torch.tanh(c_t)
            hidden_seq.append(h_t)
        hidden_seq = torch.cat(hidden_seq, dim=0)
        # reshape from shape (sequence, batch, feature) to (batch, sequence, feature)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()
        return hidden_seq, (h_t, c_t)