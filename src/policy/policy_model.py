import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from scipy.stats import entropy
import numpy as np

class PolicyModel(nn.Module):
    def __init__(self, num_layer, num_ops, num_features, lstm_size=64, lstm_num_layers=2, tanh_constant=2.5, temperature=5):
        super(PolicyModel, self).__init__()
        self.num_layer  = num_layer
        self.num_ops   = num_ops 
        self.num_features   = num_features 
        self.lstm_size = lstm_size
        self.lstm_N    = lstm_num_layers
        self.tanh_constant = tanh_constant
        self.temperature   = temperature
        self.op_probs = []

        #create lstm model
        self.register_parameter('input_vars', nn.Parameter(torch.Tensor(1, 1, lstm_size)))
        self.w_lstm = nn.LSTM(input_size=self.lstm_size, hidden_size=self.lstm_size, num_layers=self.lstm_N)
        self.w_embd_feature = nn.Embedding(self.num_features, self.lstm_size)
        self.w_embd_op = nn.Embedding(self.num_ops, self.lstm_size)
        self.w_predA = nn.Linear(self.lstm_size, self.num_features)
        self.w_predO = nn.Linear(self.lstm_size, self.num_ops)
        self.w_predB = nn.Linear(self.lstm_size, self.num_features)

        nn.init.uniform_(self.input_vars         , -0.1, 0.1)
        nn.init.uniform_(self.w_lstm.weight_hh_l0, -0.1, 0.1)
        nn.init.uniform_(self.w_lstm.weight_ih_l0, -0.1, 0.1)
        nn.init.uniform_(self.w_embd_feature.weight      , -0.1, 0.1)
        nn.init.uniform_(self.w_embd_op.weight      , -0.1, 0.1)
        nn.init.uniform_(self.w_predA.weight      , -0.1, 0.1)
        nn.init.uniform_(self.w_predO.weight      , -0.1, 0.1)
        nn.init.uniform_(self.w_predB.weight      , -0.1, 0.1)

    def forward(self, actions=None):
        inputs, h0 = self.input_vars, None
        log_probs, entropys, op_probs, sampled_arch = [], [], [], []

        for i_layer in range(self.num_layer):
            outputs, h0 = self.w_lstm(inputs, h0)
      
            A_feature = self.w_predA(outputs)
            op_feature = self.w_predO(outputs)
            B_feature = self.w_predB(outputs)

            A_feature = A_feature / self.temperature
            A_feature = self.tanh_constant * torch.tanh(A_feature)

            B_feature = B_feature / self.temperature
            B_feature = self.tanh_constant * torch.tanh(B_feature)

            op_feature = op_feature / self.temperature
            op_feature = self.tanh_constant * torch.tanh(op_feature)

            # distribution
            A_distribution = Categorical(logits=A_feature)
            op_distribution = Categorical(logits=op_feature)
            B_distribution = Categorical(logits=B_feature)

            A_idx    = A_distribution.sample()
            op_idx    = op_distribution.sample()
            B_idx    = B_distribution.sample()
            sampled_arch.append((A_idx.item(), op_idx.item(), B_idx.item()) )

            A_log_prob = A_distribution.log_prob(A_idx)
            op_log_prob = op_distribution.log_prob(op_idx)
            B_log_prob = B_distribution.log_prob(B_idx)
            log_probs.append(A_log_prob.view(-1))
            log_probs.append(op_log_prob.view(-1))
            log_probs.append(B_log_prob.view(-1))

            op_entropy  = A_distribution.entropy()
            #op_entropy  = A_distribution.entropy()
            #op_entropy  = A_distribution.entropy()
            #op_probs.append(op_distribution.probs)
            entropys.append(op_entropy.view(-1))
      
            # obtain the input embedding for the next step
            inputs = self.w_embd_feature(A_idx) + self.w_embd_feature(B_idx) + self.w_embd_op(op_idx)

        return torch.sum(torch.cat(log_probs)), torch.sum(torch.cat(entropys)), sampled_arch
        #return torch.sum(torch.cat(log_probs)), ent, sampled_arch
  
