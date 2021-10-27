# -*- coding: utf-8 -*-
"""
Created on Mon May 24 12:33:46 2021

@author: Woojin
"""
import torch
import torch.nn as nn
from torch.autograd import Variable

class CharRNN(nn.Module):
    def __init__(self, input_size, hidden_size=30, n_layer=1):
        super(CharRNN, self).__init__()
        
        # RNN 설정
        # 1. input size : 등장인물 숫자
        # 2. hidden size : hyperparameter --> 128
        # 3. bias
        # 4. n_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layer = n_layer
        self.batch=100
        
        self.rnn = nn.RNN(input_size=self.input_size, hidden_size=self.hidden_size, bias=True, num_layers=self.n_layer, dropout=0.5, batch_first=True)
        self.linear = nn.Linear(self.hidden_size, self.input_size)

           
    def forward(self, input, hidden):

        x, hidden = self.rnn(input, hidden)
        x = x.reshape(x.size(0) * x.size(1), x.size(2))
        output = self.linear(x)
            
        return output, hidden
    
    
    def init_hidden(self, batch_size):
        # input weight parameter W_x : (hidden_size, input_size)
        # hidden weight parameter W_h : (hidden_size, hidden_size)
        # bias input-hidden : (hidden)
        # bias hidden-hidden : (hidden)
        
        hidden_state = Variable(torch.zeros(self.n_layer, batch_size, self.hidden_size))
        
        return hidden_state



class CharLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, n_layer):
        super(CharLSTM, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layer = n_layer
        self.batch = 100
        
        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, bias=True, num_layers=self.n_layer, dropout=0.5, batch_first=True)
        self.linear = nn.Linear(self.hidden_size, self.input_size)


    def forward(self, input, hidden):

        x, (hidden_state, cell_state) = self.lstm(input, hidden)
        x = x.reshape(x.size(0)*x.size(1), x.size(2))
        output = self.linear(x)
            
        return output, (hidden_state, cell_state)


    def init_hidden(self, batch_size):
        # input weight parameter W_x : (hidden_size, input_size)
        # hidden weight parameter W_h : (hidden_size, hidden_size)
        # bias input-hidden : (hidden)
        # bias hidden-hidden : (hidden)
        
        # LSTM에서는 hidden state와 cell state가 따로 필요함
        weight = next(self.parameters()).data
        hidden_state = weight.new(self.n_layer, batch_size, self.hidden_size).zero_()
        cell_state = weight.new(self.n_layer, batch_size, self.hidden_size).zero_()
        
        return (hidden_state, cell_state)


