import torch
import torch.nn as nn
from torch.nn import init
from decayer import Decayer

class TLSTM(nn.Module):
	def __init__(self,input_size, hidden_size,  bias = True):
		super(TLSTM,self).__init__()
		self.i2h = nn.Linear(input_size, 4*hidden_size, bias)
		self.h2h = nn.Linear(hidden_size, 4*hidden_size, bias)
		self.c2s = nn.Sequential(nn.Linear(hidden_size, hidden_size, bias), nn.Tanh())
		self.sigmoid = nn.Sigmoid()
		self.tanh = nn.Tanh()

	def forward(self,input, cell, hidden, transed_delta_t):

		cell_short = self.c2s(cell)
		cell_new = cell - cell_short + cell_short* transed_delta_t 
		gates = self.i2h(input) + self.h2h(hidden)
		ingate, forgate, cellgate, outgate = gates.chunk(4,1)
		ingate = self.sigmoid(ingate)
		forgate = self.sigmoid(forgate)
		cellgate = self.tanh(cellgate)
		outgate = self.sigmoid(outgate)

		cell_output = forgate*cell_new + ingate*cellgate
		hidden_output = outgate*self.tanh(cell_output) 
		return cell_output, hidden_output



