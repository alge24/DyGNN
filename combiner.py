import torch
import torch.nn as nn
from torch.nn import init

class Combiner(nn.Module):
	def __init__(self, input_size, output_size,act, bias = True ):
		super(Combiner,self).__init__()
		self.h2o = nn.Linear(input_size,output_size,bias)
		self.l2o = nn.Linear(input_size,output_size,bias)
		if act == 'tanh':
			self.act = nn.Tanh()
		elif act == 'sigmoid':
			self.act = nn.Sigmoid()
		else:
			self.act = nn.ReLU() 

	def forward(self, head_info, tail_info):
		node_output = self.h2o(head_info) + self.l2o(tail_info)
		node_output_tanh = self.act(node_output)
		return node_output_tanh
