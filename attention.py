import torch
import torch.nn as nn
from torch.nn import init
from combiner import Combiner
from edge_updater import Edge_updater_nn
from node_updater import TLSTM
from scipy.sparse import lil_matrix, find
import numpy as np
import random

class Attention(nn.Module):
    def __init__(self, embedding_dims):
        super(Attention,self).__init__()
        self.bilinear = nn.Bilinear(embedding_dims,embedding_dims,1)
        self.softmax = nn.Softmax(0)

    def forward(self,node1, node2):
    	return self.softmax( self.bilinear(node1, node2).view(-1,1) )

