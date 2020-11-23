import torch
import torch.nn as nn
from torch.nn import init
from torch.utils.data import Dataset
import numpy as np

class Temporal_Dataset(Dataset):
	def __init__(self, file_name,starting = 0,skip_rows=0, div =3600):
		self.data = np.loadtxt(fname=file_name, skiprows=skip_rows)[:,[0,1,3]]
		self.time = self.data[:,2]
		self.trans_time = (self.time - self.time[0])/div
		self.data[:,2] = self.trans_time
		self.data[:,[0,1]] = self.data[:,[0,1]] - starting

	def __len__(self):
		return self.time.shape[0]

	def __getitem__(self,idx):
		sample = self.data[idx,:]
		return sample


