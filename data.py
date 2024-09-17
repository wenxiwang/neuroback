import os
import torch
import gzip
import json
from torch_geometric.data import Data, Dataset
import pickle


class MyOwnDataset(Dataset):
	def __init__(self, root, transform=None, pre_transform=None):
		super(MyOwnDataset, self).__init__(root, transform, pre_transform)

	@property
	def raw_file_names(self):
		return []

	@property
	def processed_file_names(self):
		return sorted(list(os.listdir(self.processed_dir)), key=lambda fn: os.path.getsize(self.processed_dir + "/" + fn))

	def download(self):
		pass

	def process(self):
		pass

	def len(self):
		return len(self.processed_file_names)

	def get(self, idx):
		data = torch.load(os.path.join(self.processed_dir, self.processed_file_names[idx]))

		reverse = data.edge_index.index_select(0, torch.LongTensor([1, 0]))
		data.edge_index = torch.cat([data.edge_index, reverse], dim=1)
		data.edge_attr = torch.cat([data.edge_attr, data.edge_attr], dim=0)

		data.x = data.x.float()
		data.edge_index = data.edge_index.long()
		data.edge_attr = data.edge_attr.float()

		if data.y != None:
			data.y = data.y.long()

		return data