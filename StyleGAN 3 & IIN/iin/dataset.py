import torch
import pandas as pd
import numpy as np


class TensorDataset(torch.utils.data.Dataset):
    def __init__(self, source_path):
        self.tensor = torch.load(source_path)
    
    def __getitem__(self, index):
        return self.tensor[index]
    
    def __len__(self):
        return self.tensor.size(0)


class TensorPairedDataset(torch.utils.data.Dataset):
    def __init__(self, source_path, attribute_path):
        self.tensor = torch.load(source_path)
        
        self.attributes = pd.read_csv(attribute_path)
        self.attr_index_dict = {self.attributes.columns[i]: i for i in [0, 1, 4]}
        self.attribute_names = list(self.attr_index_dict.keys())
        self.positive_attr_dict = {index:np.where(self.attributes[index] > 0)[0] for index in self.attribute_names}
        self.negative_attr_dict = {index:np.where(self.attributes[index] < 0)[0] for index in self.attribute_names}
        
        self.n_factors = len(self.attribute_names)
    
    def __getitem__(self, index):
        factor = np.random.choice(self.attribute_names)
        z1 = [self.tensor[index] for i in range(16)]
        attr = self.attributes[factor][index]
        if attr > 0:
            index2 = np.random.choice(self.positive_attr_dict[factor])
        else:
            index2 = np.random.choice(self.negative_attr_dict[factor])
        z2 = [self.tensor[index2] for i in range(16)]
        
        return self.attr_index_dict[factor], torch.cat(z1), torch.cat(z2)
    
    def __len__(self):
        return self.tensor.size(0)

