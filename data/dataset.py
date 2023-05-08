import torch
from PIL import Image
import numpy as np

class DataSet(torch.utils.data.Dataset):
    def __init__(self, paths, labels, one_hot_encodings):
        self.paths = paths
        self.labels = labels
        self.one_hot_encodings = one_hot_encodings

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]

        data = np.array(Image.open("dataset/" + path).convert("L"))
        y = self.one_hot(self.labels[path])

        return data, y
    
    def one_hot(self, label):
        v = np.zeros(len(self.one_hot_encodings))
        v[self.one_hot_encodings[label]] = 1
        return v
