from scipy.io import loadmat
import torch
import numpy as np
import os
import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler


class JSBChorales(Dataset):

    def __init__(self):
        data = loadmat('/home/yonathan95/SequntialModeling1/Assignment2/JSB_Chorales.mat')

        X_train = data['traindata'][0]
        X_valid = data['validdata'][0]
        X_test = data['testdata'][0]

        for data in [X_train, X_valid, X_test]:
            for i in range(len(data)):
                data[i] = torch.Tensor(data[i].astype(np.float64))

        self.data = X_train
        
    def __getitem__(self, index):
        """
        :param index: index of the sample
        :return: x, y of shape (seq_len, 2)
        """
        return self.data[index, :-1, :], self.data[index, 1:, :]

    def __len__(self):
        return len(self.data)

def load_jsb(as_tensor=False):
    data = loadmat('/home/yonathan95/SequntialModeling1/Assignment2/JSB_Chorales.mat')

    X_train = data['traindata'][0]
    X_valid = data['validdata'][0]
    X_test = data['testdata'][0]

    if as_tensor:
        for data in [X_train, X_valid, X_test]:
            for i in range(len(data)):
                data[i] = torch.Tensor(data[i].astype(np.float64))

    return X_train, X_valid, X_test

