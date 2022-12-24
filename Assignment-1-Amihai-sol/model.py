import torch
from torch import nn


class MyModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=2, hidden_size=64, num_layers=1, batch_first=True)
        self.linear = nn.Linear(64, 2)

    def forward(self, x, h=None):
        """
        :param x: input of shape (batch_size, seq_len, 2)
        :param h: hidden state of shape (num_layers, batch_size, hidden_size)
        :return: output of shape (batch_size, seq_len, 2)
        """
        x, h = self.lstm(x, h)
        x = self.linear(x)
        return x, h

    def inference(self, x, steps=20):
        """
        :param x: initial condition of shape (batch_size, 1, 2)
        :param steps: the horizon length of the prediction
        :return: predicted trajectory of shape (batch_size, steps, 2)
        """
        self.eval()
        ys = torch.zeros(x.shape[0], steps + 1, 2)
        ys[:, :1, :] = x
        with torch.no_grad():
            for i in range(1, steps + 1):
                x, h = self.forward(ys[:, :i])
                ys[:, 1:i + 1, :] = x
        return ys[:, 1:]
