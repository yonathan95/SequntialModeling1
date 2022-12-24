from torch import nn

class RNN(nn.Module):

    def __init__(self, input_size=1, hidden_size=64, output_size=1, num_layers=3, batch_first=True):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=batch_first)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x, h=None):
        """
        :param x: input of shape (batch_size, seq_len, 2)
        :param h: hidden state of shape (num_layers, batch_size, hidden_size)
        :return: output of shape (batch_size, seq_len, 2)
        """
        x, h = self.lstm(x)
        x = self.linear(x)
        return x, h

