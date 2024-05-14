import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size  # 上个时刻隐藏层的状态

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)  # Uxi
        self.i2o = nn.Linear(input_size + hidden_size, output_size)  # Vst
        self.softmax = nn.LogSoftmax()

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)  # Vst
        output = self.softmax(output)
        return output, hidden  # 返回当前时刻输出和当前时刻隐藏层状态

    def initHidden(self):
        return Variable(torch.zeros(1, self.hidden_size))





