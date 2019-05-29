import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models
import math


class LSTM_ResNet18(nn.Module):
    def __init__(self,
                 sample_size,
                 sample_duration,
                 num_classes=13,
                 fc_size=1024,
                 num_lstm_layers=1,
                 hidden_size=512):
        super(LSTM_ResNet18, self).__init__()
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.fc_size = fc_size

        # except the last linear layer
        self.features = nn.Sequential(*list(models.__dict__['resnet18'](pretrained=False).children())[:-1])
        for i, param in enumerate(self.features.parameters()):
            param.requires_grad = False

        # last_duration = int(math.floor(sample_duration / 16))
        # last_size = int(math.floor(sample_size / 32))
        # fc_size = 512 * last_duration * last_size * last_size
        # self.fc_size = fc_size
        self.fc_pre = nn.Sequential(nn.Linear(512, fc_size), nn.Dropout())

        self.rnn = nn.LSTM(input_size=fc_size,
                           hidden_size=hidden_size,
                           num_layers=num_lstm_layers,
                           batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, inputs, hidden=None):
        inputs = inputs[0][0]
        print(inputs)
        length = len(inputs)
        print('Len: ' + str(length))
        fs = torch.zeros(length, self.rnn.input_size)
        for frame in inputs:
            f = self.features(frame)
        # fs = Variable(torch.zeros(length, self.rnn.input_size))
        # #fs = torch.zeros(length, self.rnn.input_size)
        # for i in range(length):
        #     f = self.features(inputs[i].unsqueeze(0))
        #     #f = self.features(inputs[i])
        #     f = f.view(f.size(0), -1)
        #     f = self.fc_pre(f)
        #     fs[i] = f
        # fs = fs.unsqueeze(0)

        out = self.rnn(fs)
        out = self.fc(out)

        return out

def lstm_resnet18(**kwargs):
    model = LSTM_ResNet18(**kwargs)
    return model
