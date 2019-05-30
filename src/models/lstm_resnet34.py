import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models


class LSTM_ResNet34(nn.Module):
    def __init__(self,
                 sample_size,
                 sample_duration,
                 num_classes=13,
                 fc_size=1024,
                 num_lstm_layers=1,
                 hidden_size=512):
        super(LSTM_ResNet34, self).__init__()
        self.sample_duration = sample_duration
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.fc_size = fc_size

        # except the last linear layer
        self.features = nn.Sequential(*list(models.__dict__['resnet34'](pretrained=False).children())[:-1])
        for i, param in enumerate(self.features.parameters()):
            param.requires_grad = False

        self.fc_pre = nn.Sequential(nn.Linear(512, fc_size), nn.Dropout())

        self.rnn = nn.LSTM(input_size=fc_size,
                           hidden_size=hidden_size,
                           num_layers=num_lstm_layers,
                           batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, batch):
        clip_quantities = len(batch)
        out = Variable(torch.zeros(clip_quantities, self.num_classes)).cuda()

        num_clip = 0
        for clip in batch:
            rnn_clip_input = torch.zeros(self.sample_duration, self.rnn.input_size)

            # CNN part
            num_frame = 0
            new_clip = clip.transpose(0, 1)
            for frame in new_clip:
                cnn_frame_out = self.features(frame.unsqueeze(0))
                cnn_frame_out = cnn_frame_out.reshape(1, -1)
                cnn_frame_out = self.fc_pre(cnn_frame_out)

                rnn_clip_input[num_frame] = cnn_frame_out
                num_frame += 1

            # RNN part
            rnn_clip_input = rnn_clip_input.unsqueeze(0).cuda()
            rnn_clip_out, _ = self.rnn(rnn_clip_input)
            rnn_clip_out = self.fc(rnn_clip_out)

            # Average for all frames in clip
            rnn_clip_out = torch.mean(rnn_clip_out[0], dim=0).unsqueeze(0)

            out[num_clip] = rnn_clip_out
            num_clip += 1

        return out


def lstm_resnet34(**kwargs):
    model = LSTM_ResNet34(**kwargs)
    return model
