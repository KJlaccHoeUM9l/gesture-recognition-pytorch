'''
	Network Architectures

	Siqin Li
	April, 2018
'''

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms, utils
from torch.autograd import Variable


class FineTuneLstmModel(nn.Module):
	def __init__(self, original_model, arch, num_classes, lstm_layers, hidden_size, fc_size):
		super(FineTuneLstmModel, self).__init__()
		self.hidden_size = hidden_size
		self.num_classes = num_classes
		self.fc_size = fc_size

		# select a base model
		if arch.startswith('alexnet'):
			self.features = original_model.features
			self.fc_pre = nn.Sequential(nn.Linear(256 * 6 * 6, fc_size),
						nn.Dropout())

			self.rnn = nn.LSTM(input_size = fc_size,
						hidden_size = hidden_size,
						num_layers = lstm_layers,
						batch_first = True)
			self.fc = nn.Linear(hidden_size, num_classes)
			self.modelName = 'alexnet_lstm'

		elif arch.startswith('vgg16'):
			self.features = original_model.features
			self.rnn = nn.LSTM(input_size = 25088,
						hidden_size = hidden_size,
						num_layers = lstm_layers,
						batch_first = True)
			self.fc = nn.Linear(hidden_size, num_classes)
			self.modelName = 'vgg16_lstm'

		elif arch.startswith('resnet'):
			# except the last linear layer
			self.features = nn.Sequential(*list(original_model.children())[:-1])
			self.rnn = nn.LSTM(input_size = 512,
						hidden_size = hidden_size,
						num_layers = lstm_layers,
						batch_first = True)
			self.fc = nn.Linear(hidden_size, num_classes)
			self.modelName = 'resnet_lstm'

		else:
			raise Exception("This architecture has not been supported yet")

	def init_hidden(self, num_layers, batch_size):
		return (Variable(torch.zeros(num_layers, batch_size, self.hidden_size)).cuda(),
				Variable(torch.zeros(num_layers, batch_size, self.hidden_size)).cuda())

	def forward(self, inputs, hidden=None, steps=0):
		'''
		inputs: sequence of images 
		'''
		length = len(inputs)
		fs = Variable(torch.zeros(length, self.rnn.input_size)).cuda()
		for i in range(length):
			# print self.features(inputs[i].unsqueeze(0)).shape
			f = self.features(inputs[i].unsqueeze(0))
			f = f.view(f.size(0), -1)
			f = self.fc_pre(f)
			fs[i] = f
		fs = fs.unsqueeze(0)

		outputs, hidden = self.rnn(fs, hidden)
		outputs = self.fc(outputs[0])
		return outputs, hidden


class flowNet(nn.Module):

	def __init__(self, num_classes, lstm_layers, hidden_size, fc_size):
		super(flowNet, self).__init__()
		self.features = nn.Sequential(
			nn.Conv2d(2, 64, kernel_size=11, stride=4, padding=2),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=3, stride=2),
			nn.Conv2d(64, 192, kernel_size=5, padding=2),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=3, stride=2),
			nn.Conv2d(192, 384, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(384, 256, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(256, 256, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=3, stride=2),
		)
		self.fc_pre = nn.Sequential(nn.Linear(256 * 6 * 6, fc_size),
						nn.Dropout())
		self.rnn = nn.LSTM(input_size = fc_size,#256 * 6 * 6,
					hidden_size = hidden_size,
					num_layers = lstm_layers,
					batch_first = True)
		self.fc = nn.Linear(hidden_size, num_classes)
		self.modelName = 'alexnet_lstm'

	def forward(self, inputs, hidden=None, steps=0):
		'''
		inputs: sequence of images 
		'''
		length = len(inputs)
		fs = Variable(torch.zeros(length, self.rnn.input_size)).cuda()
		for i in range(length):
			# print self.features(inputs[i].unsqueeze(0)).shape
			f = self.features(inputs[i].unsqueeze(0))
			f = f.view(f.size(0), -1)
			f = self.fc_pre(f)
			fs[i] = f
		fs = fs.unsqueeze(0)

		outputs, hidden = self.rnn(fs, hidden)
		length_last = min(length // 4, 3)
		output = self.fc(outputs[0][-length_last:])
		outputs = self.fc(outputs[0])
		return output, hidden, outputs
