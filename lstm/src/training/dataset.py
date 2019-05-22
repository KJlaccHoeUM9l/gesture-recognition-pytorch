import os
import torch
import numpy as np
import PIL.Image as Image

from torch.utils.data import Dataset


class CLMarshallingDataset(Dataset):
	def __init__(self, root_dir, transform=None):
		'''
		structure of root_dir: 'root_dir/class_i/video_i/img_i.jpg'
		'''
		#print('\t\tCLMarshallingDataset.__init__()')
		self.root_dir = root_dir
		self.transform = transform
		self.classes = sorted(os.listdir(self.root_dir))
		self.count = [len(os.listdir(self.root_dir + '/' + c)) for c in self.classes]
		self.acc_count = [self.count[0]]
		for i in range(1, len(self.count)):
				self.acc_count.append(self.acc_count[i-1] + self.count[i])


	def __len__(self):
		#print('\t\tCLMarshallingDataset.__len__()')
		return np.sum(np.array([len(os.listdir(self.root_dir + '/' + c)) for c in self.classes]))


	def __getitem__(self, idx):
		#print('\t\tCLMarshallingDataset.__getitem__()')
		for i in range(len(self.acc_count)):
			if idx < self.acc_count[i]:
				label = i
				break

		class_path = self.root_dir + '/' + self.classes[label]
		if label:
			video_path = class_path + '/' + sorted(os.listdir(class_path))[idx-self.acc_count[label]]
		else:
			video_path = class_path + '/' + sorted(os.listdir(class_path))[idx]

		frames = []
		if self.transform is not None:
			frame_list = sorted(os.listdir(video_path))
			for i, frame in enumerate(frame_list):
				frame = Image.open(video_path + '/' + frame)	# Read in BGR
				#frame = frame.convert('RGB')
				frame = self.transform[0](frame)
				frames.append(frame)

		frames = torch.stack(frames)
		frames = frames[: -1] - frames[1:]

		return frames, label
