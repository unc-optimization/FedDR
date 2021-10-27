import torch
import torchvision
import torchvision.transforms as transforms

from random import Random

from asyncfeddr.utils.file_io import *

class Dataset(torch.utils.data.Dataset):
	'Characterizes a dataset for PyTorch'
	def __init__(self, data, labels):
		'Initialization'
		self.labels = torch.tensor(labels)
		self.data = torch.tensor(data)

	def __len__(self):
		'Denotes the total number of samples'
		return len(self.data)

	def __getitem__(self, index):
		'Generates one sample of data'
		# Get data and label
		X = self.data[index]
		y = self.labels[index]

		return X, y

class Partition(object):
	""" Dataset-like object, but only access a subset of it. """

	def __init__(self, data, index):
		self.data = data
		self.index = index

	def __len__(self):
		return len(self.index)

	def __getitem__(self, index):
		data_idx = self.index[index]
		return self.data[data_idx]


class DataPartitioner(object):
	""" Partitions a dataset into different chuncks. """

	def __init__(self, data, sizes=[0.7, 0.2, 0.1], seed=1234):
		self.data = data
		self.partitions = []
		rng = Random()
		rng.seed(seed)
		data_len = len(data)
		indexes = [x for x in range(0, data_len)]
		rng.shuffle(indexes)

		for frac in sizes:
			part_len = int(frac * data_len)
			self.partitions.append(indexes[0:part_len])
			indexes = indexes[part_len:]

	def use(self, partition):
		return Partition(self.data, self.partitions[partition])

def partition_dataset(args):
	""" Partitioning CIFAR10 """

	if args.dataset == 'MNIST':
		transform = transforms.Compose([
					transforms.ToTensor(),
					transforms.Normalize((0.1307, ), (0.3081, ))
				])

		trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
		testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

	elif args.dataset == 'FEMNIST':
		data_path = './data'
		dataset = 'FEMNIST'
		train_path = os.path.join(data_path, dataset, 'train')
		test_path = os.path.join(data_path, dataset, 'test')

		dataset = read_data(train_path, test_path)

		users, groups, train_data, test_data = dataset
	else:
		raise ValueError('Unsupported dataset')

	if args.rank > 0:
		if args.dataset != 'FEMNIST':
			# total number of workers
			size = args.world_size-1

			partition_sizes = [1.0 / size for _ in range(size)]

			train_partition = DataPartitioner(trainset, partition_sizes)
			train_partition = train_partition.use(args.rank-1)
			train_loader = torch.utils.data.DataLoader(
				train_partition, batch_size=int(args.batch_size), shuffle=True)

			test_partition = DataPartitioner(testset, partition_sizes)
			test_partition = test_partition.use(args.rank-1)
			test_loader = torch.utils.data.DataLoader(
				test_partition, batch_size=int(args.test_batch_size), shuffle=False)
		else:
			key_list = list(train_data.keys())
			key = key_list[args.rank-1]
			train_data, train_label = train_data[key]['x'], train_data[key]['y']
			test_data, test_label = test_data[key]['x'], test_data[key]['y']

			trainset = Dataset(train_data,train_label)
			testset = Dataset(test_data,test_label)

			train_loader = torch.utils.data.DataLoader(
				trainset, batch_size=int(args.batch_size), shuffle=True)

			test_loader = torch.utils.data.DataLoader(
				testset, batch_size=int(len(test_label)), shuffle=False)

		return train_loader, test_loader

	else:
		train_loader_list, test_loader_list = [], []
		if args.dataset != 'FEMNIST':

			# total number of workers
			size = args.world_size-1

			# partition size
			partition_sizes = [1.0 / size for _ in range(size)]

			train_partition = DataPartitioner(trainset, partition_sizes)
			test_partition = DataPartitioner(testset, partition_sizes)

			for rank in range(args.world_size - 1):
				train_data = train_partition.use(rank)
				train_loader = torch.utils.data.DataLoader(
					train_data, batch_size=int(args.batch_size), shuffle=True)

				
				test_data = test_partition.use(rank)
				test_loader = torch.utils.data.DataLoader(
					test_data, batch_size=int(args.test_batch_size), shuffle=False)

				train_loader_list.append(train_loader)
				test_loader_list.append(test_loader)
		else:
			key_list = list(train_data.keys())
			for rank in range(args.world_size-1):
				key = key_list[rank]

				data, label = train_data[key]['x'], train_data[key]['y']
				trainset = Dataset(data,label)
				train_loader = torch.utils.data.DataLoader(
					trainset, batch_size=int(args.batch_size), shuffle=True)

				data, label = test_data[key]['x'], test_data[key]['y']	
				testset = Dataset(data,label)
				test_loader = torch.utils.data.DataLoader(
					testset, batch_size=int(len(label)), shuffle=False)

				train_loader_list.append(train_loader)
				test_loader_list.append(test_loader)

		return train_loader_list, test_loader_list