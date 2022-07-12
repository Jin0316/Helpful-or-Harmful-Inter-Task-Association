from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
import numpy as np 
import torch
from PIL import Image

class split_cifar10(CIFAR10):
    def __init__(self, root, classes = range(10), train = True, transform = None, download = False):
        super(split_cifar10, self).__init__(root, train = train, transform = transform, download = download)

        self.tensorTransform = transforms.ToTensor()
        self.train = train 
        self.transform = transform 
        
        self.train_data, self.train_target = [], []
        self.test_data, self.test_target = [], []

        if self.train: 
            for i in range(len(self.data)): 
                if self.targets[i] in classes: 
                    self.train_data.append(self.data[i])
                    self.train_target.append(self.targets[i] - classes[0])
        else: 
            for i in range(len(self.data)):
                if self.targets[i] in classes: 
                    self.test_data.append(self.data[i])
                    self.test_target.append(self.targets[i] - classes[0])

    def __getitem__(self, index): 
        if self.train: 
            img, target = Image.fromarray(self.train_data[index]), self.train_target[index]
            if self.transform: 
                img = self.transform(img)
        
        else: 
            img, target = Image.fromarray(self.test_data[index]), self.test_target[index]
            if self.transform: 
                img = self.transform(img)
        return index, img, target 

    def __len__(self):
        if self.train: 
            return len(self.train_data)
        else: 
            return len(self.test_data)

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ]),
    'test': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ]),
}

def get_cifar10_data(classes):
    train_data = split_cifar10(root = './data', 
                                classes = classes, 
                                train = True, 
                                transform = data_transforms['train'], 
                                download = True)

    test_data = split_cifar10(root = './data', 
                                classes = classes, 
                                train = False, 
                                transform = data_transforms['test'], 
                                download = True)
    return train_data, test_data
