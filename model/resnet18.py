import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data.dataloader import default_collate
from torch.utils.data import DataLoader

import copy
import sys
import os 
  
current_path = os.getcwd()
sys.path.insert(0, str(current_path[:-3]))
import config.CONFIG as CONFIG

device = CONFIG.DEVICE

global fisher_n 
fisher_n = 30

def get_data_loader(dataset, batch_size, collate_fn = None, drop_last = False, augment = True):
    """
    Return <Dataloader> object for the provided <Dataset> object
    """
    if augment:
        dataset_ = copy.deepcopy(dataset)
        dataset_.transform  = transforms.Compose([transforms.ToTensor()])
    else:
        dataset_ = dataset
    
    return DataLoader(
        dataset_, batch_size=batch_size, shuffle = True, 
        collate_fn=(collate_fn or default_collate), drop_last = drop_last, 
        num_workers=0
        )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=2):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes, bias = True)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
    
    def feature_extraction(self, x): 
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        return out

def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])


def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])


class ModifiedResNet18(nn.Module):
    def __init__(self, previous_mask, current_task, make_model=True):
        super(ModifiedResNet18, self).__init__()
        self.first_fisher = 1
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.previous_mask = previous_mask
        self.current_task = current_task
        self.start = False

        if make_model:
            self.make_model()

    def train_nobn(self, mode=True):
        """Override the default module train."""
        super(ModifiedResNet18, self).train(mode)
        for module in self.shared.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()

    def make_model(self):
        """Creates the model."""
        resnet = ResNet18()
        self.datasets, self.classifiers = [], nn.ModuleList()

        self.shared = nn.Sequential()
        for name, module in resnet.named_children():
            if name != 'linear' and name != 'avg_pool':
                self.shared.add_module(name, module)

        self.feature_extraction = nn.Sequential()
        for name, module in resnet.named_children():
            if name == 'layer4':
                break
            else:
                self.feature_extraction.add_module(name, module)

        self.batchnorms = nn.ModuleList()
        self.classifier = None
        self.batchnorm = None

    def SELECTION_ON(self):
        if self.current_task != 1: 
            self.probability = torch.Tensor([1.0 for i in range(1, self.current_task)])
            self.gradient = []
        else: 
            self.probability = None 

    def set_dataset(self, dataset):
        """Change the active classifier."""
        self.classifier = self.classifiers[self.datasets.index(dataset)]
        self.batchnorm = self.batchnorms[self.datasets.index(dataset)]

        for name, module in enumerate(self.shared.modules()): 
            if isinstance(module, nn.BatchNorm2d):
                #print(name)
                module.weight.data.copy_(self.batchnorm[str(name)].weight.data)
                module.bias.data.copy_(self.batchnorm[str(name)].bias.data)
                module.running_var = self.batchnorm[str(name)].running_var
                module.running_mean = self.batchnorm[str(name)].running_mean


    def add_dataset(self, dataset, num_outputs):
        """Adds a new dataset to the classifier."""
        if dataset not in self.datasets:
            self.datasets.append(dataset)
            self.classifiers.append(nn.Linear(512, num_outputs, bias = True))
        
        initial_bn = nn.ModuleDict()
        for name, module in enumerate(self.shared.modules()):
            if isinstance(module, nn.BatchNorm2d):
                new_bn = copy.deepcopy(module)
                new_bn.weight.data.fill_(1)
                new_bn.bias.data.zero_()
                initial_bn[str(name)] = new_bn
        self.batchnorms.append(initial_bn)

    def save_bn(self):
        for name, module in enumerate(self.shared.modules()):
            if isinstance(module, nn.BatchNorm2d):
                self.batchnorm[str(name)].weight.data.copy_(module.weight.data)
                self.batchnorm[str(name)].bias.data.copy_(module.bias.data)
                self.batchnorm[str(name)].running_var = module.running_var
                self.batchnorm[str(name)].running_mean = module.running_mean

    def start_finding(self, start = False):
        if start == True: 
            self.start = True
        else: 
            self.start = False
    
    def forward(self, x): 
        output = self.shared(x)
        output = self.avgpool(output)
        output = output.view(output.size(0), -1)
        output = self.classifier(output)
        return output

    def update_sampled(self):
        gradient_temp = {}
        for name, parameters in self.shared.named_parameters():
            name = name.replace('.', '__')
            if ('conv' in name) or ('0__weight' in name):
                if parameters.requires_grad:
                    parameters_ = parameters.detach().clone()
                    parameter_grad = parameters.grad.detach().clone()
                    gradient = torch.mul(parameters_, parameter_grad)
                    gradient_temp[name] = gradient
        
        probability = self.probability.detach().clone()
        for prev_task in range(1, self.current_task):    
            total_sum = 0
            for name, parameters in self.shared.named_parameters():
                name = name.replace('.', '__')
                if ('conv' in name) or ('0__weight' in name):
                    mask = self.previous_mask[name]
                    gradient = gradient_temp[name]
                    total_sum += gradient[mask.eq(prev_task)].sum()
            gradient = total_sum
            self.gradient.append(gradient)
        for prev_task in range(1, self.current_task):
            probability[prev_task-1] = probability[prev_task-1] - CONFIG.alpha_lr * self.gradient[prev_task-1]
        self.probability = probability
        self.gradient = []

    def estimate_fisher(self, dataset, allowed_class = None, collate_fn = None):
        est_fisher_info = {}
        change_of_fisher = {}
        for n, p in self.shared.named_parameters():
            if p.requires_grad:
                n = n.replace('.', '__')
                est_fisher_info[n] = p.detach().clone().zero_()
                change_of_fisher[n] = p.detach().clone().zero_()

        mode = self.training
        self.eval()

        data_loader = get_data_loader(dataset, batch_size = 1, collate_fn = collate_fn, augment = True)

        for step, (index, img, label) in enumerate(data_loader):
            if fisher_n is not None:
                if step >= fisher_n:
                    break
            x = img.to(device)
            output = self(x) if allowed_class is None else self(x)[:, allowed_class]
            label = torch.LongTensor([label]).to(device)
            negloglikelihood = F.nll_loss(F.log_softmax(output, dim = 1), label)
            self.zero_grad()
            negloglikelihood.backward()
            for n, p in self.shared.named_parameters():
                if p.requires_grad:
                    n = n.replace('.', '__')
                    if p.grad is not None:
                        est_fisher_info[n] += p.grad.detach()**2

        est_fisher_info = {n: p/step for n, p in est_fisher_info.items()}

        for n, p in self.shared.named_parameters():
            if p.requires_grad:
                n = n.replace('.', '__')
                self.register_buffer('{}_estimated_fisher'.format(n), est_fisher_info[n])
 
        self.train(mode = mode)