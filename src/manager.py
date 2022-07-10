import torch.nn as nn 
import torch 

import torch.optim as optim
import torch.utils.data as data
from torch.utils import data

import copy
from torch.autograd import Variable
import torch.optim.lr_scheduler as lr_scheduler

from pruner import Pruner
from tqdm import tqdm
import sys
import os 
  
current_path = os.getcwd()
sys.path.insert(0, str(current_path[:-3]))
import config.CONFIG as CONFIG

device = CONFIG.DEVICE

class Manager(object):
    """Handles training and pruning."""

    def __init__(self, model, pruning_rate, previous_masks, new_masks, train_data, test_data, selected = None):
        self.model = model
        self.batch_size = 128
        self.train_loader = data.DataLoader(train_data, batch_size = self.batch_size, shuffle = True, num_workers = 2)
        self.test_loader = data.DataLoader(test_data, self.batch_size, num_workers = 2)
        self.train_data = train_data
        
        self.criterion = nn.CrossEntropyLoss()
        self.pruning_rate = pruning_rate
        self.pruner = Pruner(self.model, self.pruning_rate, previous_masks, new_masks)

        self.probability_params = None 
        self.pro_optimizer = None
        self.selected = selected
        self.others = [param for name, param in self.model.named_parameters() if name != 'probability']

        self.optimizer = optim.SGD(params = self.others, lr=0.1, momentum=0.9)
        self.lr_scheduler = lr_scheduler.MultiStepLR(self.optimizer, milestones = [CONFIG.Train_epoch * 0.5, CONFIG.Train_epoch * 0.75], gamma = 0.1)

    def eval(self, dataset_idx, biases=None):
        """Performs evaluation."""

        self.model = self.model.to(device)
        self.model.start_finding(start = False)

        if dataset_idx == 1:
            self.pruner.apply_mask_task_one(dataset_idx)
        else:
            self.pruner.apply_mask(dataset_idx, total_epoch = CONFIG.ft_epoch)

        self.model.eval()

        test_loss = 0
        total = 0
        correct = 0

        print('Performing eval...')
        time_count = 0
        with torch.no_grad():
            for step, (index, image, label) in enumerate(self.test_loader):

                image = image.to(device)
                label = label.to(device)
                image, label = Variable(image), Variable(label)
                pred = self.model(image)
                time_count += 1
                loss = self.criterion(pred, label)
                test_loss += loss.item()

                _, predicted = torch.max(pred.data, 1)
                correct += (predicted == label).sum().item()
                total += label.size(0) 
        
        acc = 100 * correct / total
        print('[test] loss: {:.3f} | acc: {:.3f}'.format(test_loss/(step+1), acc))

        return acc


    def train_selection(self, dataset_idx, epochs, isRetrain=False):
        """Performs training."""
        self.model = self.model.to(device)
        self.model.train()
        self.optimizer = optim.SGD(params = self.others, lr=0.01, momentum=0.9)
        
        if dataset_idx > 1: 
            original_model = copy.deepcopy(self.model)

        if isRetrain == 0 and dataset_idx > 1:               
            self.pruner.initialize_new_masked_weights(dataset_idx)

        epochs = tqdm(range(epochs))
        for epoch in epochs:
            running_loss = 0
            acc = 0
            total = 0
            correct = 0

            running_loss = 0.0
            for step, (index, image, label) in enumerate(self.train_loader):
                image = image.to(device)
                label = label.to(device)
                self.optimizer.zero_grad()
                image, label = Variable(image), Variable(label)

                pred = self.model(image)
                loss = self.criterion(pred, label)
                loss.backward(retain_graph = True)
                if epoch >= CONFIG.Selection_warmup:
                    self.model.update_sampled()
                self.pruner.make_grads_zero()   # set fixed param grads to 0.
                self.optimizer.step()                # update parameters
                self.pruner.make_pruned_zero()  # set pruned weights to zero

                running_loss += loss.item()

                _, predicted = torch.max(pred.data, 1)
                correct += (predicted == label).sum().item()
                total += label.size(0)
        
            self.lr_scheduler.step()
            acc = 100 * (correct / total)

            epochs.set_description('Task-association for Task {} | loss: {:.3f} | acc: {:.3f}'.format(dataset_idx, running_loss/(step+1), acc))
            if dataset_idx > 1:
                self.pruner.concat_original_model(dataset_idx, original_model)

            self.probability_params = nn.functional.gumbel_softmax(self.model.probability, 1.0)
            print(self.probability_params)
            
        del self.model

        return self.probability_params

    def train(self, dataset_idx, epochs, isRetrain=False, probability_param = None):
        """Performs training."""
        self.model = self.model.to(device)
        self.model.train()
        self.model.start_finding(start = False)

        if isRetrain == 0:
            self.optimizer = optim.SGD(self.model.parameters(), lr = 0.01, momentum = 0.9)
        elif isRetrain == 1: 
            self.optimizer = optim.SGD(self.model.parameters(), lr = 0.001, momentum = 0.9)

        if dataset_idx > 1: 
            original_model = copy.deepcopy(self.model)

        if isRetrain == 0 and dataset_idx > 1:               
            self.pruner.initialize_new_masked_weights(dataset_idx)
            self.selected = probability_param.clone().detach()
            # self.selected = torch.tensor(probability_param).clone()

        epochs = tqdm(range(epochs))
        for epoch in epochs:
            running_loss = 0
            total = 0
            correct = 0

            optimizer = self.optimizer

            if isRetrain == 1 and dataset_idx > 1: 
                self.pruner.apply_mask(dataset_idx, total_epoch= CONFIG.ft_epoch)

            elif isRetrain == 1 and dataset_idx == 1: 
                self.pruner.apply_mask_task_one(dataset_idx)

            if epoch >= 0 and epoch < (CONFIG.ft_epoch) and dataset_idx > 1 and isRetrain == 0:
                self.model.estimate_fisher(self.train_data)
                self.pruner.select_forward_weights_fisher(current_epoch=epoch, select=self.selected)

            elif epoch > (CONFIG.ft_epoch) and dataset_idx > 1 and isRetrain == 0: 
                self.pruner.apply_mask(dataset_idx, total_epoch= CONFIG.ft_epoch)

            self.model.start_finding(start = False)
            running_loss = 0.0
            for step, (index, image, label) in enumerate(self.train_loader):
                image = image.to(device)
                label = label.to(device)
                # optimizer.zero_grad()
                self.model.zero_grad()
                image, label = Variable(image), Variable(label)
                
                pred = self.model(image)
                loss = self.criterion(pred, label)
                loss.backward(retain_graph = True)
                self.pruner.make_grads_zero()   
                optimizer.step()                
                self.pruner.make_pruned_zero()  

                running_loss += loss.item()

                _, predicted = torch.max(pred.data, 1)
                correct += (predicted == label).sum().item()
                total += label.size(0)
            self.lr_scheduler.step()
            acc = 100 * (correct / total)
            epochs.set_description('Task {} train phase | loss: {:.3f} | acc: {:.3f}'.format(dataset_idx, running_loss/(step+1), acc))
            if dataset_idx > 1:
                self.pruner.concat_original_model(dataset_idx, original_model)

    def save_model(self, dataset_idx):
        """Saves model to file."""
        model = self.model
        self.model.save_bn()
        ckpt = {
            'previous_masks': self.pruner.current_masks,
            'new_masks' : self.pruner.new_masks,
            'model': model,
            'selected': self.selected,
        }
        torch.save(ckpt, 'cifar'+str(dataset_idx)+'.pt')

    def prune(self, dataset_idx):
        """Perform pruning."""
        retraining_epochs = CONFIG.Retrain_epoch

        self.pruner.prune()

        print('retraining after pruning...')
        self.model.start_finding(start = False)
        self.train(dataset_idx, retraining_epochs, True)
        self.save_model(dataset_idx)