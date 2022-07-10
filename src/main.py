import torch 
import os 
from manager import Manager
from model.resnet18 import ModifiedResNet18
from data_loader.split_cifar10_data import get_cifar10_data
import sys
  
current_path = os.getcwd()
sys.path.insert(0, str(current_path[:-3]))
log_path = current_path[:-3]

import config.CONFIG as CONFIG
device = CONFIG.DEVICE if torch.cuda.is_available() else 'cpu'

print('log path : ',log_path)


LOG_FILE_DIR = CONFIG.LOG_FILE_DIR
FILE_NAME = CONFIG.FILE_NAME
if os.path.isdir(log_path+'/output') == False:
    os.mkdir(log_path+'/output')
if os.path.isdir(log_path+str(LOG_FILE_DIR)) == False: 
    os.makedirs(log_path+LOG_FILE_DIR)
file = open(log_path+str(LOG_FILE_DIR)+str(FILE_NAME), 'w', encoding='UTF8')
file.close()

dataset_list = CONFIG.dataset_list

for repeat in range(int(CONFIG.REPEAT)):
    TI, after_eval = {}, {}
    FILE = open(log_path+str(LOG_FILE_DIR)+str(FILE_NAME), 'a')
    FILE.write('START' +'\n')
    REP = 'repeat '+str(repeat) +' times'
    FILE.write(str(REP)+ '\n')
    FILE.close()
    
    for i in range(1, len(dataset_list)):
        model = ModifiedResNet18(previous_mask=None, current_task = i+1, make_model= True)
        previous_masks = {}
        new_mask = {}
        for n, p in model.shared.named_parameters():
            if p.requires_grad:
                n = n.replace('.', '__')
                if ('conv' in n) or ('0__weight' in n):
                    mask = torch.randn(p.size()).fill_(1)
                    mask = mask.to(device)
                    previous_masks[n] = mask 
                    mask_mnist = torch.randn(p.size()).fill_(0)
                    mask_mnist = mask_mnist.to(device)
                    new_mask[n] = mask_mnist
        if i > 0:
            ckpt = torch.load(dataset_list[i-1] + '.pt', map_location = device)
            model = ckpt['model']
            previous_masks = ckpt['previous_masks']

        print('Training for Split CIFAR-10 {}'.format(range(i*2, (i+1)*2)))
        train_data, test_data = get_cifar10_data(range(i*2, (i+1)*2))

        model.current_task = i + 1
        model.SELECTION_ON()
        model.add_dataset(dataset_list[i], 2)
        model.set_dataset(dataset_list[i])

        manager = Manager(model, 0.75, previous_masks, new_mask, train_data, test_data)
        if i == 0 : 
            manager.pruner.initialize_first_mask()
        else: 
            manager.pruner.initialize_new_mask()
            mask_ = manager.pruner.previous_masks
            manager.model.previous_mask = mask_

        print('Task {} training...'.format(dataset_list[i]))
        prob = None

        if i != 0: 
            prob = manager.train_selection(dataset_idx = i+1, epochs=CONFIG.SELECTION_EPOCH, isRetrain=False)

            ckpt = torch.load(dataset_list[i-1] + '.pt', map_location = device)
            model = ckpt['model']
            model = model.to(device)
            model.add_dataset(dataset_list[i], 2)
            model.set_dataset(dataset_list[i])
            previous_masks = ckpt['previous_masks']

            manager = Manager(model, 0.75, previous_masks, new_mask, train_data, test_data)
            if i == 0 : 
                manager.pruner.initialize_first_mask()
            else: 
                manager.pruner.initialize_new_mask()

            manager.train(dataset_idx = i+1, epochs=CONFIG.Train_epoch, isRetrain=False, probability_param=prob)
        else: 
            manager.train(dataset_idx = i+1, epochs=CONFIG.Train_epoch, isRetrain=False, probability_param=prob)

        print('task {} pruning...'.format(dataset_list[i]))
        manager.prune(i+1)
        TI[dataset_list[i]] = manager.eval(dataset_idx = i+1)
        FILE = open(log_path+str(LOG_FILE_DIR)+str(FILE_NAME), 'a')
        FILE.write(str(TI)+ '\n')

        FILE.close()

    ###################
    # Eval after train all tasks
    ################### 
    for i in range(len(dataset_list)):
        ckpt = torch.load(dataset_list[-1] + '.pt', map_location = device)
        model = ckpt['model']
        previous_masks = ckpt['previous_masks']
        model.set_dataset(dataset_list[i])

        ckpt_ = torch.load(dataset_list[i] + '.pt', map_location = device)
        new_masks = ckpt_['new_masks']

        train_data, test_data = get_cifar10_data(range(i*2, (i+1)*2))
        manager = Manager(model, 0.75, previous_masks, new_masks, train_data, test_data)
        after_eval[dataset_list[i]] = manager.eval(dataset_idx = i + 1)
    print('Final evalutation :', after_eval)
    print('Task incremental acc :', TI)
    
    assert after_eval == TI, 'Accuracy of some tasks are changed'
