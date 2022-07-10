import time 

DEVICE = 'cuda:0'

Train_epoch = 100
ft_epoch = Train_epoch * 0.9
Retrain_epoch = 10
Selection_warmup = 5

Threshold = 0.01
Thre_freq = 3

NUMBER_OF_TASKS = 5 
Class_PER_TASK = 2 

SELECTION_EPOCH = 15 
alpha_lr = 0.001 
GUMBEL_TAU = 1.0

network = 'resnet18'
dataset_name = 'cifar'

if dataset_name == 'cifar':
    assert NUMBER_OF_TASKS * Class_PER_TASK == 10, 'n task * n cls per task == n original cls'

dataset_list = []
for i in range(0, NUMBER_OF_TASKS):
    dataset_list.append(str(dataset_name)+str(i+1))

REPEAT = 5
t = time.localtime()
LOG_FILE_DIR = '/output' + '/' + str(network) +'/' + str(dataset_name)
FILE_NAME = '/H2_'+str(NUMBER_OF_TASKS) + '_' + str(t.tm_mon) + str(t.tm_mday) + str(t.tm_hour) + str(t.tm_min) + '.log'