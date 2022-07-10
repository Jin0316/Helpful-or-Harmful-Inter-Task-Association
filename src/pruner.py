import torch
import sys
import os 

current_path = os.getcwd()
sys.path.insert(0, str(current_path[:-3]))
import config.CONFIG as CONFIG

device = CONFIG.DEVICE

class Pruner(object):
    """Performs pruning on the given model."""

    def __init__(self, model, prune_perc, previous_masks, new_masks):
        self.model = model
        self.prune_perc = prune_perc

        self.train_bias = False
        self.train_bn = False
        self.current_masks = None
        self.threshold = None
        self.new_masks = new_masks
        self.previous_masks = previous_masks
        valid_key = list(previous_masks.keys())[0] #return first key of previous_masks
        self.current_dataset_idx = previous_masks[valid_key].max() 
        self.prob = None

    def apply_weight_fisher(self):
        for n, p in self.model.shared.named_parameters():
            n = n.replace('.','__')
            if ('conv' in n) or ('0__weight' in n):
                grad_square = getattr(self.model, '{}_estimated_fisher'.format(n))
                for i in range(int(self.current_dataset_idx - 1)):
                    mask = self.previous_masks[n].eq(i+1)
                    grad_square[mask.eq(1)] = grad_square[mask.eq(1)] * self.prob[i]
                self.model.register_buffer('{}_estimated_fisher'.format(n), grad_square)

    """NOTE: This is the criterion of sharing weight when dataset idx > 1 """
    def find_fisher_mean(self):
        temp = 0.0 
        count = 0.0
        for n, p in self.model.shared.named_parameters():
            n = n.replace('.', '__')
            if ('conv' in n) or ('0__weight' in n):
                grad_square = getattr(self.model, '{}_estimated_fisher'.format(n))
                grad_square_ = grad_square.to(device)
                temp += grad_square_.mean()
                count += 1.0
        return temp/count 

    def find_fisher_std(self):
        total_array = None
        for n, p in self.model.shared.named_parameters():
            n = n.replace('.', '__')
            if ('conv' in n) or ('0__weight' in n):
                grad_square = getattr(self.model, '{}_estimated_fisher'.format(n))
                grad_square = grad_square.clone()
                grad_square = grad_square.flatten()
                if total_array is not None:
                    total_array = torch.cat((total_array, grad_square), dim = 0)
                else: 
                    total_array = grad_square
        if total_array.std() == float('inf'):
            return 1 
        else: 
            return total_array.std()
    
    def select_forward_weights_fisher(self, current_epoch, select):
        mean_value = self.find_fisher_mean()
        std_value = self.find_fisher_std()

        self.prob = select
        self.apply_weight_fisher()
        
        self.threshold = CONFIG.Threshold

        for n, p in self.model.shared.named_parameters():
            n = n.replace('.', '__')
            if ('conv' in n) or ('0__weight' in n):
                mask = self.previous_masks[n].to(device)
                p_clone = p.detach()
                p_clone[mask.eq(0)] = 0.0

                grad_square = getattr(self.model, '{}_estimated_fisher'.format(n))
                grad_square = grad_square.to(device)
                grad_square = (grad_square)/std_value

                new_mask = mask.lt(self.current_dataset_idx).__and__(grad_square.gt(self.threshold))
                new_mask = new_mask.__and__(mask.gt(0))
                
                self.new_masks[n][new_mask.eq(1)] = self.new_masks[n][new_mask.eq(1)] + 1
                self.new_masks[n] = self.new_masks[n].to(device)

                over_masks = self.new_masks[n].gt(current_epoch/CONFIG.Thre_freq)
                over_masks = over_masks.to(device)

                make_zero = over_masks.eq(1).__or__(mask.eq(self.current_dataset_idx))
                make_zero = make_zero.to(device)

                p_clone[make_zero.eq(0)] = 0.0 
                
        
    """NOTE: This is criterion of pruning when dataset idx == 1 """
    def pruning_mask_weights(self, weights, previous_mask, layer_name):
        """Pruning criterion: Based on the fisher matrix. 
        """
        # Select all prunable weights, ie. belonging to current dataset.
        previous_mask = previous_mask.to(device)
        tensor = weights[previous_mask.eq(self.current_dataset_idx)] 
        abs_tensor = tensor.abs()
        cutoff_rank = round(self.prune_perc * tensor.numel()) 
        cutoff_value = abs_tensor.view(-1).cpu().kthvalue(cutoff_rank)[0]

        remove_mask = weights.abs().le(cutoff_value) * previous_mask.eq(self.current_dataset_idx)

        previous_mask[remove_mask.eq(1)] = 0 #set zero to pruned weights
        mask = previous_mask
        # print('Layer #%s, pruned %d/%d (%.2f%%) (Total in layer: %d)' %
        #       (layer_name, mask.eq(0).sum(), tensor.numel(),
        #        100 * mask.eq(0).sum() / tensor.numel(), weights.numel()))
        return mask

    def prune(self):
        """Gets pruning mask for each layer, based on previous_masks.
           Sets the self.current_masks to the computed pruning masks.
        """
        print('Pruning for dataset idx: %d' % (self.current_dataset_idx))

        self.previous_masks = self.current_masks

        # print('Pruning each layer by removing %.2f%% of values' % (100 * self.prune_perc))
        for n, p in self.model.shared.named_parameters(): 
            n = n.replace('.', '__')
            if ('conv' in n) or ('0__weight' in n):
                if p.requires_grad:
                    p_ = p.detach().clone()
                    mask = self.pruning_mask_weights(p_, self.previous_masks[n], n)
                    self.current_masks[n] = mask.to(device)
                    p = p.detach()
                    p[self.current_masks[n].eq(0)] = 0.0
                

    def make_grads_zero(self):
        """Sets grads of fixed weights to 0."""
        assert self.current_masks
        for n, p in self.model.shared.named_parameters():
            if p.requires_grad:
                n = n.replace('.', '__')
                if ('conv' in n) or ('0__weight' in n):
                    layer_mask = self.current_masks[n]
                    if p.grad is not None:
                        p.grad.data[layer_mask.ne(self.current_dataset_idx)] = 0
                    

    def make_pruned_zero(self):
        """Makes pruned weights 0."""
        assert self.current_masks
        for n, p in self.model.shared.named_parameters():
            n = n.replace('.', '__')
            if ('conv' in n) or ('0__weight' in n):
                layer_mask = self.current_masks[n]
                p_clone = p.detach()
                p_clone[layer_mask.eq(0)] = 0.0


    def apply_mask(self, dataset_idx, total_epoch):
        """To be done to retrieve weights just for a particular dataset."""
        for n, p in self.model.shared.named_parameters():
            n = n.replace('.', '__')
            if ('conv' in n) or ('0__weight' in n):
                mask = self.previous_masks[n].to(device)

                p_clone = p.detach()
                p_clone[mask.eq(0)] = 0.0           # make pruned zero. 

                new_mask = self.new_masks[n].to(device)
                mask = self.previous_masks[n].to(device)
                apply_mask = new_mask.gt(total_epoch/CONFIG.Thre_freq).__or__(mask.eq(dataset_idx))
                apply_mask = apply_mask.to(device)
                p_clone[apply_mask.eq(0)] = 0.0

    def apply_mask_task_one(self, dataset_idx):
        """To be done to retrieve weights just for a particular dataset."""
        for n, p in self.model.shared.named_parameters():
            n = n.replace('.', '__')
            if ('conv' in n) or ('0__weight' in n):
                mask = self.previous_masks[n].to(device)
                p = p.detach()
                p[mask.ne(dataset_idx)] = 0.0 

    def select_mask(self, dataset_idx, select):
        selected = (select != 1).nonzero() + 1.0
        selected = selected.to('cuda:1')
        self.model.to(device)
        for n, p in self.model.shared.named_parameters():
            n = n.replace('.', '__')
            if ('conv' in n) or ('0__weight' in n):
                mask = self.previous_masks[n].to(device)
                p_clone = p.detach()
                p_clone[mask.eq(0)] = 0.0 # make pruned zero. 
                for s in selected:
                    p_clone[mask.eq(s)] = 0.0
                

    def initialize_imagenet_mask(self):
        """Turns previously pruned weights into trainable weights for
           current dataset.
        """
        assert self.previous_masks
        self.current_masks = self.previous_masks 

    def initialize_first_mask(self):
        assert self.previous_masks
        self.current_masks = self.previous_masks 

    def initialize_new_mask(self):
        """Turns previously pruned weights into trainable weights for
           current dataset.
        """
        assert self.previous_masks
        self.current_dataset_idx += 1
        for n, p in self.model.shared.named_parameters():
            n = n.replace('.', '__')
            if ('conv' in n) or ('0__weight' in n):
                mask = self.previous_masks[n]
                mask[mask.eq(0)] = self.current_dataset_idx
        self.current_masks = self.previous_masks

    def concat_original_model(self, dataset_idx, original_model):
        for (n, p), (original_n, original_p) in zip(self.model.shared.named_parameters(), original_model.shared.named_parameters()):
            n = n.replace('.', '__')
            if ('conv' in n) or ('0__weight' in n):
                weight = p.detach()
                original_weight = original_p.detach()
                mask = self.previous_masks[n].to(device)
                mask_ = mask.lt(dataset_idx).__and__(mask.gt(0)) 
                weight[mask_.eq(1)] = original_weight[mask_.eq(1)]


    def initialize_new_masked_weights(self, dataset_idx):
        for n, p in self.model.shared.named_parameters():
            n = n.replace('.', '__')
            if ('conv' in n) or ('0__weight' in n):
                weight = p.detach()
                mask = self.previous_masks[n].to(device)
                random_init = 0.001 * torch.randn((p.size())).to(device)
                weight[mask.eq(dataset_idx)] = random_init[mask.eq(dataset_idx)]