import torch
import torch.nn as nn

class AvgMeter:
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.cnt = 0
        
    def update(self, val, cnt=1):
        self.val = val
        self.sum += val * cnt
        self.cnt += cnt 
        self.avg = self.sum / self.cnt
        
def print_logger(args, epoch, metric_dict, phase="Train"):
    
    print(f"Epoch: [{epoch+1}/{args.epochs}]", end=" ")
    for key, value in metric_dict.items():
        print(f"{phase} {key}: {value:.4f} ", end=" ")
    print()
    
def wandb_logger(dict_list, flags=['Train', 'Val']):
    wandb_dict = dict()
    for i, metric_dict in enumerate(dict_list):
        for key, value in metric_dict.items():
            wandb_dict[f"{flags[i]} {key}"] = value
            
    return wandb_dict

####################### EAC #######################
def generate_flip_grid(w, h, device):
    # used to flip attention maps
    x_ = torch.arange(w).view(1, -1).expand(h, -1)
    y_ = torch.arange(h).view(-1, 1).expand(-1, w)
    grid = torch.stack([x_, y_], dim=0).float().to(device)
    grid = grid.unsqueeze(0).expand(1, -1, -1, -1)
    grid[:, 0, :, :] = 2 * grid[:, 0, :, :] / (w - 1) - 1
    grid[:, 1, :, :] = 2 * grid[:, 1, :, :] / (h - 1) - 1
    grid[:, 0, :, :] = -grid[:, 0, :, :]
    return grid
####################### EAC #######################

####################### RUL #######################
class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
    
def mixup_data(x, y, att, use_cuda=True):
    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)
    att1 = att / (att + att[index])
    att2 = att[index] / (att + att[index])
    mixed_x = att1 * x + att2 * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, att1, att2

def mixup_criterion(y_a, y_b):
    return lambda criterion, pred:  0.5 *  criterion(pred, y_a) + 0.5 * criterion(pred, y_b)
####################### RUL #######################

####################### LNSU ######################
class LSR2(nn.Module):
    def __init__(self, e):
        super().__init__()
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.e = e

    def _one_hot(self, labels, classes, value=1):
        one_hot = torch.zeros(labels.size(0), classes)
        labels = labels.view(labels.size(0), -1)
        value_added = torch.Tensor(labels.size(0), 1).fill_(value)
        value_added = value_added.to(labels.device)
        one_hot = one_hot.to(labels.device)
        one_hot.scatter_add_(1, labels, value_added)
        return one_hot

    def _smooth_label(self, target, length, smooth_factor):
        one_hot = self._one_hot(target, length, value=1 - smooth_factor)
        mask = (one_hot==0)
        balance_weight = torch.tensor([0.95124031, 4.36690391, 1.71143654, 0.25714585, 0.6191221, 1.74056738, 0.48617274]).to(one_hot.device)
        ex_weight = balance_weight.expand(one_hot.size(0),-1)
        resize_weight = ex_weight[mask].view(one_hot.size(0),-1)
        resize_weight /= resize_weight.sum(dim=1, keepdim=True)
        one_hot[mask] += (resize_weight*smooth_factor).view(-1)
        
#         one_hot += smooth_factor / length
        return one_hot.to(target.device)

    def forward(self, x, target):
        smoothed_target = self._smooth_label(target, x.size(1), self.e)
        x = self.log_softmax(x)
        loss = torch.sum(- x * smoothed_target, dim=1)
        return torch.mean(loss)
####################### LNSU ######################