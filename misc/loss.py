import torch
import torch.nn as nn

import torch.nn.functional as F
from torch.autograd import Variable

def ACLoss(att_map1, att_map2, grid_l, output):
    flip_grid_large = grid_l.expand(output.size(0), -1, -1, -1)
    flip_grid_large = Variable(flip_grid_large, requires_grad = False)
    flip_grid_large = flip_grid_large.permute(0, 2, 3, 1)
    att_map2_flip = F.grid_sample(att_map2, flip_grid_large, mode = 'bilinear', padding_mode = 'border', align_corners=True)
    flip_loss_l = F.mse_loss(att_map1, att_map2_flip)
    return flip_loss_l

def ACLoss2(att_map1, att_map2, grid_l, output):
    flip_grid_large = grid_l.expand(output.size(0), -1, -1, -1)
    flip_grid_large = Variable(flip_grid_large, requires_grad = False)
    flip_grid_large = flip_grid_large.permute(0, 2, 3, 1)
    att_map2_flip = F.grid_sample(att_map2, flip_grid_large, mode = 'bilinear', padding_mode = 'border', align_corners=True)
    flip_loss_l = F.mse_loss(att_map1, att_map2_flip, reduction='none')
    return flip_loss_l
class RankMixup_MRL(nn.Module):
    def __init__(self, num_classes: int = 10,
                       margin: float = 0.1,
                       lambd: float = 0.1,
                       ignore_index: int =-100):
        super().__init__()
        self.margin = margin
        self.lambd = lambd
        self.ignore_index = ignore_index
        self.cross_entropy = nn.CrossEntropyLoss()

    @property
    def names(self):
        return "loss", "loss_ce", "loss_mixup"

    def get_logit_diff(self, inputs, mixup):
        max_values, indices = inputs.max(dim=1)
        max_values = max_values.unsqueeze(dim=1)
       
        max_values_mixup, indices_mixup = mixup.max(dim=1)
        max_values_mixup = max_values_mixup.unsqueeze(dim=1)
        # diff = max_values - max_values_mixup
        diff = max_values_mixup -  max_values 

        return diff
    
    def get_conf_diff(self, inputs, mixup):
        inputs = F.softmax(inputs, dim=1)
        max_values, indices = inputs.max(dim=1)
        max_values = max_values.unsqueeze(dim=1)

        mixup = F.softmax(mixup, dim=1)
        max_values_mixup, indices_mixup = mixup.max(dim=1)
        max_values_mixup = max_values_mixup.unsqueeze(dim=1)
        
        # diff = max_values - max_values_mixup
        diff = max_values_mixup -  max_values 

        return diff

    def forward(self, inputs, targets, mixup, target_re):
        if inputs.dim() > 2:
            inputs = inputs.view(inputs.size(0), inputs.size(1), -1)  # N,C,H,W => N,C,H*W
            inputs = inputs.transpose(1, 2)    # N,C,H*W => N,H*W,C
            inputs = inputs.contiguous().view(-1, inputs.size(2))   # N,H*W,C => N*H*W,C
            targets = targets.view(-1)

        if self.ignore_index >= 0:
            index = torch.nonzero(targets != self.ignore_index).squeeze()
            inputs = inputs[index, :]
            targets = targets[index]

        loss_ce = self.cross_entropy(inputs, targets)
        
        self_mixup_mask = (target_re == 1.0).sum(dim=1).reshape(1, -1) 
        self_mixup_mask = (self_mixup_mask.sum(dim=0) == 0.0) 
     
        # diff = self.get_conf_diff(inputs, mixup) # using probability
        diff = self.get_logit_diff(inputs, mixup)
        loss_mixup = (self_mixup_mask * F.relu(diff+self.margin)).mean()

        loss = loss_ce + self.lambd * loss_mixup

        return loss, loss_ce, loss_mixup
    

class RankMixup_MNDCG(nn.Module):
    def __init__(self, num_classes: int = 10,
                       lambd: float = 0.1,
                       ignore_index: int =-100):
        super().__init__()
        self.lambd = lambd
        self.ignore_index = ignore_index
        self.num_classes = num_classes
        self.cross_entropy = nn.CrossEntropyLoss()

    @property
    def names(self):
        return "loss", "loss_ce", "loss_mixup"

    def get_indcg(self, inputs, mixup, lam, target_re):
        mixup = mixup.reshape(len(lam), -1, self.num_classes) # [3, N, 7]
        target_re = target_re.reshape(len(lam), -1, self.num_classes) # [3, N, 7]
       
        mixup = F.softmax(mixup, dim=2)
        inputs = F.softmax(inputs, dim=1) # [N, 7]

        inputs_lam = torch.ones(inputs.size(0), 1, device=inputs.device) # [N, 1]
        max_values = inputs.max(dim=1, keepdim=True)[0] # get confidences: [N, 1]
        max_mixup = mixup.max(dim=2)[0].t() #  confidences per mixup images: [N, 3]
        max_lam = target_re.max(dim=2)[0].t() # [N, 3]
        
        # compute dcg         
        sort_index = torch.argsort(max_lam, descending=True) # indexes of confidences in descending order: [N, 3]
        max_mixup_sorted = torch.gather(max_mixup, 1, sort_index) # confidences corresponding to sort_indx [N, 3]
        order = torch.arange(1, 2+len(lam), device = max_mixup.device) # [1, 2, 3, 4]
        dcg_order = torch.log2(order + 1)
        
        # [confidence of origial input, confidences of mixup samples]: [N, 4]
        max_mixup_sorted = torch.cat((max_values, max_mixup_sorted), dim=1) 
        dcg = (max_mixup_sorted / dcg_order).sum(dim=1)

        # [label of original input, labels of mixup samples]: [N, 4]
        max_lam_sorted = torch.gather(max_lam, 1, sort_index)
        max_lam_sorted = torch.cat((inputs_lam, max_lam_sorted), dim=1)
        idcg = (max_lam_sorted / dcg_order).sum(dim=1)

        #compute ndcg
        ndcg = dcg / idcg
        inv_ndcg = idcg / dcg
        ndcg_mask = (idcg > dcg)
        ndcg = ndcg_mask * ndcg + (~ndcg_mask) * inv_ndcg   

        return ndcg

    def forward(self, inputs, targets, mixup, target_re, lam):
        if inputs.dim() > 2:
            inputs = inputs.view(inputs.size(0), inputs.size(1), -1)  # N,C,H,W => N,C,H*W
            inputs = inputs.transpose(1, 2)    # N,C,H*W => N,H*W,C
            inputs = inputs.contiguous().view(-1, inputs.size(2))   # N,H*W,C => N*H*W,C
            targets = targets.view(-1)

        if self.ignore_index >= 0:
            index = torch.nonzero(targets != self.ignore_index).squeeze()
            inputs = inputs[index, :]
            targets = targets[index]

        loss_ce = self.cross_entropy(inputs, targets) # [N, 7], [N]
        #NDCG loss
        loss_mixup = (1.0 - self.get_indcg(inputs, mixup, lam, target_re)).mean()
        loss = loss_ce + self.lambd * loss_mixup 

        return loss, loss_ce, loss_mixup