import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import random

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

def print_args(args):
    print("***************")
    print("** Arguments **")
    print("***************")
    optkeys = list(args.__dict__.keys())
    optkeys.sort()
    for key in optkeys:
        print("{}: {}".format(key, args.__dict__[key]))

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

def mixup_data_orig(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion_orig(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def sample_gumbel(args, shape, eps=1e-20):
    U = torch.rand(shape).to(args.device)
    return -torch.log(-torch.log(U + eps) + eps)

def gumbel_sampling(args, output):
    probs = F.softmax(output, dim=-1) # [N, 7]
    confidences, _ = torch.max(probs, 1) # [N]

    inverse_confidence = 1.0 / (confidences + 1e-10)
    inverse_confidence = torch.log(inverse_confidence)

    gumbel_noise = sample_gumbel(args, inverse_confidence.shape)
    y = inverse_confidence + gumbel_noise
    prob_gumbel = F.softmax(y, dim=-1) # [N]

    sample_idxs = torch.multinomial(prob_gumbel, output.size(0), replacement=True) # [N]

    return sample_idxs

def to_one_hot(inp, num_classes):
    
    y_onehot = torch.zeros(inp.size(0), num_classes, device=inp.device, requires_grad=False) # [N, 7]
    # y_onehot.zero_()
    y_onehot.scatter_(1, inp.unsqueeze(1).data, 1)

    return y_onehot # [N, 7]

def special_cutmix(x_i, x_j, label, label_j):

    assert x_i.shape == x_j.shape, "Images must have the same shape"
    
    B, C, H, W = x_i.shape
    
    quadrants = [(0, 0), (0, W // 2), (H // 2, 0), (H // 2, W // 2)]
    
    def get_random_quadrants(n):
        return random.sample(quadrants, n)
    
    synthetic_samples = []
    synthetic_labels = []
    lams = []
    
    for _ in range(3):
        qi = x_i.clone()
        # qj = x_j.clone()
        
        if _ == 0:
            # 1. x_i가 이미지의 3/4, x_j가 1/4을 차지하는 이미지
            h_offset, w_offset = get_random_quadrants(1)[0]
            qi[:, :, h_offset:h_offset + H // 2, w_offset:w_offset + W // 2] = x_j[:, :, h_offset:h_offset + H // 2, w_offset:w_offset + W // 2]
            synthetic_samples.append(qi)
            synthetic_labels.append(3/4 * label + 1/4 * label_j)
            lams.append(3/4)
        
        elif _ == 1:
            # 2. x_i, x_j 모두 2/4를 차지하는 이미지
            selected_quadrants = get_random_quadrants(2)
            for h_offset, w_offset in selected_quadrants:
                qi[:, :, h_offset:h_offset + H // 2, w_offset:w_offset + W // 2] = x_j[:, :, h_offset:h_offset + H // 2, w_offset:w_offset + W // 2]
            synthetic_samples.append(qi)
            synthetic_labels.append(2/4 * label + 2/4 * label_j)
            lams.append(2/4)
        
        else:
            # 3. x_i는 1/4, x_j는 3/4을 차지하는 이미지
            selected_quadrants = get_random_quadrants(3)
            for h_offset, w_offset in selected_quadrants:
                qi[:, :, h_offset:h_offset + H // 2, w_offset:w_offset + W // 2] = x_j[:, :, h_offset:h_offset + H // 2, w_offset:w_offset + W // 2]
            synthetic_samples.append(qi)
            synthetic_labels.append(1/4 * label + 3/4 * label_j)
            lams.append(1/4)
    
    return synthetic_samples, synthetic_labels, lams


def special_cutmix_ver3(x_i, x_j, label, label_j):
    assert x_i.shape == x_j.shape, "Images must have the same shape"
    
    B, C, H, W = x_i.shape
    
    quadrants = [(0, 0), (0, W // 2), (H // 2, 0), (H // 2, W // 2)]
    
    def get_random_quadrants(n):
        return random.sample(quadrants, n)
    
    qi = x_i.clone()
    
    # 랜덤 비율 생성
    lam = random.uniform(0, 1)
    
    synthetic_samples = []
    synthetic_labels = []
    lams = []
    
    if lam <= 0.25:
        # 1. x_i가 이미지의 3/4, x_j가 1/4을 차지하는 이미지
        h_offset, w_offset = get_random_quadrants(1)[0]
        qi[:, :, h_offset:h_offset + H // 2, w_offset:w_offset + W // 2] = x_j[:, :, h_offset:h_offset + H // 2, w_offset:w_offset + W // 2]
        synthetic_label = 3/4 * label + 1/4 * label_j
        synthetic_samples.append(qi)
        synthetic_labels.append(synthetic_label)
        lams.append(0.75)
        
    elif lam <= 0.5:
        # 2. x_i, x_j 모두 2/4를 차지하는 이미지
        selected_quadrants = get_random_quadrants(2)
        for h_offset, w_offset in selected_quadrants:
            qi[:, :, h_offset:h_offset + H // 2, w_offset:w_offset + W // 2] = x_j[:, :, h_offset:h_offset + H // 2, w_offset:w_offset + W // 2]
        synthetic_label = 2/4 * label + 2/4 * label_j
        synthetic_samples.append(qi)
        synthetic_labels.append(synthetic_label)
        lams.append(0.5)
        
    else:
        # 3. x_i는 1/4, x_j는 3/4을 차지하는 이미지
        selected_quadrants = get_random_quadrants(3)
        for h_offset, w_offset in selected_quadrants:
            qi[:, :, h_offset:h_offset + H // 2, w_offset:w_offset + W // 2] = x_j[:, :, h_offset:h_offset + H // 2, w_offset:w_offset + W // 2]
        synthetic_label = 1/4 * label + 3/4 * label_j
        synthetic_samples.append(qi)
        synthetic_labels.append(synthetic_label)
        lams.append(0.25)
    
    return synthetic_samples, synthetic_labels, lams

def calculate_feature_map_ratios_tensor(output_predictions, feature_map):
    # Apply softmax to get the confidence scores
    confidences = F.softmax(output_predictions, dim=1)
    
    # Get the indices of the maximum confidence for each batch
    max_conf_indices = torch.argmax(confidences, dim=1)
    
    batch_size, num_classes, height, width = feature_map.shape
    
    # Initialize a tensor to store ratios
    ratios = torch.zeros(batch_size, 4, dtype=feature_map.dtype, device=feature_map.device)
    
    for i in range(batch_size):
        # Get the feature map for the highest confidence index
        selected_feature_map = feature_map[i, max_conf_indices[i]]
        
        # Divide the selected feature map into 4 parts
        half_height = height // 2
        half_width = width // 2
        
        top_left = selected_feature_map[:half_height, :half_width]
        top_right = selected_feature_map[:half_height, half_width:]
        bottom_left = selected_feature_map[half_height:, :half_width]
        bottom_right = selected_feature_map[half_height:, half_width:]
        
        # Calculate the sum of each part
        sum_top_left = torch.sum(top_left)
        sum_top_right = torch.sum(top_right)
        sum_bottom_left = torch.sum(bottom_left)
        sum_bottom_right = torch.sum(bottom_right)
        
        # Calculate the total sum of the selected feature map
        total_sum = sum_top_left + sum_top_right + sum_bottom_left + sum_bottom_right
        
        # Calculate the ratios
        ratios[i, 0] = sum_top_left / total_sum
        ratios[i, 1] = sum_top_right / total_sum
        ratios[i, 2] = sum_bottom_left / total_sum
        ratios[i, 3] = sum_bottom_right / total_sum
    
    return ratios


def calculate_feature_map_ratios_tensor(feature_map, output_predictions):
    confidences = F.softmax(output_predictions, dim=1)
    max_conf_indices = torch.argmax(confidences, dim=1)
    
    batch_size, num_classes, height, width = feature_map.shape
    
    ratios = torch.zeros(batch_size, 4, dtype=feature_map.dtype, device=feature_map.device)
    
    for i in range(batch_size):
        selected_feature_map = feature_map[i, max_conf_indices[i]]
        
        half_height = height // 2
        half_width = width // 2
        
        top_left = selected_feature_map[:half_height, :half_width]
        top_right = selected_feature_map[:half_height, half_width:]
        bottom_left = selected_feature_map[half_height:, :half_width]
        bottom_right = selected_feature_map[half_height:, half_width:]
        
        sum_top_left = torch.sum(top_left)
        sum_top_right = torch.sum(top_right)
        sum_bottom_left = torch.sum(bottom_left)
        sum_bottom_right = torch.sum(bottom_right)
        
        total_sum = sum_top_left + sum_top_right + sum_bottom_left + sum_bottom_right
        
        ratios[i, 0] = sum_top_left / total_sum
        ratios[i, 1] = sum_top_right / total_sum
        ratios[i, 2] = sum_bottom_left / total_sum
        ratios[i, 3] = sum_bottom_right / total_sum
    
    return ratios, confidences, max_conf_indices

def mix_samples_and_create_label(feature_map1, output_predictions1, feature_map2, output_predictions2, mix_ratios):
    ratios1, confidences1, max_conf_indices1 = calculate_feature_map_ratios_tensor(feature_map1, output_predictions1)
    ratios2, confidences2, max_conf_indices2 = calculate_feature_map_ratios_tensor(feature_map2, output_predictions2)
    
    batch_size = feature_map1.shape[0]
    mixed_labels = torch.zeros(batch_size, feature_map1.shape[1], dtype=feature_map1.dtype, device=feature_map1.device)
    
    for i in range(batch_size):
        for j in range(4):
            mixed_labels[i] += mix_ratios[j] * (ratios1[i, j] * confidences1[i] + ratios2[i, j] * confidences2[i])
        
        # Normalize the mixed_labels to ensure the sum is 1
        mixed_labels[i] = mixed_labels[i] / torch.sum(mixed_labels[i])
    
    return mixed_labels