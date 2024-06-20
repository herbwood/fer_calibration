import torch
import pickle
from .resnet import ResNet, Bottleneck
import torch.nn as nn
import torch.nn.functional as F

import torch
import pickle
from .abc_modules import ABC_Model
from .resnet import *
from torch.autograd import Variable
from misc.utils import Flatten, mixup_data


class Model(nn.Module):
    
    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args
        resnet50 = ResNet(Bottleneck, [3, 4, 6, 3])
        with open(args.resnet50_path, 'rb') as f:
            obj = f.read()
        weights = {key: torch.from_numpy(arr) for key, arr in pickle.loads(obj, encoding='latin1').items()}
        resnet50.load_state_dict(weights)
        
        self.features = nn.Sequential(*list(resnet50.children())[:-2])  
        self.features2 = nn.Sequential(*list(resnet50.children())[-2:-1])  
        self.fc = nn.Linear(2048, args.num_classes)  
        
        ####################### SCN #######################
        if self.args.trainer == "scn":
            self.alpha = nn.Sequential(
                                nn.Linear(2048, 1),
                                nn.Sigmoid()
                                    )
        ####################### SCN #######################
        
        ####################### RUL #######################
        if self.args.trainer == "rul":
            self.mu = nn.Sequential(
                            nn.BatchNorm2d(2048, eps=2e-5, affine=False),
                            nn.Dropout(p=0.4),
                            Flatten(),
                            nn.Linear(2048 * 7 * 7, 64),
                            nn.BatchNorm1d(64, eps=2e-5)
                                   )

            self.log_var = nn.Sequential(
                                nn.BatchNorm2d(2048, eps=2e-5, affine=False),
                                nn.Dropout(p=0.4),
                                Flatten(),
                                nn.Linear(2048 * 7 * 7, 64),
                                nn.BatchNorm1d(64, eps=2e-5)
                                    )
            self.fc_rul = nn.Linear(64, args.num_classes)
            
            self.backbone = nn.ModuleList([self.features, self.mu, self.log_var])
            self.newly_added = nn.ModuleList([self.fc_rul])
        ####################### RUL #######################
        
    def forward(self, x, target=None):      
        
        #####################BASELINE #####################
        if self.args.trainer == "baseline":
            x = self.features(x) # [N, 2048, h, w]
            feature = self.features2(x) # [N, 2048, 1, 1]
            feature = feature.view(feature.size(0), -1) # [N, 2048]
            output = self.fc(feature) # [N, C]
            
            return output
        #####################BASELINE #####################  
        
        ####################### SCN #######################
        if self.args.trainer == "scn":
            x = self.features(x) # [N, 2048, h, w]
            feature = self.features2(x) # [N, 2048, 1, 1]
            feature = feature.view(feature.size(0), -1)
            attention_weights = self.alpha(feature)
            out = attention_weights * self.fc(feature)
            
            return attention_weights, out 
        ####################### SCN #######################
        
        ####################### RUL #######################
        if self.args.trainer == "rul":
            x = self.features(x) # [N, 2048, h, w]
            mu = self.mu(x)
            logvar = self.log_var(x)
            
            mixed_x, y_a, y_b, att1, att2 = mixup_data(mu, target, logvar.exp().mean(dim=1, keepdim=True), use_cuda=True)
            output = self.fc_rul(mixed_x)
            
            return output, y_a, y_b, att1, att2
        ####################### RUL #######################
        
        ####################### EAC #######################
        if self.args.trainer == "eac":
            x = self.features(x) # [N, 2048, h, w]
            feature = self.features2(x) # [N, 2048, 1, 1]
            feature = feature.view(feature.size(0), -1) # [N, 2048]
            output = self.fc(feature) # [N, C]
            
            params = list(self.parameters())
            fc_weights = params[-2].data # [C, 2048]
            fc_weights = fc_weights.view(1, 7, 2048, 1, 1)
            fc_weights = Variable(fc_weights, requires_grad = False)
            feat = x.unsqueeze(1) # [N, 1, C, 2048, h, w]
            hm = feat * fc_weights
            hm = hm.sum(2) # [N, C, h, w]

            return output, hm
        ####################### EAC #######################
        if self.args.trainer == "lnsu":
            x = self.features(x) # [N, 2048, h, w]
            feature = self.features2(x) # [N, 2048, 1, 1]
            feature = feature.view(feature.size(0), -1) # [N, 2048]
            output = self.fc(feature) # [N, C]
            
            fc_weights = self.fc.weight 
            fc_weights = fc_weights.view(1, self.args.num_classes, 2048, 1, 1)
            fc_weights = Variable(fc_weights, requires_grad = False)
            feat = x.unsqueeze(1) # [N, 1, C, 2048, h, w]
            hm = feat * fc_weights
            hm = hm.sum(2) # [N, C, h, w]
            
            return output, hm
        ####################### EAC #######################
        
        return 