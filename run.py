import os 
import argparse 
import importlib 
import wandb

import torch 
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim

from datasets.dataloader import build_dataloader
from models.model import Model
from misc.utils import print_logger, wandb_logger, print_args

def main(args):
    
    # Dataloader 
    dataloader = build_dataloader(args)
    train_loader = dataloader["train"]
    test_loader = dataloader["test"]
    
    # Model
    model = Model(args)
    model = model.to(args.device)
    
    trainer = importlib.import_module(f"trainers.{args.trainer}")
    
    if args.trainer == "rul":
        optimizer = torch.optim.Adam([
        {'params': model.backbone.parameters()},
        {'params': model.newly_added.parameters(), 'lr': 0.002}], lr=0.0002, weight_decay=1e-4)
    else:
        optimizer = torch.optim.Adam(model.parameters() , lr=args.lr, weight_decay=0.0001)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    criterion = nn.CrossEntropyLoss(reduction='mean')
    
    best_metric = 1e+10
    best_epoch= 0
    
    if args.wandb_name:
        wandb.init(project=args.wandb_project)
        wandb.run.name = args.wandb_name
        wandb.config.update(args)
        wandb.watch(model)
    
    savepath = os.path.join(args.savepath, args.wandb_name)
    os.makedirs(savepath, exist_ok=True)
    
    print_args(args)
    
    for epoch in range(args.epochs):
        train_metric_dict = trainer.train_epoch(args, epoch, model, train_loader, optimizer, criterion)
        test_metric_dict  = trainer.test_epoch(args, model, test_loader, criterion)
        lr_scheduler.step()
        
        print_logger(args, epoch, train_metric_dict, phase="Train")
        print_logger(args, epoch, test_metric_dict, phase="Test")
        
        # if test_metric_dict[args.metric_key] > best_metric:
        if test_metric_dict[args.metric_key] < best_metric:
            best_metric = test_metric_dict[args.metric_key]
            best_epoch = epoch
            torch.save({
                        "epoch" : epoch,
                        "model_state_dict" : model.state_dict(), 
                        "optimizer_state_dict" : optimizer.state_dict(), 
                        best_metric : best_metric,
                        }, 
                       os.path.join(savepath, "best.pt"))
            print(f"\nSaved Best Model at Epoch: {epoch+1} with {args.metric_key} {best_metric:.4f}\n")
            
        else:
            print(f"\nStill Best Model with {args.metric_key} {best_metric:.4f} at epoch {best_epoch+1}\n")
        
        if args.wandb_name:
            wandb.log(wandb_logger([train_metric_dict, test_metric_dict], flags=['Train', 'Val']))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='training argparsers')
    parser.add_argument('--wandb_project', default='fer_cal', type=str, help='wandb project name')
    parser.add_argument('--wandb_name', default='rankcutmix', type=str, help='wandb name')
    parser.add_argument('--trainer', default='rankcutmix', type=str, help='trainers',
                        choices=['baseline', 'scn', 'rul', 'eac', 'lnsu', 'ours', 'mixup', 'cutmix', 'rankcutmix', 'rankcutmix_ver3'])
    parser.add_argument('--dataset', default='raf', type=str, help='dataset',choices=['raf', 'affectnet', 'ferplus'])
    parser.add_argument('--num_classes', type=int, default=7, help='raf : 7, affectnet : 8, ferplus : 8')
    parser.add_argument('--basepath', type=str, default='/nas_homes/jihyun/RAF_DB/',  
                        help='raf: /nas_homes/jihyun/RAF_DB/, \
                              affectnet: /nas_homes/jihyun/datasets/AffectNet, \
                              ferplus: /nas_homes/jihyun/FERPlus'
                            )
    parser.add_argument('--batch_size', default=64, type=int, metavar='N', help='train batchsize')
    parser.add_argument('--lr', default=0.0002, type=float, metavar='LR', help='initial learning rate')
    parser.add_argument('--metric_key', default='test_acc', type=str, help='metric to save')
    
    parser.add_argument('--savepath', default='/nas_homes/junehyoung/fer_calibration', type=str, help='path to save weights')
    parser.add_argument('--device', default='cuda', type=str, help='devices to use')
    parser.add_argument('--epochs', default=60, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--workers', type=int, default=8, help='num of workers to use')
    parser.add_argument('--load_path', type=str, default="/home/jihyun/code/eccv/src/2_0.1_0.3.pth")
    parser.add_argument('--mode', type=str, default='train', help='Mode to run.')
    parser.add_argument('--resnet50_path', type=str, default='/nas_homes/junehyoung/fer_calibration/resnet50_ft_weight.pkl',  help='pretrained_backbone_path')
    parser.add_argument('--log_freq', type=int, default=50,  help='log print frequency')
    parser.add_argument('--label_path', type=str, default='list_patition_label.txt', help='label_path')
    
    parser.add_argument('--alpha', default=0.2, type=float, help='sampling degree for beta distribution (for mixup)')
    parser.add_argument('--cutmix_ver', type=str, default='vertical', help='cutmix version to implement')
    parser.add_argument('--margin', default=0.2, type=float, help='margin hyperparameter (for rankcutmix)')
    parser.add_argument('--lambd', default=0.2, type=float, help='hyperparameter for balancing ce loss and rank loss (for rankcutmix)')
    
    args = parser.parse_args()
    
    main(args)