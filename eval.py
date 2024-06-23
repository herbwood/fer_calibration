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
from misc.utils import print_logger, wandb_logger

def main(args):
    
    # Dataloader 
    dataloader = build_dataloader(args)
    test_loader = dataloader["test"]
    
    # Model
    model = Model(args)
    model = model.to(args.device)
    
    trainer = importlib.import_module(f"trainers.{args.trainer}")
    criterion = nn.CrossEntropyLoss(reduction='mean')
    
    if args.wandb_name:
        wandb.init(project=args.wandb_project)
        wandb.run.name = args.wandb_name
        wandb.config.update(args)
        wandb.watch(model)
    
    checkpoint = torch.load(os.path.join(args.train_load_path, "best.pt"))
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metric_dict  = trainer.test_epoch(args, model, test_loader, criterion)
    
    for key, value in test_metric_dict.items():
        print(f"Test {key}: {value:.4f} ", end=" ")
    print()
    
    if args.wandb_name:
        wandb.log(wandb_logger([test_metric_dict], flags=['Test']))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='training argparsers')
    parser.add_argument('--train_load_path', default='/nas_homes/junehyoung/fer_calibration/raf_lnsu', type=str, help='load path')
    parser.add_argument('--dataset_analysis', default='', type=str, help='load path')
    parser.add_argument('--dataset_analysis_path', default='/home/junehyoung/code/fer_calibration/analysis/results/0_dataset_analysis.csv', type=str, help='load path')
    
    parser.add_argument('--wandb_project', default='fer_cal', type=str, help='wandb project name')
    parser.add_argument('--wandb_name', default='lnsu', type=str, help='wandb name')
    parser.add_argument('--trainer', default='lnsu', type=str, help='trainers',choices=['baseline', 'scn', 'rul', 'eac', 'lnsu', 'ours'])
    parser.add_argument('--dataset', default='raf', type=str, help='dataset',choices=['raf', 'affectnet', 'ferplus'])
    parser.add_argument('--num_classes', type=int, default=7, help='raf : 7, affectnet : 8, ferplus : 8')
    parser.add_argument('--basepath', type=str, default='/nas_homes/jihyun/RAF_DB/',  
                        help='raf: /nas_homes/jihyun/RAF_DB/, \
                              affectnet: /nas_homes/jihyun/datasets/AffectNet, \
                              ferplus: /nas_homes/jihyun/FERPlus'
                            )
    parser.add_argument('--metric_key', default='test_acc', type=str, help='metric to save')
    parser.add_argument('--savepath', default='/nas_homes/junehyoung/fer_calibration', type=str, help='path to save weights')
    parser.add_argument('--device', default='cuda', type=str, help='devices to use')
    parser.add_argument('--epochs', default=60, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--batch_size', default=1, type=int, metavar='N', help='train batchsize')
    parser.add_argument('--lr', default=0.0002, type=float, metavar='LR', help='initial learning rate')
    parser.add_argument('--workers', type=int, default=8, help='num of workers to use')
    parser.add_argument('--load_path', type=str, default="/home/jihyun/code/eccv/src/2_0.1_0.3.pth")
    parser.add_argument('--mode', type=str, default='train', help='Mode to run.')
    parser.add_argument('--resnet50_path', type=str, default='/nas_homes/junehyoung/fer_calibration/resnet50_ft_weight.pkl',  help='pretrained_backbone_path')
    parser.add_argument('--log_freq', type=int, default=50,  help='log print frequency')
    parser.add_argument('--label_path', type=str, default='list_patition_label.txt', help='label_path')
    args = parser.parse_args()
    
    main(args)