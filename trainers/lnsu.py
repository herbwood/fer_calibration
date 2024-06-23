import time
import datetime 
from tqdm import tqdm 
import torch 
from torch.nn import functional as F
from misc.utils import AvgMeter
import torchmetrics
from tqdm import tqdm
from misc.utils import generate_flip_grid, LSR2
from misc.loss import ACLoss2
from misc.metric import ECELoss

def train_epoch(args, epoch, model, train_loader, optimizer, criterion):
    
    batch_meter = AvgMeter()
    train_loss_meter = AvgMeter()
    train_acc_meter = AvgMeter()
    meter_dict = dict()
    acc_metric = torchmetrics.Accuracy(task="multiclass", num_classes=args.num_classes).to(args.device)
        
    end = time.time()
    model.train()
    
    for step, (image1, label, idx, image2, filename) in enumerate(train_loader):
        
        image1 = image1.to(args.device)
        image2 = image2.to(args.device)
        label = label.to(args.device)
        
        output, feat_map1 = model(image1)
        output2, feat_map2 = model(image2)
        
        grid_l = generate_flip_grid(7, 7, args.device)
        flip_loss = ACLoss2(feat_map1, feat_map2, grid_l, output)
        flip_loss = flip_loss.mean(dim=-1).mean(dim=-1) #N*7
        
        if args.dataset == "raf":
            balance_weight = torch.tensor([0.95124031, 4.36690391, 1.71143654, 0.25714585, 0.6191221, 1.74056738, 0.48617274]).cuda().view(args.num_classes,1)
        else:
            balance_weight = torch.tensor([1, 1, 1, 1, 1, 1, 1]).cuda().view(args.num_classes,1)
        flip_loss = torch.mm(flip_loss, balance_weight).squeeze()
        
        loss = LSR2(0.3)(output, label) + 0.1 * flip_loss.mean()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        acc = acc_metric(output.argmax(dim=-1), label).item()
        train_loss_meter.update(loss.item(), image1.size(0))
        train_acc_meter.update(acc, image1.size(0))
        
        batch_meter.update(time.time() - end)
        
        nb_remain = 0
        nb_remain += len(train_loader) - step -1
        nb_remain += (args.epochs - epoch -1) * len(train_loader)
        eta_seconds = batch_meter.avg * nb_remain
        eta = str(datetime.timedelta(seconds=int(eta_seconds)))
        
        if step % args.log_freq == 0 or step == len(train_loader) - 1:
            print(f"Epoch: [{epoch+1}/{args.epochs}] Iter: [{step}/{len(train_loader)}] Train loss: {loss.item():.4f} ETA: {eta}")
        
        end = time.time()
        
    meter_dict["train_loss"] = train_loss_meter.avg
    meter_dict["train_acc"] = train_acc_meter.avg 
    
    return meter_dict 


def test_epoch(args, model, test_loader, criterion):
    
    test_loss_meter = AvgMeter()
    test_acc_meter = AvgMeter()
    test_ece_meter = AvgMeter()
    meter_dict = dict()
    acc_metric = torchmetrics.Accuracy(task="multiclass", num_classes=args.num_classes).to(args.device)
    ece_metric = ECELoss().to(args.device)

    model.eval()
    
    with torch.no_grad():
        for step, (image1, label, idx, image2, filename) in enumerate(test_loader):
        
            image1 = image1.to(args.device)
            image2 = image2.to(args.device)
            label = label.to(args.device)
            
            output, feat_map1 = model(image1)
            loss = criterion(output, label) 
            acc = acc_metric(output.argmax(dim=-1), label).item()
            ece = ece_metric(output, label).item()
            
            test_loss_meter.update(loss.item(), image1.size(0))
            test_acc_meter.update(acc, image1.size(0))
            test_ece_meter.update(ece, image1.size(0))
            
        meter_dict["test_loss"] = test_loss_meter.avg
        meter_dict["test_acc"] = test_acc_meter.avg 
        meter_dict["test_ece"] = test_ece_meter.avg
        
        return meter_dict 