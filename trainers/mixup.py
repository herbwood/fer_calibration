import time
import datetime 
from tqdm import tqdm 
import torch 
from torch.nn import functional as F
from misc.utils import AvgMeter
import torchmetrics
from tqdm import tqdm
import os 
import pandas as pd 
from misc.metric import ECELoss
from misc.utils import mixup_data_orig, mixup_criterion_orig

def train_epoch(args, epoch, model, train_loader, optimizer, criterion):
    
    batch_meter = AvgMeter()
    train_loss_meter = AvgMeter()
    train_acc_meter = AvgMeter()
    meter_dict = dict()
    acc_metric = torchmetrics.Accuracy(task="multiclass", num_classes=args.num_classes).to(args.device)
        
    end = time.time()
    model.train()
    
    for step, (data, label, filename) in enumerate(train_loader):
        
        data = data.to(args.device)
        label = label.to(args.device)
        
        data, label_a, label_b, lam = mixup_data_orig(data, label, args.alpha)
        output = model(data)
        loss = mixup_criterion_orig(criterion, output, label_a, label_b, lam)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        acc = acc_metric(output.argmax(dim=-1), label).item()
        train_loss_meter.update(loss.item(), data.size(0))
        train_acc_meter.update(acc, data.size(0))
        
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
    # df = pd.read_csv(args.dataset_analysis_path)
    
    with torch.no_grad():
        for step, (data, label, filename) in enumerate(tqdm(test_loader)):
            
            data = data.to(args.device)
            label = label.to(args.device)
            
            output = model(data)
            loss = criterion(output, label)
            # confidence = torch.max(F.softmax(output, dim=1), dim=1)[0].item()
            
            # if args.dataset_analysis:
            #     if args.trainer not in df.columns:
            #         df[args.trainer] = 0.0 
            #     filename = os.path.basename(filename[0]).replace('_aligned', '')
            #     df.loc[df['filename'] == filename, args.trainer] = confidence
            acc = acc_metric(output.argmax(dim=-1), label).item()
            ece = ece_metric(output, label).item()
            
            test_loss_meter.update(loss.item(), data.size(0))
            test_acc_meter.update(acc, data.size(0))
            test_ece_meter.update(ece, data.size(0))
            
        meter_dict["test_loss"] = test_loss_meter.avg
        meter_dict["test_acc"] = test_acc_meter.avg 
        meter_dict["test_ece"] = test_ece_meter.avg
        
        # print(df.tail(20))
        # df.to_csv("/home/junehyoung/code/fer_calibration/analysis/results/ontest.csv", index=False)
        
        return meter_dict 