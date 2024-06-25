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
from misc.utils import gumbel_sampling, to_one_hot, special_cutmix_ver3
from misc.loss import RankMixup_MRL

def train_epoch(args, epoch, model, train_loader, optimizer, criterion):
    
    batch_meter = AvgMeter()
    train_loss_meter = AvgMeter()
    train_ce_loss_meter = AvgMeter()
    train_rank_loss_meter = AvgMeter()
    train_acc_meter = AvgMeter()
    meter_dict = dict()
    acc_metric = torchmetrics.Accuracy(task="multiclass", num_classes=args.num_classes).to(args.device)
    
    rank_criterion = RankMixup_MRL(num_classes=args.num_classes, margin=args.margin, lambd=args.lambd)
    # rank_criterion = RankMixup_MNDCG(num_classes=args.num_classes, lambd=args.lambd)
        
    end = time.time()
    model.train()
    
    for step, (data, label, filename) in enumerate(train_loader):
        
        data = data.to(args.device) # [N, 3, 224, 224]
        label = label.to(args.device) # [N]
        
        output = model(data) # [N, 7]
        
        # 1) gumbel distribution sampled indexes 
        gumbel_idxs = gumbel_sampling(args, output) # [N] 
        
        # 2) permute input and label 
        data_j = data[gumbel_idxs] # [N, 3, 224, 224]
        label_j = label[gumbel_idxs] # [N]
        
        # 3) one-hot encode label and label_j 
        perm_label = to_one_hot(label, args.num_classes) # [N, 7]
        perm_label_j = to_one_hot(label_j, args.num_classes) # [N, 7]
        
        # 4) cutmix on data and label 
        # length : 1
        synthetic_samples, synthetic_labels, lams = special_cutmix_ver3(data, data_j, perm_label, perm_label_j)
        
        new_data = torch.cat(synthetic_samples, dim=0) # [N x 1, 3, 224, 224]
        new_label = torch.cat(synthetic_labels, dim=0) # [N x 1, 7]
        
        new_output = model(new_data) # [N x 3, 7]
        
        loss, loss_ce, loss_mixup = rank_criterion(output, label, new_output, new_label)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        acc = acc_metric(output.argmax(dim=-1), label).item()
        train_loss_meter.update(loss.item(), data.size(0))
        train_ce_loss_meter.update(loss_ce.item(), data.size(0))
        train_rank_loss_meter.update(loss_mixup.item(), data.size(0))
        train_acc_meter.update(acc, data.size(0))
        
        batch_meter.update(time.time() - end)
        
        nb_remain = 0
        nb_remain += len(train_loader) - step -1
        nb_remain += (args.epochs - epoch -1) * len(train_loader)
        eta_seconds = batch_meter.avg * nb_remain
        eta = str(datetime.timedelta(seconds=int(eta_seconds)))
        
        if step % args.log_freq == 0 or step == len(train_loader) - 1:
            print(f"Epoch: [{epoch+1}/{args.epochs}] Iter: [{step}/{len(train_loader)}] Train loss: {loss.item():.4f}  Train CE loss: {loss_ce.item():.4f}  Train Rank loss: {loss_mixup.item():.4f} ETA: {eta}")
        
        end = time.time()
        
    meter_dict["train_loss"] = train_loss_meter.avg
    meter_dict["train_ce_loss"] = train_ce_loss_meter.avg
    meter_dict["train_rank_loss"] = train_rank_loss_meter.avg
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