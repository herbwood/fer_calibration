import time
import datetime 
from tqdm import tqdm 
import torch 
from torch.nn import functional as F
from misc.utils import AvgMeter
import torchmetrics
from tqdm import tqdm

def train_epoch(args, epoch, model, train_loader, optimizer, criterion):
    
    batch_meter = AvgMeter()
    train_loss_meter = AvgMeter()
    train_acc_meter = AvgMeter()
    meter_dict = dict()
    acc_metric = torchmetrics.Accuracy(task="multiclass", num_classes=args.num_classes).to(args.device)
    
    margin_1 = 0.07
    margin_2 = 0.2 
    beta = 0.7 
    relabel_epoch = 10
    
    end = time.time()
    model.train()
    
    for step, (data, label, idx) in enumerate(train_loader):
        
        tops = int(data.size(0) * beta)
        optimizer.zero_grad()
        
        data = data.to(args.device)
        label = label.to(args.device)
        idx = idx.to(args.device)
        
        attention_weights, output = model(data)
        
        # rank regularization 
        _, top_idx = torch.topk(attention_weights.squeeze(), tops)
        _, down_idx = torch.topk(attention_weights.squeeze(), data.size(0) - tops, largest=False)
        
        high_group = attention_weights[top_idx]
        low_group = attention_weights[down_idx]
        high_mean = torch.mean(high_group)
        low_mean = torch.mean(low_group)
        
        diff = low_mean - high_mean + margin_1 
        
        if diff > 0:
            RR_loss = diff 
        else:
            RR_loss = 0.0 
            
        loss_ce = criterion(output, label)
        loss = loss_ce + RR_loss
        
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
        
        if epoch + 1 > relabel_epoch:
            softmax_output = torch.softmax(output, dim=1)
            Pmax, predicted_labels = torch.max(softmax_output, 1)
            Pgt = torch.gather(softmax_output, 1, label.view(-1, 1)).squeeze()
            true_or_false = Pmax - Pgt > margin_2 
            update_idx = true_or_false.nonzero().squeeze()
            label_idx = idx[update_idx]
            relabels = predicted_labels[update_idx]
            train_loader.dataset.label[label_idx.cpu().numpy()] = relabels.cpu().numpy()
        
    meter_dict["train_loss"] = train_loss_meter.avg
    meter_dict["train_acc"] = train_acc_meter.avg 
    
    return meter_dict 


def test_epoch(args, model, test_loader, criterion):
    
    test_loss_meter = AvgMeter()
    test_acc_meter = AvgMeter()
    meter_dict = dict()
    acc_metric = torchmetrics.Accuracy(task="multiclass", num_classes=args.num_classes).to(args.device)

    model.eval()
    
    with torch.no_grad():
        for step, (data, label, idx) in enumerate(tqdm(test_loader)):
            
            data = data.to(args.device)
            label = label.to(args.device)
            
            _, output = model(data)
            loss = criterion(output, label)
            acc = acc_metric(output.argmax(dim=-1), label).item()
            
            test_loss_meter.update(loss.item(), data.size(0))
            test_acc_meter.update(acc, data.size(0))
            
        meter_dict["test_loss"] = test_loss_meter.avg
        meter_dict["test_acc"] = test_acc_meter.avg 
        
        return meter_dict 