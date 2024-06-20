import torch
import importlib
import torchvision.transforms as transforms

def build_dataloader(args):
    
    # Dataloader 
    if args.trainer == "baseline":
        transform_train = transforms.Compose(
                [
                        transforms.ToPILImage(),
                        transforms.Resize((224, 224)),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                            std=[0.229, 0.224, 0.225]),
                ])
    else:
        transform_train = transforms.Compose(
                [
                        transforms.ToPILImage(),
                        transforms.Resize((224, 224)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                            std=[0.229, 0.224, 0.225]),
                        transforms.RandomErasing(scale=(0.02, 0.25)),
                ])

    transform_val = transforms.Compose(
            [
                    transforms.ToPILImage(),
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                         std=[0.229, 0.224, 0.225]), 
        ])
    
    train_dataset = getattr(importlib.import_module(
                        f"datasets.{args.dataset}_dataset"),
                        f"{args.dataset.upper()}Dataset")(
                            args,
                            "train",
                            transform=transform_train
                        )
    
    test_dataset = getattr(importlib.import_module(
                        f"datasets.{args.dataset}_dataset"),
                        f"{args.dataset.upper()}Dataset")(
                            args,
                            "test",
                            transform=transform_val
                        )
                        
    train_dataloader =  torch.utils.data.DataLoader(
                            train_dataset,
                            batch_size=args.batch_size,
                            shuffle=True,
                            num_workers=args.workers,
                            pin_memory=True,
                            )
    
    test_dataloader =  torch.utils.data.DataLoader(
                            test_dataset,
                            batch_size=args.batch_size,
                            shuffle=False,
                            num_workers=args.workers,
                            pin_memory=True,
                            )
                        
    dataloader_dict = dict()
    dataloader_dict["train"] = train_dataloader
    dataloader_dict["test"] = test_dataloader
    
    return dataloader_dict
    