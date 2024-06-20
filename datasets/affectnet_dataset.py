import os
import glob 
import cv2
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm 

class AFFECTNETDataset(Dataset):
    def __init__(self, args, phase, transform=None):
        self.phase = phase
        self.transform = transform
        self.aff_path = args.basepath

        df = self.get_df()
        self.data = df[df['phase'] == phase]

        self.file_paths = self.data.loc[:, 'img_path'].values
        self.label = self.data.loc[:, 'label'].values

        _, self.sample_counts = np.unique(self.label, return_counts=True)

    def get_df(self):
        train_path = os.path.join(self.aff_path,'train_set/')
        val_path = os.path.join(self.aff_path,'val_set/')
        data = []
        
        if self.phase == "train":
            print(f"Loading training data..")
            for anno in tqdm(glob.glob(train_path + 'annotations/*_exp.npy')):
                idx = os.path.basename(anno).split('_')[0]
                img_path = os.path.join(train_path,f'images/{idx}.jpg')
                label = int(np.load(anno))
                data.append(['train',img_path,label])
                
        else:
            print(f"Loading validation data..")
            for anno in tqdm(glob.glob(val_path + 'annotations/*_exp.npy')):
                idx = os.path.basename(anno).split('_')[0]
                img_path = os.path.join(val_path,f'images/{idx}.jpg')
                label = int(np.load(anno))
                data.append(['val',img_path,label])
        
        return pd.DataFrame(data = data,columns = ['phase','img_path','label'])

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        image = cv2.imread(self.file_paths[idx]) 
        image = image[:, :, ::-1] # BGR to RGB
        label = self.label[idx]
        
        if self.transform is not None:
            image = self.transform(image)
        
        return image, label