import os
import cv2
from torch.utils.data import Dataset
import pandas as pd
from torchvision import transforms

class RAFDataset(Dataset):
    def __init__(self, args, phase, transform=None):
        self.args = args
        self.raf_path = args.basepath
        self.phase = phase
        self.transform = transform
        df = pd.read_csv(os.path.join(self.raf_path, 'EmoLabel', args.label_path), sep=' ', header=None)
        
        name_c = 0
        label_c = 1
        if phase == 'train':
            dataset = df[df[name_c].str.startswith('train')]
        else:
            df = pd.read_csv(os.path.join(self.raf_path, 'EmoLabel/list_patition_label.txt'), sep=' ', header=None)
            dataset = df[df[name_c].str.startswith('test')]
            
        self.label = dataset.iloc[:, label_c].values - 1
        images_names = dataset.iloc[:, name_c].values
        self.file_paths = []
        self.clean = (args.label_path == 'list_patition_label.txt')
        
        for f in images_names:
            f = f.split(".")[0]
            f += '_aligned.jpg'
            file_name = os.path.join(self.raf_path, 'Image/aligned', f)
            self.file_paths.append(file_name)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        label = self.label[idx]
        image = cv2.imread(self.file_paths[idx]) 
        image = image[:, :, ::-1] # BGR to RGB
        
        if self.transform is not None:
            image = self.transform(image)
        
        if self.args.trainer == "eac" or self.args.trainer == "lnsu":
            image1 = transforms.RandomHorizontalFlip(p=1)(image)
            return image, label, idx, image1, self.file_paths[idx]
        
        if self.args.trainer == "scn":
            return image, label, idx, self.file_paths[idx]
        else:
            return image, label, self.file_paths[idx]