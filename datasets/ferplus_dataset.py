import os
import torch.utils.data as data
import pandas as pd
import random
from PIL import Image
import torch
from torch.utils.data import dataset
import pandas as pd
import numpy as np


class FERPLUSDataset(dataset.Dataset):
    """
    Creats a PyTorch custom Dataset for batch iteration
    """
    def __init__(self, args, phase="train", transform=None):
        self.fer_data_dir = args.basepath
        self.transform = transform
        self.mode = phase
        if self.mode == "train":
            self.img_dir = os.path.join(self.fer_data_dir, "FER2013Train")
        elif self.mode == "val":
            self.img_dir = os.path.join(self.fer_data_dir, "FER2013Valid")
        elif self.mode == "test":
            self.img_dir = os.path.join(self.fer_data_dir, "FER2013Test")
        self.label_file = os.path.join(self.img_dir, "label.csv")

        self.label_data_df = pd.read_csv(self.label_file, header=None)
        self.label_data_df.columns = [
            "img_name", "dims", "0", "1", "2", "3", "4", "5", "6", "7",
            "Unknown", "NF"
        ]

        # The arg-max label is the selected as the actual label for Majority Voting
        self.label_data_df['actual_label'] = self.label_data_df[[
            '0', '1', '2', '3', '4', '5', '6', '7'
        ]].apply(lambda x: self._process_row(x), axis=1)

        # get all ilocs with actual label 0
        self.label_data_df.sort_values(by=['img_name'])

        if phase == "train":
            locs0 = sorted(self.label_data_df[
                self.label_data_df['actual_label'] == '0'].index.values)

            # Sampling can be turned off otherwise selects only 40% of neutral ~ 4k images
            sample_indices0 = random.Random(1).sample(locs0,
                                                      int(len(locs0) * 0.6))
            locs1 = sorted(self.label_data_df[
                self.label_data_df['actual_label'] == '1'].index.values)

            # Select only 50% of neutral ~ 4k images
            sample_indices1 = random.Random(1).sample(locs1,
                                                      int(len(locs1) * 0.5))

            locs5 = sorted(self.label_data_df[
                self.label_data_df['actual_label'] == '5'].index.values)
            locs6 = sorted(self.label_data_df[
                self.label_data_df['actual_label'] == '6'].index.values)
            locs7 = sorted(self.label_data_df[
                self.label_data_df['actual_label'] == '7'].index.values)
            self.label_data_df = self.label_data_df.drop(sample_indices0 +
                                                         sample_indices1 +
                                                         locs5 + locs6 + locs7)

        elif phase in ["val", "test"]:
            locs5 = sorted(self.label_data_df[
                self.label_data_df['actual_label'] == '5'].index.values)
            locs6 = sorted(self.label_data_df[
                self.label_data_df['actual_label'] == '6'].index.values)
            locs7 = sorted(self.label_data_df[
                self.label_data_df['actual_label'] == '7'].index.values)
            self.label_data_df = self.label_data_df.drop(locs5 + locs6 + locs7)

        self.image_file_names = self.label_data_df['img_name'].values

    def __getitem__(self, idx):
        img_file_name = self.image_file_names[idx]
        img_file = os.path.join(self.img_dir, img_file_name)
        img = Image.open(img_file).convert('RGB')
        img_class = self.get_class(img_file_name)
        label = torch.tensor(img_class).to(torch.long) # Convert to tensor

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def get_class(self, file_name):
        """
        Returns the label for a corresponding file
        :param file_name: Image file name
        :return:
        """
        row_df = self.label_data_df[self.label_data_df["img_name"] ==
                                    file_name]
        init_val = -1
        init_idx = -1
        for x in range(2, 10):
            max_val = max(init_val, row_df.iloc[0].values[x])
            if max_val > init_val:
                init_val = max_val
                init_idx = int(
                    x - 2
                )  # Labels indices must start at 0, -2 if all else -4!!!!!!
        return init_idx
    
    def _process_row(self, row):
        return np.argmax(row)

    def __len__(self):
        return len(self.image_file_names)