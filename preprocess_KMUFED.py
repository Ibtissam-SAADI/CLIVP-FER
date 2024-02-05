from __future__ import print_function
from PIL import Image
import numpy as np
import torch
import clip
import torch.utils.data as data
import pandas as pd

clip_model, preprocess = clip.load("ViT-B/32", device="cpu")
class KMU(data.Dataset):
    def __init__(self, split='Training', fold=1, transform=None, file_path=None):
        self.transform = transform
        self.split = split
        self.fold = fold
        self.file_path = file_path
        self.data = pd.read_csv(self.file_path)

        number = len(self.data['labels'])
        sum_number = [0, 196, 316, 516, 725, 905, 1104]
        test_number = [19, 12, 20, 21, 18, 20]
        # sum_number = [0,196,316,516,725,905,1104] # the sum of class number
        # test_number = [39,24,40,42,36,40]

        test_index = []
        train_index = []

        for j in range(len(test_number)):
            for k in range(test_number[j]):
                if self.fold != 10:
                    test_index.append(sum_number[j] + (self.fold - 1) * test_number[j] + k)
                else:
                    test_index.append(sum_number[j + 1] - 1 - k)

        for i in range(number):
            if i not in test_index:
                train_index.append(i)

        self.train_data = self.data.loc[train_index] if self.split == 'Training' else self.data.loc[test_index]
        print(len(train_index),len(test_index))
    
    def __getitem__(self, index):
        row = self.train_data.iloc[index]
        image_path = row['images']
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        text = row['text']
        caption = clip.tokenize([text])
        label = row['labels']
        label = torch.tensor(label, dtype=torch.long)

        return image, caption.squeeze(0), label

    def __len__(self):
        return len(self.train_data)
