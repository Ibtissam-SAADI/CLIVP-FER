from PIL import Image
import os
import numpy as np
import torch
import clip
import torch.utils.data as data
import pandas as pd

clip_model, preprocess = clip.load("ViT-B/32", device="cpu")

class RAFDB(data.Dataset):
    def __init__(self, split='Training', fold=1, transform=None):
        self.transform = transform
        self.split = split
        self.fold = fold
        self.base_image_path = 'EmoLabel/rafdb'
        self.train_file_path = 'EmoLabel/train_labels.csv'
        self.test_file_path = 'EmoLabel/test_labels.csv'
     
        self.file_path = (
            self.train_file_path
            if split == 'Training'
            else self.test_file_path
            
        )
        self.data = pd.read_csv(self.file_path)

        if self.split == 'Training':
            self.data_pixels = np.asarray(self.data['pixels'])
            print("Size of the data:", self.data_pixels.shape)
        else:
            self.data_prvpixels = np.asarray(self.data['pixels'])
            print("Size of the data:", self.data_prvpixels.shape)
        if self.split == 'Training':
            self.data_pixels = self.data['pixels']
            self.data_text = self.data['text']
            self.data_labels = self.data['label']
            self.data_pixels = np.asarray(self.data_pixels)
        else:
            self.data = pd.read_csv(self.test_file_path)
            self.data_prvpixels = self.data['pixels']
            self.data_prvtext = self.data['text']
            self.data_prvlabels = self.data['label']
            self.data_prvpixels = np.asarray(self.data_prvpixels)
     
         
    def __getitem__(self, index):
        row = self.data.iloc[index]
        image_filename = row['pixels']
        image_path = os.path.join(self.base_image_path, image_filename)
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(image)

        
        text = row['text']
        caption = clip.tokenize([text])
        label = torch.tensor(row['label'], dtype=torch.long)

        return img, caption.squeeze(0), label


    def __len__(self):
        if self.split == 'Training':
            return len(self.data_pixels)
        else:
            return len(self.data_prvpixels)    
    