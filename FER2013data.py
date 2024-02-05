from PIL import Image
import numpy as np
import torch
import clip
import torch.utils.data as data
import pandas as pd

clip_model, preprocess = clip.load("ViT-B/32", device="cpu")

class FER2013(data.Dataset):
    def __init__(self, split='Training', fold=1, transform=None):
        self.transform = transform
        self.split = split
        self.fold = fold
        self.train_file_path = 'data\Training1.csv'
        self.public_file_path = 'data\PublicTest1.csv'
        self.private_file_path = 'data\PrivateTest1.csv'
        self.file_path = (
            self.train_file_path
            if split == 'Training'
            else self.public_file_path
            if split == 'PublicTest'
            else self.private_file_path
        )
        self.data = pd.read_csv(self.file_path)

        if self.split == 'Training':
  
            self.data_pixels = self.data['pixels']
            self.data_text = self.data['text']
            self.data_labels = self.data['label']
            self.data_pixels = np.asarray(self.data_pixels)

        elif self.split == 'PublicTest':
            self.data = pd.read_csv(self.public_file_path)
            self.data_pubpixels = self.data['pixels']
            self.data_pubtext = self.data['text']
            self.data_publabels = self.data['label']
            self.data_pubpixels = np.asarray(self.data_pubpixels)
            
        else:
            self.data = pd.read_csv(self.private_file_path)
            self.data_prvpixels = self.data['pixels']
            self.data_prvtext = self.data['text']
            self.data_prvlabels = self.data['label']
            self.data_prvpixels = np.asarray(self.data_prvpixels)
           

    def __getitem__(self, index):
        row = self.data.iloc[index]
        image_pixels = row['pixels']

        # Convert the string representation of pixels to a NumPy array
        image_pixels = np.array([int(pixel) for pixel in image_pixels.split()], dtype=np.uint8)

        # Determine the image dimensions based on the length of the array
        img_height = img_width = int(np.sqrt(len(image_pixels)))

        # Reshape the array to the original image dimensions
        image_pixels = image_pixels.reshape((img_height, img_width))

        # Convert grayscale image to RGB
        img = image_pixels[:, :, np.newaxis]
        img = np.concatenate((img, img, img), axis=2)
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        text = row['text']
        caption = clip.tokenize([text])
        label = torch.tensor(row['label'], dtype=torch.long)

        return img, caption.squeeze(0), label


    def __len__(self):
        if self.split == 'Training':
            return len(self.data_pixels)
        elif self.split == 'PublicTest':
            return len(self.data_pubpixels)
        else:
            return len(self.data_prvpixels)    
    