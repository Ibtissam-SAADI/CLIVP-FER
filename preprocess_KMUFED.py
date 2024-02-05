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

# Example usage
# file_path = 'path_to_your_data.csv'
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
# ])


# class KMU(data.Dataset):
#     def __init__(self, split='Training', images, captions, labels, fold = 1, transform=None):
#         self.transform = transform
#         self.split = split  # training set or test set
#         self.fold = fold # the k-fold cross validation
#         self.data = pd.read_csv('des3data.csv') # 
#         self.images = images
#         self.text = clip.tokenize(captions)
#         self.label = torch.from_numpy(labels)
    

#         number = len(self.data['labels']) #981
#         sum_number = [0,196,316,516,725,905,1104] # the sum of class number
#         test_number = [19,12,20,21,18,20]#should modify 19 to 19.6
#         #sum_number = [0,196,316,516,725,905,1104] # the sum of class number
#         #test_number = [39,24,40,42,36,40]
        
#         test_index = []
#         train_index = []

#         for j in range(len(test_number)):
#             for k in range(test_number[j]):
#                 if self.fold != 10: #the last fold start from the last element
#                     test_index.append(sum_number[j]+(self.fold-1)*test_number[j]+k)
#                 else:
#                     test_index.append(sum_number[j+1]-1-k)

#         for i in range(number):
#             if i not in test_index:
#                 train_index.append(i)

#         print(len(train_index),len(test_index))
#         print(f"Fold {self.fold}: Train samples: {len(train_index)}, Test samples: {len(test_index)}")

#         # now load the picked numpy arrays
#         if self.split == 'Training':
#             self.train_data = []
#             self.train_inputs = []
#             self.train_labels = []
#             for ind in range(len(train_index)):
#                 self.train_data.append(self.data['images'][train_index[ind]])
#                 self.train_inputs.append(self.data['text'][train_index[ind]])
#                 self.train_labels.append(self.data['labels'][train_index[ind]])
             
#         elif self.split == 'Testing':
#             self.test_data = []
#             self.test_inputs = []
#             self.test_labels = []
#             for ind in range(len(test_index)):
#                 self.test_data.append(self.data['images'][test_index[ind]])
#                 self.test_inputs.append(self.data['text'][test_index[ind]])
#                 self.test_labels.append(self.data['labels'][test_index[ind]])

 

#     def __getitem__(self, index):
#     # Apply the transform to the image only
#         if self.split == 'Training':
#             # img, target = self.train_data[index], self.train_labels[index]
#             image = np.asarray(self.train_data[index])
#             caption = np.asarray(self.train_inputs[index])
           

#             return torch.tensor(image).float(), caption, self.y[index]
        
#         elif self.split == 'Testing':
#             image = np.asarray(self.test_data[index])
#             caption = np.asarray(self.test_inputs[index])

#             return torch.tensor(image).float(), caption, self.y[index]
    
#     def __len__(self):
#         if self.split == 'Training':
#             return len(self.train_data)
#         elif self.split == 'Testing':
#             return len(self.test_data)

