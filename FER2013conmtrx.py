import itertools
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import argparse
from torch.autograd import Variable
import torchvision
import transforms as transforms
from sklearn.metrics import confusion_matrix, classification_report
from FER2013data import FER2013
from torch.autograd import Variable
from torchvision import models
import csv
from models.clip import clip
import timm
import time
from torchvision import models

import ssl 
ssl._create_default_https_context = ssl._create_unverified_context


parser = argparse.ArgumentParser(description='PyTorch Fer2013 CNN Training')
parser.add_argument('--model', type=str, default='Ourmodel', help='CNN architecture')
parser.add_argument('--mode', type=int, default=1, help='CNN architecture')
parser.add_argument('--dataset', type=str, default='2FER2013', help='CNN architecture')
parser.add_argument('--bs', default=32, type=int, help='learning rate')
parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')


opt = parser.parse_args()
device = "cuda:0" if torch.cuda.is_available() else "cpu"

transforms_vaild = torchvision.transforms.Compose([
                                    
                                     torchvision.transforms.Resize((224,)),

                                     torchvision.transforms.ToTensor(),

                                     torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225,))
                        
                                     ])


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=16)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else '.2f'
    #fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
     plt.text(j, i, format(cm[i, j], fmt),
             horizontalalignment="center",
             color="white" if cm[i, j] > thresh else "black") 



    plt.ylabel('True label', fontsize=18)
    plt.xlabel('Predicted label', fontsize=18)
    plt.tight_layout()


class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

class CustomNet(nn.Module):
    def __init__(self, num_classes, feature_dim):
        super(CustomNet, self).__init__()
        self.fc1 = nn.Linear(feature_dim, 512)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)  
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)  # Apply dropout
        x = self.fc2(x)
        return x
 
# Model
if opt.model == 'Ourmodel':
   num_classes = 7  # Adjust based on your task
   clip_model, preprocess = clip.load("ViT-B/32", device=device)
   dim = 0.5
   clip_model = clip_model.float()
   clip_model.eval()
   feature_dim = clip_model.visual.output_dim
   input_dim = 1024
   hidden_dims = [512, 256, 128]  
   output_dim = 6
 
   if opt.mode == 0:
       
       net = CustomNet(num_classes=num_classes, feature_dim=256)
      
   if opt.mode == 1:
     
        net = CustomNet(num_classes=num_classes, feature_dim=384)
   else:
        net = torch.nn.Sequential(
                torch.nn.ReLU(),
                torch.nn.Linear(512, num_classes))
    
    
    
path = os.path.join(opt.dataset + '_' + opt.model)
checkpoint = torch.load('2FER2013_Ourmodel\PrivateTest_model2.t7')

net.load_state_dict(checkpoint['net'])
net.to(device)
net.eval()

PrivateTestset = FER2013(split = 'PrivateTest', transform=transforms_vaild)
PrivateTestloader = torch.utils.data.DataLoader(PrivateTestset, batch_size=64, shuffle=False, num_workers=0)

correct = 0
total = 0
all_target = []

avg_pool = nn.AvgPool1d(kernel_size=2)
for index, (images, captions, labels) in enumerate(PrivateTestloader):
       
        #print('size',len(testloader.dataset))
        images, captions, labels = images.to(device), captions.to(device), labels.to(device)
        with torch.no_grad():
             image_features = clip_model.encode_image(images)
             image_features = image_features.unsqueeze(1)  # Reshape for 1D pooling
             image_features = avg_pool(image_features).squeeze(1)
             if opt.mode ==1:
                text_features = clip_model.encode_text(captions)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                features = torch.cat((image_features, text_features), dim = 1)
             else:
                features = image_features

             pooling_layer = nn.AvgPool1d(kernel_size=2, stride=2) 
             features = pooling_layer(features)
             features = features.view(features.size(0), -1)
             #features = torch.nn.functional.normalize(features)
             features = features.float()
             #features = torch.nn.functional.normalize(features)
         
             test_pred = net(features)

             _, predicted = torch.max(test_pred.data, 1)
             total += labels.size(0)
             correct += predicted.eq(labels.data).cpu().sum().item()
 
         # Collect all true labels and predictions for F1 score calculation
             if index == 0:
                all_predicted = predicted
                all_targets = labels
             else:
                all_predicted = torch.cat((all_predicted, predicted), 0)
                all_targets = torch.cat((all_targets, labels), 0)

acc = 100. * correct / total
print("accuracy: %0.3f" % acc)

print('Classification Report:\n', classification_report(all_targets.data.cpu().numpy(), all_predicted.cpu().numpy(), target_names=class_names))
# Compute confusion matrix
matrix = confusion_matrix(all_targets.data.cpu().numpy(), all_predicted.cpu().numpy())
np.set_printoptions(precision=2)

# Plot normalized confusion matrix
plt.figure(figsize=(10, 8))
plot_confusion_matrix(matrix, classes=class_names, normalize=False,
                      title= ' Confusion Matrix (Accuracy: %0.3f%%)' %acc)
plt.savefig(os.path.join(opt.dataset + '_' + opt.model, 'FER20132mtrx.png'))
plt.close()