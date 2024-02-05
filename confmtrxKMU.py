from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import transforms as transforms
import argparse
import itertools
import os
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
from models.clip import clip
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import timm
from preprocess_KMUFED import KMU
from torchvision import models

from models import *


parser = argparse.ArgumentParser(description='PyTorch CK+ CNN Training')
parser.add_argument('--model', type=str, default='Ourmodel', help='CNN architecture')
parser.add_argument('--mode', type=int, default=1, help='CNN architecture')
parser.add_argument('--dataset', type=str, default='KMU-FED', help='dataset')
parser.add_argument('--fold', default=1, type=int, help='k fold number')
parser.add_argument('--bs', default=64, type=int, help='batch_size')
parser.add_argument('--lr', default=0.003, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--workers', type=int, default=8)
parser.add_argument('--epochs', type=int, default=1)


opt = parser.parse_args()
device = "cuda:0" if torch.cuda.is_available() else "cpu"

file_path = 'des5data5.csv'
transforms_vaild = torchvision.transforms.Compose([
                               
                                     torchvision.transforms.Resize((224,)),
                            
                                     torchvision.transforms.ToTensor(),
                          
                                     torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225,)),
                                  
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

class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sadness', 'Surprise']

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
        x = self.dropout(x)  
        x = self.fc2(x)
        return x
 
# Model
if opt.model == 'Ourmodel':
   num_classes = 6  # Adjust based on your task
   clip_model, preprocess = clip.load("ViT-B/32", device=device)
   dim = 0.5
   clip_model = clip_model.float()
   clip_model.eval()

   if opt.mode == 0:
       
       net = CustomNet(num_classes=num_classes, feature_dim=512)
      
   if opt.mode == 1:
     
        net = CustomNet(num_classes=num_classes, feature_dim=384)
   else:
        net = torch.nn.Sequential(
                torch.nn.ReLU(),
                torch.nn.Linear(512, num_classes))



all_target = []
fold_accuracies = []
overall_conf_matrix = np.zeros((6, 6)) 
class_labels = [0, 1, 2, 3, 4, 5]
for i in range(10):

    correct = 0
    total = 0
    print("%d fold" % (i+1))
    path = os.path.join(opt.dataset + '_' + opt.model,  '%d' %(i+1))
    checkpoint = torch.load(os.path.join(path, 'best_modelokok01.t7'))

    net.load_state_dict(checkpoint['net'])
    #net.cuda()
    net.to(device)
    net.eval()
    testset = KMU(split = 'Testing', fold = i+1, transform=transforms_vaild, file_path=file_path)
    testloader = torch.utils.data.DataLoader(testset, batch_size=16, shuffle=False)
  
    avg_pool = nn.AvgPool1d(kernel_size=2)
    for index, (images, captions, labels) in enumerate(testloader):
       
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

             pooling_layer = nn.AvgPool1d(kernel_size=2, stride=2) #nn.MaxPool1d(kernel_size=2, stride=2)
             features = pooling_layer(features)
             features = features.view(features.size(0), -1)
             features = features.float()
         
             test_pred = net(features)

        _, predicted = torch.max(test_pred.data, 1)
        total += labels.size(0)
        correct += predicted.eq(labels.data).cpu().sum().item()
        conf_matrix = confusion_matrix(labels.data.cpu().numpy(), predicted.cpu().numpy(), labels=class_labels)
        overall_conf_matrix += conf_matrix
         # Collect all true labels and predictions for F1 score calculation
        if index == 0 and i == 0:
            all_predicted = predicted
            all_targets = labels
        else:
            all_predicted = torch.cat((all_predicted, predicted), 0)
            all_targets = torch.cat((all_targets, labels), 0)
 

 
    fold_acc = 100. * correct / total
    fold_accuracies.append(fold_acc)
    print("Fold %d accuracy: %0.3f" % (i+1, fold_acc))


average_accuracy = sum(fold_accuracies) / len(fold_accuracies)
print("Average accuracy: %0.3f" % average_accuracy)


print('Classification Report:\n', classification_report(all_targets.data.cpu().numpy(), all_predicted.cpu().numpy(), target_names=class_names))

print('Average Confusion Matrix:\n', overall_conf_matrix)
plt.figure(figsize=(10, 8))
plot_confusion_matrix(overall_conf_matrix, classes=class_names, normalize=False,
                      title='Average Confusion Matrix (Average Accuracy: %0.3f%%)' % average_accuracy)
plt.savefig(os.path.join(opt.dataset + '_' + opt.model, 'Overall_CLIPmatrix.png'))
plt.close()
