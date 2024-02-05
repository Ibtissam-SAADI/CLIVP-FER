import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import transforms as transforms
from sklearn.metrics import f1_score
import numpy as np
import os
import argparse
from cbam import CBAM
import utils
from preprocess_FER2013 import FER2013
from torch.autograd import Variable
import timm
from torchvision import models
import utils
import clip
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

use_cuda = torch.cuda.is_available()
device = "cuda:0" if torch.cuda.is_available() else "cpu"

best_PublicTest_acc = 0  # best PublicTest accuracy
best_PublicTest_acc_epoch = 0
best_PrivateTest_acc = 0  # best PrivateTest accuracy
best_PrivateTest_acc_epoch = 0
start_epoch = 0  # start from epoch 0 or last checkpoint epoch


total_epoch = 60

path = os.path.join(opt.dataset + '_' + opt.model)

# Data
print('==> Preparing data..')
transforms_vaild = torchvision.transforms.Compose([
                              
                                     torchvision.transforms.Resize((224,)),
                                     torchvision.transforms.ToTensor(),
                                     torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225,))
                                    
                                     ])

# For training data , we add some augmentation
transforms_train = torchvision.transforms.Compose([
                                  
                                      torchvision.transforms.Resize((224,)),
                               
                                      torchvision.transforms.RandomRotation(40),
                                      torchvision.transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)),
                                      torchvision.transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
                              
                                      torchvision.transforms.ToTensor(),
                                      torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225,))
                              
                                     ])


trainset = FER2013(split = 'Training', transform=transforms_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=opt.bs, shuffle=True, num_workers=0)
PublicTestset = FER2013(split = 'PublicTest', transform=transforms_vaild)
PublicTestloader = torch.utils.data.DataLoader(PublicTestset, batch_size=64, shuffle=False, num_workers=0)
PrivateTestset = FER2013(split = 'PrivateTest', transform=transforms_vaild)
PrivateTestloader = torch.utils.data.DataLoader(PrivateTestset, batch_size=64, shuffle=False, num_workers=0)

class CustomNet(nn.Module):
    def __init__(self, num_classes, feature_dim):
        super(CustomNet, self).__init__()
        self.fc1 = nn.Linear(feature_dim, 512)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)  # New: Dropout for regularization
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)  # Apply dropout
        x = self.fc2(x)
        return x

class EarlyStopping:
    def __init__(self, patience=7, delta=0.001):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0

    def __call__(self, accuracy, model):
        score = accuracy

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
# Model
if opt.model == 'Ourmodel':
   num_classes = 7 # Adjust based on your task
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

if opt.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir(path), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(os.path.join(path,'PrivateTest_model.t7'))

    net.load_state_dict(checkpoint['net'])
    best_PublicTest_acc = checkpoint['best_PublicTest_acc']
    best_PrivateTest_acc = checkpoint['best_PrivateTest_acc']
    best_PrivateTest_acc_epoch = checkpoint['best_PublicTest_acc_epoch']
    best_PrivateTest_acc_epoch = checkpoint['best_PrivateTest_acc_epoch']
    start_epoch = checkpoint['best_PrivateTest_acc_epoch'] + 1
else:
    print('==> Building model..')



criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=opt.lr, weight_decay=1e-4)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_hours = int(elapsed_time // 3600)
    elapsed_time = elapsed_time - elapsed_hours * 3600
    elapsed_mins = int(elapsed_time // 60)
    elapsed_secs = int(elapsed_time % 60)
    return elapsed_hours, elapsed_mins, elapsed_secs

total_processing_time_train = 0
total_processing_time_test = 0


all_labels = []
all_predictions = []
train_accuracy_values = []
test_accuracy_values = []
train_loss_values = []
test_loss_values = []

early_stopping = EarlyStopping(patience=10, delta=0.001)
# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    global Train_acc
    global total_processing_time_train
    net.to(device)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    avg_pool = nn.AvgPool1d(kernel_size=2)
    start_time = time.monotonic()
    for index, (images, captions, labels) in enumerate(trainloader):

        images, captions, labels = images.to(device), captions.to(device), labels.to(device)
        optimizer.zero_grad()
        image_features = clip_model.encode_image(images)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True) 
        image_features = image_features.unsqueeze(1)  # Reshape for 1D pooling
        image_features = avg_pool(image_features).squeeze(1)
        if opt.mode == 1:
            text_features = clip_model.encode_text(captions)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)  # Modified line
            features = torch.cat((image_features, text_features), dim=1)
        else:
            features = image_features

        pooling_layer = nn.AvgPool1d(kernel_size=2, stride=2)
        features = pooling_layer(features)
        features = features.view(features.size(0), -1) 
        features = features.float()
        start_batch_time = time.time()
        outputs = net(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        end_batch_time = time.time()
        batch_processing_time = end_batch_time - start_batch_time
        total_processing_time_train += batch_processing_time
        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += predicted.eq(labels.data).cpu().sum().item()
        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())
        utils.progress_bar(index, len(trainloader), 'TrainLoss: %.3f | TrainAcc: %.3f%% (%d/%d)'
            % (train_loss/(index+1), 100.*correct/total, correct, total))
    #scheduler.step()  # Update learning rate
    Train_acc = 100.*correct/total
    train_accuracy_values.append(Train_acc)
    train_loss_values.append(train_loss / (index + 1))
    average_processing_time_per_image = total_processing_time_train / len(trainloader.dataset)
    train_f1 = f1_score(all_labels, all_predictions, average='weighted')
    print(f'Epoch {epoch}: Train F1 Score: {train_f1:.4f}')
    print(f'Average Processing Time per Image (Training): {average_processing_time_per_image:.6f} seconds')


def PublicTest(epoch):
    global PublicTest_acc
    global best_PublicTest_acc
    global best_PublicTest_acc_epoch
    net.to(device)
    net.eval()
    PublicTest_loss = 0
    correct = 0
    total = 0
    avg_pool = nn.AvgPool1d(kernel_size=2)
    for index, (images, captions, labels) in enumerate(PublicTestloader):

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
             features = features.float()
             #features = torch.nn.functional.normalize(features)
             start_batch_time = time.time()
             test_pred = net(features)
        
        loss = criterion(test_pred, labels)
        end_batch_time = time.time()

        PublicTest_loss += loss.item()
        _, predicted = torch.max(test_pred.data, 1)
        total += labels.size(0)
        correct += predicted.eq(labels.data).cpu().sum().item()

        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())
        utils.progress_bar(index, len(PublicTestloader), 'TestLoss: %.3f | TestAcc: %.3f%% (%d/%d)'
            % (PublicTest_loss / (index + 1), 100. * correct / total, correct, total))

    test_f1 = f1_score(all_labels, all_predictions, average='weighted')
    print(f'Epoch {epoch}: Test F1 Score: {test_f1:.4f}')


    # Save checkpoint.
    PublicTest_acc = 100.*correct/total
    if PublicTest_acc > best_PublicTest_acc:
        print('Saving..')
        print("best_PublicTest_acc: %0.3f" % PublicTest_acc)
        state = {
            'net': net.state_dict() if use_cuda else net,
            'acc': PublicTest_acc,
            'epoch': epoch,
        }
        if not os.path.isdir(path):
            os.mkdir(path)
        torch.save(state, os.path.join(path,'PublicTest_model2.t7'))
        best_PublicTest_acc = PublicTest_acc
        best_PublicTest_acc_epoch = epoch

def PrivateTest(epoch):
    global PrivateTest_acc
    global best_PrivateTest_acc
    global best_PrivateTest_acc_epoch
    net.to(device)
    net.eval()
    PrivateTest_loss = 0
    correct = 0
    total = 0
    avg_pool = nn.AvgPool1d(kernel_size=2)
    for index, (images, captions, labels) in enumerate(PrivateTestloader):
   
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
             features = features.float()
             start_batch_time = time.time()
             test_pred = net(features)
        
        loss = criterion(test_pred, labels)
        end_batch_time = time.time()
        PrivateTest_loss += loss.item()
        _, predicted = torch.max(test_pred.data, 1)
        total += labels.size(0)
        correct += predicted.eq(labels.data).cpu().sum().item()

        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())
        utils.progress_bar(index, len(PrivateTestloader), 'TestLoss: %.3f | TestAcc: %.3f%% (%d/%d)'
            % (PrivateTest_loss / (index + 1), 100. * correct / total, correct, total))

    test_f1 = f1_score(all_labels, all_predictions, average='weighted')
    print(f'Epoch {epoch}: Test F1 Score: {test_f1:.4f}')
 
    PrivateTest_acc = 100.*correct/total
    early_stopping(PrivateTest_acc, net)
    if early_stopping.early_stop:
        print("Early stopping triggered")
        return True
    if PrivateTest_acc > best_PrivateTest_acc:
        print('Saving..')
        print("best_PrivateTest_acc: %0.3f" % PrivateTest_acc)
        state = {
            'net': net.state_dict() if use_cuda else net,
	        'best_PublicTest_acc': best_PublicTest_acc,
            'best_PrivateTest_acc': PrivateTest_acc,
    	    'best_PublicTest_acc_epoch': best_PublicTest_acc_epoch,
            'best_PrivateTest_acc_epoch': epoch,
        }
        if not os.path.isdir(path):
            os.mkdir(path)
        torch.save(state, os.path.join(path,'PrivateTest_model2.t7'))
        best_PrivateTest_acc = PrivateTest_acc
        best_PrivateTest_acc_epoch = epoch

total_start_time = time.monotonic()
for epoch in range(start_epoch, total_epoch):
    start_time = time.monotonic()
    train(epoch)
    PublicTest(epoch)
    if PrivateTest(epoch):
        break
    end_time = time.monotonic()
    epoch_hours, epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_hours}h {epoch_mins}m {epoch_secs}s')
total_end_time = time.monotonic()
total_hours, total_mins, total_secs = epoch_time(total_start_time, total_end_time)
total_time_estimate_hours = total_hours + (total_mins / 60) + (total_secs / 3600)
print(f'Total Time: {total_hours}h {total_mins}m {total_secs}s | Estimated Total Time: {total_time_estimate_hours:.2f} hours')
    
print("best_PublicTest_acc: %0.3f" % best_PublicTest_acc)
print("best_PublicTest_acc_epoch: %d" % best_PublicTest_acc_epoch)
print("best_PrivateTest_acc: %0.3f" % best_PrivateTest_acc)
print("best_PrivateTest_acc_epoch: %d" % best_PrivateTest_acc_epoch)
