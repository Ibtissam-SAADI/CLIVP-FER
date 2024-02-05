# CLIVP-FER:LEVERAGING VISION LANGUAGE MODELS FOR FACIAL EXPRESSION RECOGNITION
A CLIP based pytorch implementation on facial expression recognition (KMU-FED, FER2013, and RAF-DB), achieving an average accuracy of 97.36%  in KMU-FED, an accuracy of 99.01% and 99.22% in FER2013  and RAF-DB datasets(state-of-the-art)

This is the **official repository** for the [**paper**](https://arxiv.org/abs/) "*CLIVP-FER:LEVERAGING VISION LANGUAGE MODELS FOR FACIAL EXPRESSION RECOGNITION*".

# CLIVP-FER Architecture
![figures/CLIParch12.png](figures/CLIParch12.png)

## FER2013 Dataset ##
- Dataset from https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data
Image Properties: 48 x 48 pixels (2304 bytes)
labels: 0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral
The training set consists of 28,709 examples. The public test set consists of 3,589 examples. The private test set consists of another 3,589 examples.

### Preprocessing Fer2013 ###
- first download the dataset(fer2013.csv) then put it in the "data" folder, then
- python preprocess_fer2013.py

### Train and Eval model ###
- python mainpro_FER.py --model VGG19 --bs 128 --lr 0.01

### plot confusion matrix ###
- python plot_fer2013_confusion_matrix.py --model VGG19 --split PrivateTest

###              fer2013 Accurary             ###

- Model：    VGG19 ;       PublicTest_acc：  71.496% ;     PrivateTest_acc：73.112%     <Br/>
- Model：   Resnet18 ;     PublicTest_acc：  71.190% ;    PrivateTest_acc：72.973%     

## CK+ Dataset ##
- The CK+ dataset is an extension of the CK dataset. It contains 327 labeled facial videos,
We extracted the last three frames from each sequence in the CK+ dataset, which
contains a total of 981 facial expressions. we use 10-fold Cross validation in the experiment.

### Train and Eval model for a fold ###
- python mainpro_CK+.py --model VGG19 --bs 128 --lr 0.01 --fold 1

### Train and Eval model for all 10 fold ###
- python k_fold_train.py

### plot confusion matrix for all fold ###
- python plot_CK+_confusion_matrix.py --model VGG19

###      CK+ Accurary      ###
- Model：    VGG19 ;       Test_acc：   94.646%   <Br/>
- Model：   Resnet18 ;     Test_acc：   94.040% 
