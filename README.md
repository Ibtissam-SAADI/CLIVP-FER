# CLIVP-FER:LEVERAGING VISION LANGUAGE MODELS FOR FACIAL EXPRESSION RECOGNITION
A CLIP based pytorch implementation on facial expression recognition (KMU-FED, FER2013, and RAF-DB), achieving an average accuracy of 97.36%  in KMU-FED, an accuracy of 99.01% and 99.22% in FER2013  and RAF-DB datasets(state-of-the-art)

This is the **official repository** for the [**paper**](https://arxiv.org/abs/) "*CLIVP-FER:LEVERAGING VISION LANGUAGE MODELS FOR FACIAL EXPRESSION RECOGNITION*".

# CLIVP-FER Architecture
![figures/CLIParch12.png](figures/CLIParch12.png)

## Datasets ##
- KMU-FED dataset from https://cvpr.kmu.ac.kr/KMU-FED.html
- FER2013 dataset from https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data
- RAF-DB  dataset from http://www.whdeng.cn/RAF/model1.html


### Preprocessing ###
- first download the datasets then put them in the "data" folder, then
- python preprocess_datasetname.py

### Train and Test model ###
- python mainpro_FER.py --model VGG19 --bs 128 --lr 0.01

### Train and Eval model for all 10 fold ###
- python k_fold_train.py

### plot confusion matrix ###
- python plot_fer2013_confusion_matrix.py --model VGG19 --split PrivateTest
- 
###  KMU-FED Accurary     ###
We use 10-fold Cross validation in the experiment.
- Model：    VGG19 ;       Average accuracy：  97.364%  <Br/>
###  FER2013 Accurary     ###
- Model：    VGG19 ;       PublicTest_acc：  71.496% ;     PrivateTest_acc：73.112%     <Br/>
###  RAF-DB Accurary     ###
- Model：    VGG19 ;       PublicTest_acc：  71.496% ;     PrivateTest_acc：73.112%     <Br/>

### plot confusion matrix for all fold ###
- python plot_CK+_confusion_matrix.py --model VGG19

<div style="display: flex; justify-content: flex-start;">
  <img width=290 src="figures/both.png"/>
  <img width=290 src="figures/FER20132mtrx.png"/>
  <img width=290 src="figures/RAFmtrx.png"/>
</div>


