# LEVERAGING VISION LANGUAGE MODELS FOR FACIAL EXPRESSION RECOGNITION IN DRIVING ENVIRONMENT
A CLIP based pytorch implementation on facial expression recognition (KMU-FED), achieving an average accuracy of 97.36%  in KMU-FED.

This is the **official repository** for the [**paper**](https://arxiv.org/abs/) "*LEVERAGING VISION LANGUAGE MODELS FOR FACIAL EXPRESSION RECOGNITION IN DRIVING ENVIRONMENT*".

# CLIVP-FER Architecture
<div style="display: flex; justify-content: flex-start;">
  <img width=800 src="figures/CLIParcht.png"/>
</div>

## Datasets ##
- KMU-FED dataset from https://cvpr.kmu.ac.kr/KMU-FED.html

### Preprocessing ###
-*For KMU-FED dataset*: 'python formatdescription.py' to add the text description to the images and save them as CSV format, then, put them in the "data" folder. <Br/>
'python preprocess_KMUFED.py' to preprocess the image and text data. <Br/>

### Train and Test model ###
*Mode 0*: Image features only.<Br/>
*Mode 1*: Image and text features.

### Train and Test model for all 10 fold ###
- *KMU-FED dataset*: python 10fold_train.py

### plot confusion matrix ###
- python KMUconfmtrx.py --mode 1

###  KMU-FED Accurary     ###
We use 10-fold Cross validation in the experiment.
- Model：    CLIVP-FER ;       Average accuracy：  97.364%  <Br/>

### Confusion matrices ###

<div style="display: flex; justify-content: flex-start;">
  <img width=600 src="figures/both.png"/>
</div>


