Code used for the results in the paper  ["ClassMix: Segmentation-Based Data Augmentation for Semi-Supervised Learning"](https://arxiv.org/abs/2007.07936)
## Getting started
### Prerequisites
*  CUDA/CUDNN 
*  Python3
*  Packages found in requirements.txt

### Datasets

#### Cityscapes
Download the dataset from the Cityscapes dataset server([Link](https://www.cityscapes-dataset.com/)). Download the files named 'gtFine_trainvaltest.zip', 'leftImg8bit_trainvaltest.zip' and extract in ../data/CityScapes/

#### Pascal VOC 2012
Download the dataset from [here](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/). Download the file 'training/validation data' under 'Development kit' and extract in ../data/VOC2012/. For training, you will also need to download additional labels from this [link](https://drive.google.com/file/d/1P0YpiEX1t-gY954WFXlKII667PAs4WNP/view?usp=sharing), extract this directory into ../data/VOC2012.

### Input arguments
Arguments related to running the script are specified from terminal and include; number of gpus to use (if >1 torch.nn.DataParalell is used), path to configuration file (see below), path to .pth file if resuming training, name of the experiment, and whether to save images during training. More details can be found in the relevant scripts.

Arguments related to the algoritms are specified in the configuration files. These include model, data, hyperparameters related to the training, and what methods to apply on unlabeled data. A full description is provided further below.

### Examples
#### Training a model with semi-supervised learning with example config on a single gpu

```python3 trainSSL.py --config ./configs/configCityscapes.json --name name_of_training```

#### Resuming training of a model with semi-supervised learning

```python3 trainSSL.py --resume path/to/checkpoint.pth --name name_of_training```

#### Evaluating a trained model

```python3 evaluateSSL.py --model-path path/to/checkpoint.pth```

### Pretrained model
[Here](https://drive.google.com/file/d/1JXlFsEqCi8onvmS8JwhZlYXJIIe0Mdyj/view?usp=sharing) is a model trained with SSL with 1/8 (372) labeled samples for Cityscapes.


