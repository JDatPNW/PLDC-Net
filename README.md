# PLDC-Net

![Banner.](/banner.png)


This repository incldes the code for the architecture of **PLDC-Net** as well as a script to with the set up to train it. The default parameters are set to be the same as they were in the paper when the model was trained on [PLDC-80](https://github.com/JDatPNW/PLDC-80).

## 1. Requirements

| Package      | Version |
|--------------|------|
| python      |3.11.5|
| keras          |2.13.1|
| tensorflow       |2.13.0|

PLDC-Net was tested with these versions listed above. Other versions might work too, but we can not guarantee that they will.

## 2. How to use

- The `PLDC_Net.py` file contains the model architecture for the **PLDC-Net** model, import the model to train it at will.

## 3. PLDC-Net

**PLDC-Net** is an enchanced DenseNet201 base model for plant leaf disease classification. It utalizes SiLU activation and channel attention on top of the basic DenseNet201 to improve performance. It is pre-trained on [PLDC-80](https://github.com/JDatPNW/PLDC-80) and can be used as a base model to further train new domain specific models via domain adaptation techniques such as transfer-learning or few-shot learning (among others) with the domain related weights and improved architecture, using PLDC-Net as the pre-trained base model. 

## 4. Citation

If you use **PLDC-Net** in your work, please cite the following:

```ini
placeholder for citation
```