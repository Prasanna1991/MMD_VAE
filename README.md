# MMD_VAE

This repo contains the code for our AAAI-Workshop paper "Learning disentangled representations from 12-lead electrograms: application in localizing the origin of Ventricular Tachycardia"

As the experiments involved private dataset, the data loader are defined to take random number with same dimensions as input data.

## Dependencies

* PyTorch
* Numpy

## Description

**train_{modelName}.py** contains the training procedure of the model.
  
**models.py** contains the models used in the experiments.

**utils.py** contains utility functions.

**eval.py** contains the evaluation [i.e. linear classifier] to evaluate the model. 
