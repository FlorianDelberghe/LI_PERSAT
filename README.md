# ISCAT Cell Segmentation using U-Net  
**Florian Delberghe**
-------

This repo contains the code produces during a Lab Immersion project in the UPPESAT Lab at EPFL. The goal was to build a deep learning architecture to segment cells and pili in iSCAT image stacks.
The code present here contains all the preprocessing steps for the iSCAT images as well as the architecture of the models used to segment cells and pili.


## Project hiearchy

build_training.py: Builds the trainig set and train target for all the main tasks in the project

train_model.py: Trains the models either cell or pili detection

test_model.py: Computes network output on a sample of testing sets

code

----data_loader: loads data during training steps of the models

----models: UNet models and their components

----processing: Preprocessing and other image related functions

----utilities: Usefull functions


## Using the code

*Requiered Packages:*  
See requierments.txt. Use the following command in a new environment to install all the dependencies
```
pip install --user --requirement requirements.txt
``` 

*Build training datasets*
Builds dataset for a given task, dataset arg in [bright_field, iscat, fluo, pili]
```
python build_training.py <dataset>
```

*Training the model*
Trains either the cell or pili segmenting nets (required argument), optional args `<name>` the name given to the model for saving the weights, `<data_patjh>` for the relative path of the data given to the data_loader
```
pyhton train_model.py <cell / pili> <name> <data_path>
```

*Testing the model*
```
python test_model.py 
```





