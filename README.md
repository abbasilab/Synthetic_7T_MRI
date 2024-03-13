# Synthetic_7T_MRI
This repository contains PyTorch model implementations for generating synthetic 7T MRIs from 3T MRI inputs. The models implemented are V-Net, WATNet-2D, and WATNet-3D

## Requirements

To install requirements:
```setup
pip install -r requirements.txt
```

## Training

The training parameters should be specified in ```config/params.json```.
Place paired dataset in ```data/```,and update the dataset in ```config/params.json```.
To initiate training, run ```src/scripts/Run_model.py```

## Data Augmentation

The script for data augmentation is ```src/scripts/data_augmentation.ipynb```
Transformed datasets are saved under ```data/```

## Paper

7T MRI synthesization from 3T acquisitions

