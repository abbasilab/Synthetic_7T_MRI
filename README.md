# Synthetic_7T_MRI
This repository contains PyTorch model implementations for generating synthetic 7T MRIs from 3T MRI inputs. The models implemented are V-Net, Perceptual V-Net, V-Net-GAN, WATNet-2D, and WATNet-3D

## Requirements

To install requirements:
```setup
pip install -r requirements.txt
```

## Training

* The training parameters should be specified in ```config/params.json```.
* Place paired dataset in ```data/```,and update the dataset in ```config/params.json```.
* To initiate training, run ```src/scripts/Run_model.py```

## Data Augmentation

* The script for data augmentation is ```src/scripts/data_augmentation.ipynb```
* Transformed datasets are saved under ```data/```

## Paper BibTex Citation

7T MRI synthesization from 3T acquisitions

```
@InProceedings{Cui_7T_MICCAI2024,
        author = { Cui, Qiming and Tosun, Duygu and Mukherjee, Pratik and Abbasi-Asl, Reza},
        title = { { 7T MRI Synthesization from 3T Acquisitions } },
        booktitle = {proceedings of Medical Image Computing and Computer Assisted Intervention -- MICCAI 2024},
        year = {2024},
        publisher = {Springer Nature Switzerland},
        volume = {LNCS 15007},
        month = {October},
        page = {35 -- 44}
}
```