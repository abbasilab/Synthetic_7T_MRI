# 7T MRI Synthesization from 3T Acquisitions
[🔗 View paper on arXiv](https://arxiv.org/abs/2403.08979)

This repository contains PyTorch model implementations for the paper "7T MRI Synthesization from 3T Acquisitions", presented at MICCAI 2024 conference. This model generates synthetic T1-weighted 7T MRIs from T1-weighted 3T MRI inputs. Implemented models are V-Net, Perceptual V-Net, V-Net-GAN, WATNet-2D, and WATNet-3D.

## Requirements

To install requirements:

```sh
pip install -r requirements.txt
```

## Usage

Example command:

```sh
        python run_vnet.py -i 'path_to_input (single file or folder with files)'  -o 'folder_to_save_output' -c 'path_to_model_weight'
```

* Please make sure the input images meet the following criteria:
  - Nifti file
  - T1-weighted 3T image
  - Brain stripped (provide mask files as an optional input under ```-m``` if the image is not brain stripped.)

### Training

* The training parameters should be specified in ```config/params.json```.
* Place paired dataset in ```data/```,and update the dataset filepaths in ```config/params.json```.
* To initiate training, run ```src/scripts/Run_model.py```

### Data Augmentation

* The script for data augmentation is ```src/scripts/data_augmentation.ipynb```
* Transformed datasets are saved under ```data/```

### Pretrained Weights

* Pretrained weights for the base V-Net model can be found on [🔗 Box](https://ucsf.app.box.com/s/yekgjj3wvuih34n6zmcnnr9ji3p2uhng)

## Paper BibTex Citation
If you use this tool, please cite the following reference:

```bibtex
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