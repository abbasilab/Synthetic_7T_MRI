#%%
import torch
import torchvision
torchvision.disable_beta_transforms_warning()
import argparse
import torchio as tio
import numpy as np
import nibabel as nib
from box import Box
import os.path as path
import torch
import sys, os, json

sys.path.append(os.path.join(sys.path[0], '../'))
from dataloader.patch_dataloader import patch_dataloader
from dataloader.data_utils_eval import (load_MRI_from_paths_list,
                                        load_mask_from_paths_list,
                                        get_filelist_from_path,
                                        default_mask_generation,
                                        apply_mask_to_subject)
from network.VNet_model import VNetModel
#%%
parser = argparse.ArgumentParser(description='Generate synthetic 7T T1-weighted MRI from 3T input')

parser.add_argument('-i', '--input', '--in', type=str, help="""Path to the input file or directory containing input files.
                    Only processes .nii.gz images""")
parser.add_argument('-m', '--mask', type=str, default=None, help='path to mask file')
parser.add_argument('-c', '--ckpt', type=str, help='path to model file')
parser.add_argument('-o', '--output', '--out', type=str, help="""path to save file to""")

parser.add_argument('-bs', '--batch_size', type=int, default=5, help="""batch size for inference""")
parser.add_argument('-ps', '--patch_size', type=int, default=64, help="""patch size for inference""")
parser.add_argument('-po', '--patch_overlap', type=int, default=8, help="""patch overlap for inference""")
parser.add_argument('-ql', '--max_queue_length', type=int, default=1000, help="""max queue length for patch loading""")
parser.add_argument('-nw', '--num_workers', type=int, default=1, help="""number of workers for patch loading,
                    should not exceed number of CPU cores""")

#%%
args = parser.parse_args()

input_3T_path = args.input
input_mask_path = args.mask
ckpt_path = args.ckpt
output_path = args.output

params = {
    "data": {
            "batch_size": args.batch_size,
            "patch_size": args.patch_size,
            "patch_overlap": args.patch_overlap
    },
    "training":{ # this is a naming oversight, these are actually the parameters for the patched dataloader
                "num_workers": args.num_workers,
                "max_queue_length": args.max_queue_length
    }
}
params = Box(params)
print(params)
print(input_3T_path)
print(input_mask_path)
#%%