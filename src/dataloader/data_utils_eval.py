# load and split data
import numpy as np
import os.path as path
import torchio as tio


def load_MRI_from_paths_list(input_files, image_modality):
    # load all MRIs from a list of filepaths
    # input_files = [fp1, fp2, ...]
    subjects = []
    for i in range(len(input_files)):
        rescale = tio.RescaleIntensity(out_min_max=(0, 1), percentiles=(0, 100))
        subject_dict = {}
        subject_dict[image_modality] = rescale(tio.ScalarImage(input_files[i]))
        subject_name = path.basename(input_files[i])[:-7]  # remove file extension
        subject = tio.Subject(id=subject_name, **subject_dict)
        subjects.append(subject)
    return subjects
