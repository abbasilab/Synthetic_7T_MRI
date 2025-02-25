# load and split data
import os
import torchio as tio
import numpy as np


def get_filelist_from_path(arg_path):
    if not os.path.exists(arg_path):
        raise ValueError('Input path does not exist')

    if os.path.isdir(arg_path):
        # input is dir, get all .nii.gz files in input dir
        input_files = [os.path.join(arg_path, f) for f in os.listdir(arg_path) if f.endswith('.nii.gz')]
        if not input_files:
            raise ValueError('No .nii.gz files found in input directory')

    elif os.path.isfile(arg_path):
        # input is file, check if it is .nii.gz
        if not arg_path.endswith('.nii.gz'):
            raise ValueError('Input file is not a .nii.gz file')
        input_files = [arg_path]
    return input_files


def load_MRI_from_paths_list(input_files, image_modality):
    # load all MRIs from a list of filepaths
    # input_files = [fp1, fp2, ...]
    subjects = []
    for i in range(len(input_files)):
        rescale = tio.RescaleIntensity(out_min_max=(0, 1), percentiles=(0, 100))
        subject_dict = {}
        subject_dict[image_modality] = rescale(tio.ScalarImage(input_files[i]))
        subject_name = os.path.basename(input_files[i])[:-7]  # remove file extension
        subject = tio.Subject(id=subject_name, **subject_dict)
        subjects.append(subject)
    return subjects


def load_mask_from_paths_list(input_files):
    # load all masks from a list of filepaths
    # input_files = [fp1, fp2, ...]
    subjects = []
    for i in range(len(input_files)):
        subject_dict = {}
        subject_dict['mask'] = tio.LabelMap(input_files[i])
        subject_name = os.path.basename(input_files[i])[:-7]  # remove file extension
        subject = tio.Subject(id=subject_name, **subject_dict)
        subjects.append(subject)
    return subjects


def default_mask_generation(subject_data):
    # If no mask is provided, generate a default binary mask
    # by thresholding the image of the subject at 0
    subject_keys = list(subject_data.keys())
    subject_keys.remove('id')
    img_key = subject_keys[0]
    subject_img = subject_data[img_key]
    mask_data = (subject_img.data !=0).float()
    subject_mask = tio.LabelMap(tensor=mask_data, affine=subject_img.affine)
    subject_mask = tio.Subject(id=subject_data.id, mask=subject_mask)
    return subject_mask


def apply_mask_to_subject(subjects_t1_3T, subjects_mask=None):
    subject_attribute_combined = {}

    for subject in subjects_t1_3T:
        subject_keys = list(subject.keys())
        subject_keys.remove('id')
        img_key = subject_keys[0]
        subject_attribute_combined[subject.id] = {'id': subject.id,
                                                  img_key: subject[img_key]}

    if subjects_mask is not None:
        for subject in subjects_mask:
            subject_keys = list(subject.keys())
            subject_keys.remove('id')
            mask_key = subject_keys[0]
            if subject.id in subject_attribute_combined:
                # If the subject exists, add mask as attribute
                subject_pin = subject_attribute_combined[subject.id]
                subject_pin[mask_key] = subject[mask_key]
                # apply mask to image
                subject_pin[img_key] = tio.ScalarImage(tensor=subject_pin[mask_key].data * subject_pin[img_key].data,
                                                       affine=subject_pin[img_key].affine)
            # # check if mask is in same affine space as image
            # if np.array_equal(subject[img_key].affine, subject[mask_key].affine):
            #     print(f'For subject {raw_subject.id}, mask and image are not in the same affine space, resample mask to image space')
            #     resampler = tio.Resample(subject[img_key])
            #     resampled_mask = resampler(subject[mask_key])
            #     print(subject[img_key].affine)
            #     subject_attribute_combined[raw_subject.id][mask_key] = resampled_mask

    subjects = []
    for subject_data in subject_attribute_combined.values():
        subjects.append(tio.Subject(subject_data))

    return subjects
