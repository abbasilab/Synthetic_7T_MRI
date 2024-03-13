# load and split data
import numpy as np
import os.path as path
import torchio as tio
from sklearn.model_selection import KFold


def load_MRIs(fp, fns, postfix):
    fp3 = f'{fp}/3T'
    fp7 = f'{fp}/7T'
    fps7 = f'{fp}/7T_synthetic'
    fpm = f'{fp}/mask'
    flag3 = path.exists(fp3)
    flag7 = path.exists(fp7)
    flags7 = path.exists(fps7)
    flagm = path.exists(fpm)
    flag_original = 'original' in fp

    subjects = []
    for i in range(len(fns)):
        # normalize value between 0,1
        # outlier removal done during preprocessing (5%-95% percentile kept)
        rescale1 = tio.RescaleIntensity(out_min_max=(0, 1), percentiles=(0, 100))
        rescale2 = tio.RescaleIntensity(out_min_max=(0, 1), percentiles=(0, 100))

        subject_dict = {}
        if flag3:
            subject_dict['t1_3T'] = rescale1(tio.ScalarImage(path.join(fp3, str(fns[i]) + postfix)))

        if flag7:
            subject_dict['t1_7T'] = rescale2(tio.ScalarImage(path.join(fp7, str(fns[i]) + postfix)))

        if flags7:
            subject_dict['t1_s7T'] = tio.ScalarImage(path.join(fps7, str(fns[i]) + postfix))

        if flagm:
            subject_dict['mask'] = tio.LabelMap(path.join(fpm, str(fns[i]) + postfix))

        if flag_original:
            subject_dict['is_original'] = True
        else:
            subject_dict['is_original'] = False

        subject = tio.Subject(id=fns[i], **subject_dict)

        subjects.append(subject)
    # print('Dataset size:', len(subjects), 'subjects')
    return subjects


def get_fns(params):
    with open(params.fp.subjects, 'r') as fp:
        fns = [line.strip() for line in fp]
    return fns


def load_all_subjects(params, fns=None):
    # load all subjects with torchio
    if fns is None:
        fns = get_fns(params)
    filepaths = params.fp.filepaths
    postfixes = params.fp.postfixes

    subjects =[]
    for i in range(len(filepaths)):
        subjects.extend(load_MRIs(filepaths[i], fns, postfixes[i]))

    print('total number of subjects: ', len(subjects))
    return subjects


def kfold_split(params, fns=None, random_state=42):
    # kfold data splitting
    if fns is None:
        fns = get_fns(params)
    kf = KFold(n_splits=params.data.kfold_num,
               shuffle=True, random_state=random_state)

    train_inds = []
    val_inds = []
    fn_inds = np.arange(len(fns))

    for i, (train_index, val_index) in enumerate(kf.split(fn_inds)):
        train_inds.append(train_index)
        val_inds.append(val_index)
        print(f"Fold {i}:")
        print(f"  Train: index={train_index}")
        print(f"  Val:  index={val_index}")

    return train_inds, val_inds


def load_onefold_dataset(params, fold_ind):
    # get the trainig and validation dataset for one fold
    fns = get_fns(params)
    subjects = load_all_subjects(params, fns)
    train_inds, val_inds = kfold_split(params, fns)

    train_ind = [fns[a] for a in train_inds[fold_ind]]
    val_ind = [fns[a] for a in val_inds[fold_ind]]

    train_dataset = tio.SubjectsDataset(
        [s for s in subjects if s.id in train_ind])
    val_dataset = tio.SubjectsDataset(
        [s for s in subjects if s.id in val_ind and s.is_original])

    print(f"Fold {fold_ind}:")
    print("Train indices:", train_ind)
    print("Validation indices:", val_ind)
    print()

    return train_dataset, val_dataset


def load_all(params):
    # get the trainig and validation dataset for final model, no leave-out
    fns = get_fns(params)
    subjects = load_all_subjects(params, fns)
    train_dataset = tio.SubjectsDataset(subjects)

    print(f"using all data for training")
    print()

    return train_dataset