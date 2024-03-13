import torch
import torchio as tio


def patch_dataloader(dataset, params):
    # load image patches for training subjects and validation subjects
    patch_size = params.data.patch_size
    patch_overlap = params.data.patch_overlap
    max_queue_length = params.training.max_queue_length
    num_workers = params.training.num_workers

    # since datasize is uniform across subjects, only need one subject
    sampler = tio.data.GridSampler(subject=dataset[0],
                                   patch_size=patch_size,
                                   patch_overlap=patch_overlap)
    num_patches = len(sampler)
    # print(num_patches)
    aggregator = tio.inference.GridAggregator(sampler, 'hann')

    patches = tio.Queue(
        subjects_dataset=dataset,
        max_length=max_queue_length,  # extract all patches from each volume
        samples_per_volume=num_patches,
        sampler=sampler,
        num_workers=num_workers,
        shuffle_patches=True,
    )

    # print(len(patches))
    patch_loader = torch.utils.data.DataLoader(
        patches, batch_size=params.data.batch_size, pin_memory=True)

    return patch_loader, aggregator
