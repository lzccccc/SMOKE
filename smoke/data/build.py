import logging
import copy
import bisect
import numpy as np

import torch.utils.data

from smoke.utils.comm import get_world_size
from smoke.utils.imports import import_file
from smoke.utils.envs import seed_all_rng

from . import datasets as D
from . import samplers
from .transforms import build_transforms
from .collate_batch import BatchCollator


def build_dataset(cfg, transforms, dataset_catalog, is_train=True):
    '''
    Args:
        dataset_list (list[str]): Contains the names of the datasets.
        transforms (callable): transforms to apply to each (image, target) sample
        dataset_catalog (DatasetCatalog): contains the information on how to
            construct a dataset.
        is_train (bool): whether to setup the dataset for training or testing

    Returns:

    '''
    dataset_list = cfg.DATASETS.TRAIN if is_train else cfg.DATASETS.TEST
    if not isinstance(dataset_list, (list, tuple)):
        raise RuntimeError(
            "dataset_list should be a list of strings, got {}".format(dataset_list)
        )
    datasets = []
    for dataset_name in dataset_list:
        data = dataset_catalog.get(dataset_name)
        factory = getattr(D, data["factory"])
        args = data["args"]

        args["cfg"] = cfg
        args["is_train"] = is_train
        args["transforms"] = transforms
        # make dataset from factory
        dataset = factory(**args)
        datasets.append(dataset)

    # for testing, return a list of datasets
    if not is_train:
        return datasets

    # for training, concatenate all datasets into a single one
    dataset = datasets[0]
    if len(datasets) > 1:
        dataset = D.ConcatDataset(datasets)

    return [dataset]


def make_data_loader(cfg, is_train=True):
    num_gpus = get_world_size()
    if is_train:
        images_per_batch = cfg.SOLVER.IMS_PER_BATCH
        assert images_per_batch % num_gpus == 0, \
            "SOLVER.IMS_PER_BATCH ({}) must be divisible by the number of GPUs ({}) used." \
                .format(images_per_batch, num_gpus)

        images_per_gpu = images_per_batch // num_gpus
    else:
        images_per_batch = cfg.TEST.IMS_PER_BATCH
        assert images_per_batch % num_gpus == 0, \
            "SOLVER.IMS_PER_BATCH ({}) must be divisible by the number of GPUs ({}) used." \
                .format(images_per_batch, num_gpus)

        images_per_gpu = images_per_batch // num_gpus

    # if images_per_gpu > 1:
    #     logger = logging.getLogger(__name__)
    #     logger.warning(
    #         "When using more than one image per GPU you may encounter "
    #         "an out-of-memory (OOM) error if your GPU does not have "
    #         "sufficient memory. If this happens, you can reduce "
    #         "SOLVER.IMS_PER_BATCH (for training) or "
    #         "TEST.IMS_PER_BATCH (for inference). For training, you must "
    #         "also adjust the learning rate and schedule length according "
    #         "to the linear scaling rule. See for example: "
    #         "https://github.com/facebookresearch/Detectron/blob/master/configs/getting_started/tutorial_1gpu_e2e_faster_rcnn_R-50-FPN.yaml#L14"
    #     )

    # group images which have similar aspect ratio. In this case, we only
    # group in two cases: those with width / height > 1, and the other way around,
    # but the code supports more general grouping strategy
    aspect_grouping = [1] if cfg.DATALOADER.ASPECT_RATIO_GROUPING else []

    path_catalog = import_file(
        "smoke.config.paths_catalog", cfg.PATHS_CATALOG, True
    )
    DatasetCatalog = path_catalog.DatasetCatalog

    transforms = build_transforms(cfg, is_train)
    datasets = build_dataset(cfg, transforms, DatasetCatalog, is_train)

    data_loaders = []
    for dataset in datasets:
        sampler = samplers.TrainingSampler(len(dataset))
        batch_sampler = torch.utils.data.sampler.BatchSampler(
            sampler, images_per_gpu, drop_last=True
        )
        collator = BatchCollator(cfg.DATALOADER.SIZE_DIVISIBILITY)
        num_workers = cfg.DATALOADER.NUM_WORKERS
        data_loader = torch.utils.data.DataLoader(
            dataset,
            num_workers=num_workers,
            batch_sampler=batch_sampler,
            collate_fn=collator,
            worker_init_fn=worker_init_reset_seed,
        )
        data_loaders.append(data_loader)

    if is_train:
        # during training, a single (possibly concatenated) data_loader is returned
        assert len(data_loaders) == 1
        return data_loaders[0]
    return data_loaders


def build_test_loader(cfg, is_train=False):
    path_catalog = import_file(
        "smoke.config.paths_catalog", cfg.PATHS_CATALOG, True
    )
    DatasetCatalog = path_catalog.DatasetCatalog

    transforms = build_transforms(cfg, is_train)
    datasets = build_dataset(cfg, transforms, DatasetCatalog, is_train)

    data_loaders = []
    for dataset in datasets:
        sampler = samplers.InferenceSampler(len(dataset))
        batch_sampler = torch.utils.data.sampler.BatchSampler(
            sampler, 1, drop_last=False
        )
        collator = BatchCollator(cfg.DATALOADER.SIZE_DIVISIBILITY)
        num_workers = cfg.DATALOADER.NUM_WORKERS
        data_loader = torch.utils.data.DataLoader(
            dataset,
            num_workers=num_workers,
            batch_sampler=batch_sampler,
            collate_fn=collator,
        )
        data_loaders.append(data_loader)

    return data_loader


def trivial_batch_collator(batch):
    """
    A batch collator that does nothing.
    """
    return batch


def worker_init_reset_seed(worker_id):
    seed_all_rng(np.random.randint(2 ** 31) + worker_id)
