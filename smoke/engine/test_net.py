import os

from smoke.data import build_test_loader
from smoke.engine.inference import inference
from smoke.utils import comm
from smoke.utils.miscellaneous import mkdir


def run_test(cfg, model):
    eval_types = ("detection",)
    output_folders = [None] * len(cfg.DATASETS.TEST)
    dataset_names = cfg.DATASETS.TEST
    if cfg.OUTPUT_DIR:
        for idx, dataset_name in enumerate(dataset_names):
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
            mkdir(output_folder)
            output_folders[idx] = output_folder
    data_loaders_val = build_test_loader(cfg)
    for output_folder, dataset_name, data_loader_val in zip(output_folders, dataset_names, data_loaders_val):
        inference(
            model,
            data_loaders_val,
            dataset_name=dataset_name,
            eval_types=eval_types,
            device=cfg.MODEL.DEVICE,
            output_folder=output_folder,
        )
        comm.synchronize()
