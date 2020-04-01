from smoke.data import datasets

from .kitti.kitti_eval import kitti_evaluation


def evaluate(eval_type, dataset, predictions, output_folder):
    """evaluate dataset using different methods based on dataset type.
    Args:
        eval_type:
        dataset: Dataset object
        predictions(list[BoxList]): each item in the list represents the
            prediction results for one image.
        output_folder: output folder, to save evaluation files or results.
        **kwargs: other args.
    Returns:
        evaluation result
    """
    args = dict(
        eval_type=eval_type,
        dataset=dataset,
        predictions=predictions,
        output_folder=output_folder,

    )
    if isinstance(dataset, datasets.KITTIDataset):
        return kitti_evaluation(**args)
    else:
        dataset_name = dataset.__class__.__name__
        raise NotImplementedError("Unsupported dataset type {}.".format(dataset_name))
