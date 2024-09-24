# pylint: disable=too-many-function-args, invalid-name
""" BSNIP ICA dataset loading script"""
import numpy as np
import pandas as pd

from omegaconf import DictConfig

from src.settings import DATA_ROOT


def load_data(
    cfg: DictConfig,
    dataset_path: str = DATA_ROOT.joinpath("bsnip/BSNIP_data.npz"),
    indices_path: str = DATA_ROOT.joinpath("ICA_correct_order.csv"),
):
    """
    Return BSNIP data

    Input:
    dataset_path: str = DATA_ROOT.joinpath("bsnip/BSNIP_data.npz")
    - path to the dataset with lablels
    indices_path: str = DATA_ROOT.joinpath("bsnip/correct_indices_GSP.csv")
    - path to correct indices/components

    Output:
    features, labels
    """

    with np.load(dataset_path) as npzfile:
        data = npzfile["features"]
        labels = npzfile["labels"]

    if cfg.dataset.filter_indices:
        # get correct indices/components
        indices = pd.read_csv(indices_path, header=None)
        idx = indices[0].values - 1
        data = data[:, idx, :]

    multiclass = False
    if "multiclass" in cfg.dataset:
        multiclass = cfg.dataset.multiclass

    invert_classes = True
    if "invert_classes" in cfg.dataset:
        invert_classes = cfg.dataset.invert_classes

    filter_array = []
    if multiclass:
        unique, counts = np.unique(labels, return_counts=True)
        counts = dict(zip(unique, counts))

        print(f"Number of classes in the data: {unique.shape[0]}")
        valid_labels = []
        for label, count in counts.items():
            if count > 10:
                valid_labels += [label]
            else:
                print(
                    f"There is not enough labels '{label}' in the dataset, filtering them out"
                )

        if len(valid_labels) == unique.shape[0]:
            filter_array = [True] * labels.shape[0]
        else:
            for label in labels:
                if label in valid_labels:
                    filter_array.append(True)
                else:
                    filter_array.append(False)
    else:
        # leave subjects of class 0 and 1 only
        # {"NC": 0, "SZ": 1, "SAD": 2, "BP": 3, "BPnon": 4, "OTH": 5}
        for label in labels:
            if label in (0, 1):
                filter_array.append(True)
            else:
                filter_array.append(False)

    data = data[filter_array, :, :]
    labels = labels[filter_array]

    if not multiclass and invert_classes:
        new_labels = []
        for label in labels:
            if label == 0:
                new_labels.append(1)
            else:
                new_labels.append(0)

        labels = np.array(new_labels)

    unique = np.sort(np.unique(labels))
    shift_dict = dict(zip(unique, np.arange(unique.shape[0])))
    for i, _ in enumerate(labels):
        labels[i] = shift_dict[labels[i]]

    data = np.swapaxes(data, 1, 2)
    # data.shape = [n_samples, time_length, feature_size]

    return data, labels
