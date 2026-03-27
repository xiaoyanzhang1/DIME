import os
import copy
from collections import OrderedDict

import numpy as np
import torch


def count_parameters(model, trainable: bool = False) -> int:
    if trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def tensor2numpy(x: torch.Tensor) -> np.ndarray:
    return x.cpu().data.numpy() if x.is_cuda else x.data.numpy()


def target2onehot(targets: torch.Tensor, n_classes: int) -> torch.Tensor:
    onehot = torch.zeros(targets.shape[0], n_classes, device=targets.device)
    onehot.scatter_(dim=1, index=targets.long().view(-1, 1), value=1.0)
    return onehot


def makedirs(path: str):
    if not os.path.exists(path):
        os.makedirs(path)


def accuracy(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    nb_old: int,
    init_cls: int = 10,
    increments=None,
):
    assert len(y_pred) == len(y_true), "Data length error."
    all_acc = {}

    # Total accuracy
    all_acc["total"] = np.around(
        (y_pred == y_true).sum() * 100 / len(y_true), decimals=2
    )

    # Initial classes
    idxes = np.where((y_true >= 0) & (y_true < init_cls))[0]
    label = "{}-{}".format(str(0).rjust(2, "0"), str(init_cls - 1).rjust(2, "0"))
    all_acc[label] = (
        0.0
        if len(idxes) == 0
        else np.around(
            (y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), 2
        )
    )

    # Incremental class ranges
    if increments is not None:
        start_cls = init_cls
        for inc in increments[1:]:
            end_cls = start_cls + inc
            idxes = np.where((y_true >= start_cls) & (y_true < end_cls))[0]
            label = "{}-{}".format(
                str(start_cls).rjust(2, "0"), str(end_cls - 1).rjust(2, "0")
            )
            all_acc[label] = (
                0.0
                if len(idxes) == 0
                else np.around(
                    (y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), 2
                )
            )
            start_cls = end_cls

    # Old classes
    idxes = np.where(y_true < nb_old)[0]
    all_acc["old"] = (
        0.0
        if len(idxes) == 0
        else np.around(
            (y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), 2
        )
    )

    # New classes
    idxes = np.where(y_true >= nb_old)[0]
    all_acc["new"] = (
        0.0
        if len(idxes) == 0
        else np.around(
            (y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), 2
        )
    )

    return all_acc


def weighted_avg_normalized(class_per_task, acc_per_task):
    """
    Normalized weighted average of task accuracies (wĀ) for class-imbalanced streams.

    Args:
        class_per_task: List[int], number of classes per task.
        acc_per_task:   List[float], top-1 accuracy A_t for each task.

    Returns:
        Normalized weighted average accuracy.
    """
    T = min(len(class_per_task), len(acc_per_task))
    if T == 0:
        return float("nan")

    C = sum(class_per_task[:T])
    cum = 0
    num = 0.0
    den = 0.0

    for t in range(1, T + 1):
        cum += class_per_task[t - 1]
        # Normalized weight that corrects for varying task sizes
        w_t = cum / ((C * t) / T)
        num += w_t * acc_per_task[t - 1]
        den += w_t

    return num / den if den > 0 else float("nan")


def split_images_labels(imgs):
    images = []
    labels = []
    for item in imgs:
        images.append(item[0])
        labels.append(item[1])

    return np.array(images), np.array(labels)


def state_dict_to_vector(state_dict, remove_keys=None) -> torch.Tensor:
    if remove_keys is None:
        remove_keys = []

    shared_state_dict = copy.deepcopy(state_dict)
    shared_state_dict_keys = list(shared_state_dict.keys())

    for key in remove_keys:
        for _key in shared_state_dict_keys:
            if key in _key and _key in shared_state_dict:
                del shared_state_dict[_key]

    sorted_shared_state_dict = OrderedDict(sorted(shared_state_dict.items()))
    return torch.nn.utils.parameters_to_vector(
        [value.reshape(-1) for _, value in sorted_shared_state_dict.items()]
    )


def vector_to_state_dict(vector, state_dict, remove_keys=None):
    if remove_keys is None:
        remove_keys = []

    reference_dict = copy.deepcopy(state_dict)
    reference_dict_keys = list(reference_dict.keys())
    for key in remove_keys:
        for _key in reference_dict_keys:
            if key in _key and _key in reference_dict:
                del reference_dict[_key]

    sorted_reference_dict = OrderedDict(sorted(reference_dict.items()))
    torch.nn.utils.vector_to_parameters(vector, sorted_reference_dict.values())
    return sorted_reference_dict


def read_images_labels(lines):
    images = []
    labels = []
    for item in lines:
        item = item.rstrip()
        if not item:
            continue
        path, label = item.split(" ")
        images.append(path)
        labels.append(int(label))
    return np.array(images), np.array(labels)


def read_images_labels_imageneta(lines):
    images = []
    labels = []
    for item in lines:
        item = item.rstrip()
        if not item:
            continue
        path, label = item.rsplit(" ", 1)
        images.append(path)
        labels.append(int(label))
    return np.array(images), np.array(labels)


def read_images_labels_vfn(imgs):
    images = []
    labels = []
    for item in imgs:
        item = item.rstrip()
        if not item:
            continue

        path, label = item.rsplit(' ', 1)

        path = path.replace(' ', '_')

        images.append(path)
        labels.append(int(label))

    return np.array(images), np.array(labels)





