import sys
import logging
import copy
import os
import datetime

import numpy as np
import torch

from utils import factory
from utils.data_manager import DataManager
from utils.toolkit import count_parameters, weighted_avg_normalized


def train(args: dict) -> None:
    """
    Run training for each random seed in args["seed"].

    Args:
        args: Experiment configuration dictionary. Must contain:
              - "seed": list of random seeds
              - "device": list of device IDs (e.g., [0] or [0, 1])
    """
    seed_list = copy.deepcopy(args["seed"])
    device = copy.deepcopy(args["device"])

    for seed in seed_list:
        args["seed"] = seed
        args["device"] = device
        _train(args)


def _train(args: dict) -> None:
    """
    Main training loop for a single random seed.
    Sets up logging, data manager, model, and runs task-by-task training.
    """
    use_longtail = "lt" in args["dataset"].lower()
    data_type = "LT" if use_longtail or args.get("dataset") == "inat" else "Original"

    # --------------------------
    # Log directory and filename
    # --------------------------
    if not args["use_imbalance_setting"]:
        init_cls = 0 if args["init_cls"] == args["increment"] else args["init_cls"]
        logs_dir = "logs/{}/{}/{}/{}/{}".format(
            data_type,
            args["model_name"],
            args["dataset"],
            init_cls,
            args["increment"],
        )
    elif "task_imb_factor" in args:
        logs_dir = "logs_SI/{}/{}/{}/{}/{}".format(
            args["imbalance_order"],
            args["dataset"],
            args["task_imb_factor"],
            args["nb_tasks"],
            args["model_name"],
        )
    

    os.makedirs(logs_dir, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    if not args["use_imbalance_setting"]:
        init_cls = 0 if args["init_cls"] == args["increment"] else args["init_cls"]
        logfilename = "logs/{}/{}/{}/{}/{}/{}_BS{}_{}_{}_T{}".format(
            data_type,
            args["model_name"],
            args["dataset"],
            init_cls,
            args["increment"],
            args["prefix"],
            args["batch_size"],
            args["seed"],
            args["backbone_type"],
            timestamp,
        )
    elif "task_imb_factor" in args:
        logfilename = "logs_SI/{}/{}/{}/{}/{}/{}_BS{}_{}_{}_T{}".format(
            args["imbalance_order"],
            args["dataset"],
            args["task_imb_factor"],
            args["nb_tasks"],
            args["model_name"],
            args["prefix"],
            args["batch_size"],
            args["seed"],
            args["backbone_type"],
            timestamp,
        )
    # else:
    #     logfilename = "logs_my/{}/{}/{}/{}/{}/{}_BS{}_{}_{}_T{}".format(
    #         data_type,
    #         args["model_name"],
    #         args["dataset"],
    #         args["imbalance_order"],
    #         args["nb_tasks"],
    #         args["prefix"],
    #         args["batch_size"],
    #         args["seed"],
    #         args["backbone_type"],
    #         timestamp,
    #     )

    # --------------------------
    # Logger setup
    # --------------------------
    logger = logging.getLogger(f'seed_{args["seed"]}')
    logger.setLevel(logging.INFO)

    if logger.hasHandlers():
        logger.handlers.clear()

    file_handler = logging.FileHandler(filename=logfilename + ".log")
    stream_handler = logging.StreamHandler(sys.stdout)

    formatter = logging.Formatter("%(asctime)s [%(filename)s] => %(message)s")
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    # --------------------------
    # Reproducibility & device
    # --------------------------
    _set_random(args["seed"])
    _set_device(args)
    print_args(args, logger)
    args["logger"] = logger

    # --------------------------
    # Data & model
    # --------------------------
    data_manager = DataManager(
        args["dataset"],
        args["shuffle"],
        args["seed"],
        # args["init_cls"],
        # args["increment"],
        args,
    )

    # Update args with dataset-specific information
    args["nb_classes"] = data_manager.nb_classes
    args["nb_tasks"] = data_manager.nb_tasks

    if not args["use_imbalance_setting"]:
        args["cls_per_task"] = [args["init_cls"]] + [args["increment"]] * (
            args["nb_tasks"] - 1
        )
    else:
        args["cls_per_task"] = args["class_per_task"]

    model = factory.get_model(args["model_name"], args)

    cnn_curve, nme_curve = {"top1": [], "top5": []}, {"top1": [], "top5": []}
    cnn_matrix, nme_matrix = [], []

    # --------------------------
    # Task-by-task training loop
    # --------------------------
    for task in range(data_manager.nb_tasks):
        logger.info("All params: {}".format(count_parameters(model._network)))
        logger.info(
            "Trainable params: {}".format(count_parameters(model._network, trainable=True))
        )

        model.incremental_train(data_manager)
        cnn_accy, nme_accy = model.eval_task()

        cnn_accy = convert_all_np_float(cnn_accy)
        nme_accy = convert_all_np_float(nme_accy)
        model.after_task()

        if nme_accy is not None:
            # CNN and NME grouped accuracies
            logger.info("CNN: {}".format(cnn_accy["grouped"]))
            logger.info("NME: {}".format(nme_accy["grouped"]))

            cnn_keys = [k for k in cnn_accy["grouped"].keys() if "-" in k]
            cnn_values = [cnn_accy["grouped"][k] for k in cnn_keys]
            cnn_matrix.append(cnn_values)

            nme_keys = [k for k in nme_accy["grouped"].keys() if "-" in k]
            nme_values = [nme_accy["grouped"][k] for k in nme_keys]
            nme_matrix.append(nme_values)

            # Curves
            cnn_curve["top1"].append(cnn_accy["top1"])
            nme_curve["top1"].append(nme_accy["top1"])

            logger.info("CNN top1 curve: {}".format(cnn_curve["top1"]))
            logger.info("NME top1 curve: {}".format(nme_curve["top1"]))

            print("Average Accuracy (CNN):", sum(cnn_curve["top1"]) / len(cnn_curve["top1"]))
            print("Average Accuracy (NME):", sum(nme_curve["top1"]) / len(nme_curve["top1"]))

            logger.info(
                "Average Accuracy (CNN): {}".format(
                    sum(cnn_curve["top1"]) / len(cnn_curve["top1"])
                )
            )
            logger.info(
                "Average Accuracy (NME): {}".format(
                    sum(nme_curve["top1"]) / len(nme_curve["top1"])
                )
            )

            _log_weighted_metrics(
                logger=logger,
                class_per_task=args["cls_per_task"],
                acc_per_task=cnn_curve["top1"],
                prefix="CNN",
            )
        else:
            # Only CNN accuracy is available
            logger.info("No NME accuracy.")
            logger.info("CNN: {}".format(cnn_accy["grouped"]))

            cnn_keys = [k for k in cnn_accy["grouped"].keys() if "-" in k]
            cnn_values = [cnn_accy["grouped"][k] for k in cnn_keys]
            cnn_matrix.append(cnn_values)

            cnn_curve["top1"].append(cnn_accy["top1"])

            logger.info("CNN top1 curve: {}".format(cnn_curve["top1"]))
            print("Average Accuracy (CNN):", sum(cnn_curve["top1"]) / len(cnn_curve["top1"]))
            logger.info(
                "Average Accuracy (CNN): {}".format(
                    sum(cnn_curve["top1"]) / len(cnn_curve["top1"])
                )
            )

            _log_weighted_metrics(
                logger=logger,
                class_per_task=args["cls_per_task"],
                acc_per_task=cnn_curve["top1"],
                prefix="CNN",
            )

    # --------------------------
    # Forgetting metrics
    # --------------------------
    if args.get("print_forget", False):
        if len(cnn_matrix) > 0:
            np_acctable = np.zeros([task + 1, task + 1])
            for idx_row, line in enumerate(cnn_matrix):
                idx_col = len(line)
                np_acctable[idx_row, :idx_col] = np.array(line)
            np_acctable = np_acctable.T
            forgetting = np.mean(
                (np.max(np_acctable, axis=1) - np_acctable[:, task])[:task]
            )
            print("Accuracy Matrix (CNN):")
            print(np_acctable)
            logger.info("Forgetting (CNN): {}".format(forgetting))

        if len(nme_matrix) > 0:
            np_acctable = np.zeros([task + 1, task + 1])
            for idx_row, line in enumerate(nme_matrix):
                idx_col = len(line)
                np_acctable[idx_row, :idx_col] = np.array(line)
            np_acctable = np_acctable.T
            forgetting = np.mean(
                (np.max(np_acctable, axis=1) - np_acctable[:, task])[:task]
            )
            print("Accuracy Matrix (NME):")
            print(np_acctable)
            logger.info("Forgetting (NME): {}".format(forgetting))


def _log_weighted_metrics(
    logger: logging.Logger, class_per_task, acc_per_task, prefix: str = "CNN"
) -> None:
    """
    Log weighted normalized average accuracy, plain average accuracy, and last-task accuracy.

    Args:
        logger: Logger instance.
        class_per_task: List[int], number of classes per task.
        acc_per_task: List[float], top-1 accuracy per task.
        prefix: String used to indicate which classifier (e.g., "CNN", "NME").
    """
    try:
        if len(class_per_task) != len(acc_per_task):
            T = min(len(class_per_task), len(acc_per_task))
            class_per_task = class_per_task[:T]
            acc_per_task = acc_per_task[:T]

        w_avg = weighted_avg_normalized(class_per_task, acc_per_task)
        avg = sum(acc_per_task) / len(acc_per_task)
        a_t = acc_per_task[-1]

        logger.info(
            "%s metrics -> wA_norm: %.6f | A_bar: %.6f | A_T: %.6f",
            prefix,
            w_avg,
            avg,
            a_t,
        )
    except Exception as e:  # pragma: no cover (defensive logging)
        logger.exception("Failed to compute weighted metrics (%s): %s", prefix, e)


def _set_device(args: dict) -> None:
    """
    Convert a list of device IDs to torch.device objects and store back in args["device"].
    """
    devices = []
    for dev in args["device"]:
        if dev == -1:
            devices.append(torch.device("cpu"))
        else:
            devices.append(torch.device(f"cuda:{dev}"))
    args["device"] = devices


def _set_random(seed: int = 1) -> None:
    """
    Set random seeds for reproducibility.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_args(args: dict, logger: logging.Logger) -> None:
    """
    Log all key-value pairs in the args dictionary.
    """
    for key, value in args.items():
        logger.info("%s: %s", key, value)


def convert_all_np_float(d):
    """
    Convert numpy scalar values in a (shallow) dict to Python floats.

    Args:
        d: Dictionary or other object.

    Returns:
        A new dict with np.generic values converted to float, or the original object.
    """
    if isinstance(d, dict):
        return {k: float(v) if isinstance(v, np.generic) else v for k, v in d.items()}
    return d
