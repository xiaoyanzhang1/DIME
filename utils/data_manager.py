import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from utils.data import iCIFAR224, iImageNetR, iImageNetA, CUB


def get_exp_imbalance(
    cls_num: int,
    img_total: int,
    imb_factor: float,
    min_per_cls: int = 1,
):
    """
    Generate an exponentially imbalanced class allocation.

    Args:
        cls_num: Number of classes (tasks).
        img_total: Total number (budget) to allocate.
        imb_factor: Exponential imbalance factor.
        min_per_cls: Minimum samples per class.

    Returns:
        List[int]: Number of samples per class.
    """
    # Step 1: generate un-normalized exponential ratios
    img_num_per_cls = []
    for cls_idx in range(cls_num):
        ratio = imb_factor ** (cls_idx / (cls_num - 1.0))
        img_num_per_cls.append(ratio)

    img_num_per_cls = np.array(img_num_per_cls)

    # Step 2: normalize and scale to total remaining budget
    img_num_per_cls = img_num_per_cls / img_num_per_cls.sum()

    # Step 3: allocate min_per_cls first
    base_allocation = np.full(cls_num, min_per_cls)
    remaining = img_total - cls_num * min_per_cls
    if remaining < 0:
        raise ValueError(
            "img_total is too small to guarantee at least min_per_cls for each class."
        )

    img_num_per_cls = np.round(img_num_per_cls * remaining).astype(int)

    # Step 4: fix rounding error to exactly match `remaining`
    diff = remaining - img_num_per_cls.sum()
    if diff != 0:
        # Prefer to adjust large classes first
        indices = np.argsort(-img_num_per_cls)
        for i in range(abs(diff)):
            img_num_per_cls[indices[i % cls_num]] += np.sign(diff)

    # Step 5: add the minimum per class back
    img_num_per_cls += base_allocation

    return img_num_per_cls.tolist()


class DataManager(object):
    def __init__(
        self,
        dataset_name,
        shuffle,
        seed,
        # init_cls=None,
        # increment=None,
        args=None,
    ):
        self.args = args
        self.dataset_name = dataset_name
        self._setup_data(dataset_name, shuffle, seed)


        if not args["use_imbalance_setting"]:
            init_cls = self.args["init_cls"]
            increment = self.args["increment"]
            assert init_cls <= len(self._class_order), "Not enough classes."
            # self._increments = [init_cls]
            self._increments = [self.args["init_cls"]]
            while sum(self._increments) + increment < len(self._class_order):
                self._increments.append(increment)
            offset = len(self._class_order) - sum(self._increments)
            if offset > 0:
                self._increments.append(offset)
        else:
            total_classes = len(self._class_order)
            nb_tasks = args["nb_tasks"]
            assert total_classes >= nb_tasks, "Too few classes for the number of tasks."

            class_per_task = get_exp_imbalance(
                cls_num=nb_tasks,
                img_total=total_classes,
                imb_factor=self.args["task_imb_factor"],
            )

            # Control class-per-task ordering
            order = args.get("imbalance_order", "shuffle")
            if order == "desc":
                class_per_task.sort(reverse=True)
            elif order == "shuffle":
                np.random.seed(args["seed"])
                np.random.shuffle(class_per_task)
            else:
                # unknown: keep as is
                pass

            self._increments = class_per_task
            assert (
                sum(self._increments) == total_classes
            ), "Mismatch in class allocation."

            args["class_per_task"] = class_per_task
            args["logger"].info("Class number per task: {}".format(class_per_task))

    @property
    def nb_tasks(self):
        return len(self._increments)

    def get_task_size(self, task: int) -> int:
        return self._increments[task]

    @property
    def nb_classes(self):
        return len(self._class_order)

    def get_dataset(
        self,
        indices,
        source,
        mode,
        appendent=None,
        ret_data: bool = False,
        m_rate=None,
    ):
        if source == "train":
            x, y = self._train_data, self._train_targets
        elif source == "test":
            x, y = self._test_data, self._test_targets
        else:
            raise ValueError("Unknown data source {}.".format(source))

        if mode == "train":
            trsf = transforms.Compose([*self._train_trsf, *self._common_trsf])
        elif mode == "flip":
            trsf = transforms.Compose(
                [
                    *self._test_trsf,
                    transforms.RandomHorizontalFlip(p=1.0),
                    *self._common_trsf,
                ]
            )
        elif mode == "test":
            trsf = transforms.Compose([*self._test_trsf, *self._common_trsf])
        else:
            raise ValueError("Unknown mode {}.".format(mode))

        data, targets = [], []
        for idx in indices:
            if m_rate is None:
                class_data, class_targets = self._select(
                    x, y, low_range=idx, high_range=idx + 1
                )
            else:
                class_data, class_targets = self._select_rmm(
                    x, y, low_range=idx, high_range=idx + 1, m_rate=m_rate
                )
            data.append(class_data)
            targets.append(class_targets)

        if appendent is not None and len(appendent) != 0:
            appendent_data, appendent_targets = appendent
            data.append(appendent_data)
            targets.append(appendent_targets)

        data, targets = np.concatenate(data), np.concatenate(targets)

        if ret_data:
            return data, targets, DummyDataset(data, targets, trsf, self.use_path)
        else:
            return DummyDataset(data, targets, trsf, self.use_path)

    def get_dataset_with_split(
        self,
        indices,
        source,
        mode,
        appendent=None,
        val_samples_per_class: int = 0,
    ):
        if source == "train":
            x, y = self._train_data, self._train_targets
        elif source == "test":
            x, y = self._test_data, self._test_targets
        else:
            raise ValueError("Unknown data source {}.".format(source))

        if mode == "train":
            trsf = transforms.Compose([*self._train_trsf, *self._common_trsf])
        elif mode == "test":
            trsf = transforms.Compose([*self._test_trsf, *self._common_trsf])
        else:
            raise ValueError("Unknown mode {}.".format(mode))

        train_data, train_targets = [], []
        val_data, val_targets = [], []

        # Per-class split
        for idx in indices:
            class_data, class_targets = self._select(
                x, y, low_range=idx, high_range=idx + 1
            )
            val_indx = np.random.choice(
                len(class_data), val_samples_per_class, replace=False
            )
            train_indx = list(set(np.arange(len(class_data))) - set(val_indx))
            val_data.append(class_data[val_indx])
            val_targets.append(class_targets[val_indx])
            train_data.append(class_data[train_indx])
            train_targets.append(class_targets[train_indx])

        # Optional appendent data split
        if appendent is not None:
            appendent_data, appendent_targets = appendent
            for idx in range(0, int(np.max(appendent_targets)) + 1):
                append_data, append_targets = self._select(
                    appendent_data,
                    appendent_targets,
                    low_range=idx,
                    high_range=idx + 1,
                )
                val_indx = np.random.choice(
                    len(append_data), val_samples_per_class, replace=False
                )
                train_indx = list(set(np.arange(len(append_data))) - set(val_indx))
                val_data.append(append_data[val_indx])
                val_targets.append(append_targets[val_indx])
                train_data.append(append_data[train_indx])
                train_targets.append(append_targets[train_indx])

        train_data, train_targets = np.concatenate(train_data), np.concatenate(
            train_targets
        )
        val_data, val_targets = np.concatenate(val_data), np.concatenate(val_targets)

        return (
            DummyDataset(train_data, train_targets, trsf, self.use_path),
            DummyDataset(val_data, val_targets, trsf, self.use_path),
        )

    def _setup_data(self, dataset_name, shuffle: bool, seed: int) -> None:
        """
        Initialize raw data, transforms, and class order (possibly shuffled).
        """
        idata = _get_idata(dataset_name, self.args)
        idata.download_data()

        # Data
        self._train_data, self._train_targets = idata.train_data, idata.train_targets
        self._test_data, self._test_targets = idata.test_data, idata.test_targets
        self.use_path = idata.use_path

        # Transforms
        self._train_trsf = idata.train_trsf
        self._test_trsf = idata.test_trsf
        self._common_trsf = idata.common_trsf

        # Class order
        order = [i for i in range(len(np.unique(self._train_targets)))]
        if shuffle:
            np.random.seed(seed)
            order = np.random.permutation(len(order)).tolist()
        else:
            order = idata.class_order
        self._class_order = order
        self.args["logger"].info(self._class_order)

        # Map targets to incremental class indices
        self._train_targets = _map_new_class_index(
            self._train_targets, self._class_order
        )
        self._test_targets = _map_new_class_index(
            self._test_targets, self._class_order
        )

    def _select(self, x, y, low_range, high_range):
        idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]
        return x[idxes], y[idxes]

    def _select_rmm(self, x, y, low_range, high_range, m_rate: float):
        """
        Random sample reduction per class with a missing rate m_rate.
        """
        assert m_rate is not None
        if m_rate != 0:
            idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]
            selected_idxes = np.random.randint(
                0, len(idxes), size=int((1 - m_rate) * len(idxes))
            )
            new_idxes = idxes[selected_idxes]
            new_idxes = np.sort(new_idxes)
        else:
            new_idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]
        return x[new_idxes], y[new_idxes]

    def getlen(self, index: int):
        """
        Return a statistic for a given class index.
        (Note: this uses np.sum(np.where(...)), which may not be a pure count.)
        """
        y = self._train_targets
        return np.sum(np.where(y == index))


class DummyDataset(Dataset):
    """
    Simple dataset wrapper for numpy arrays + torchvision transforms.
    """

    def __init__(self, images, labels, trsf, use_path: bool = False):
        assert len(images) == len(labels), "Data size mismatch!"
        self.images = images
        self.labels = labels
        self.trsf = trsf
        self.use_path = use_path

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if self.use_path:
            image = self.trsf(pil_loader(self.images[idx]))
        else:
            image = self.trsf(Image.fromarray(self.images[idx]))
        label = self.labels[idx]
        return idx, image, label


def _map_new_class_index(y, order):
    return np.array(list(map(lambda x: order.index(x), y)))


def _get_idata(dataset_name, args=None):
    name = dataset_name.lower()
    if name == "cifar224":
        return iCIFAR224(args)
    elif name == "imagenetr":
        return iImageNetR(args)
    elif name == "imageneta":
        return iImageNetA()
    elif name == "cub":
        return CUB()
    else:
        raise NotImplementedError("Unknown dataset {}.".format(dataset_name))


def pil_loader(path):
    """
    Load image with PIL (always converted to RGB).

    Reference:
        https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#ImageFolder
    """
    # Open path as file to avoid ResourceWarning
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


def accimage_loader(path):
    """
    Load image with accimage, falling back to PIL on decode error.

    Reference:
        https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#ImageFolder
    """
    import accimage

    try:
        return accimage.Image(path)
    except IOError:
        # Fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    """
    Default image loader that switches between accimage and PIL.
    """
    from torchvision import get_image_backend

    if get_image_backend() == "accimage":
        return accimage_loader(path)
    else:
        return pil_loader(path)
