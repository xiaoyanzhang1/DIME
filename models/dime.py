import copy
import math
import random

import numpy as np
import torch
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.base import BaseLearner
from utils.inc_net import Adapter_merge
from utils.toolkit import tensor2numpy

num_workers = 8


"""
    1. 不用contrasive loss + CE loss，用balanced softmax
    2. hard
    3. 直接knots svd
"""





def seed_all(s: int) -> None:
    """
    Set global random seeds for reproducibility.
    """
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_worker_init_fn(base_seed: int):
    """
    Create a worker_init_fn for DataLoader that sets per-worker seeds.
    """

    def _seed_worker(worker_id: int):
        worker_seed = (base_seed + worker_id) % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    return _seed_worker


class Learner(BaseLearner):
    """
    DIME learner with adapter-based incremental training and SVD-based
    adapter alignment / merging.
    """

    def __init__(self, args):
        super().__init__(args)

        self.args = args
        self.batch_size = args["batch_size"]
        self.init_lr = args["init_lr"]
        self.weight_decay = (
            args["weight_decay"] if args["weight_decay"] is not None else 5e-4
        )
        self.min_lr = args["min_lr"] if args["min_lr"] is not None else 1e-8

        self.init_cls = args["cls_per_task"][0]
        self.inc = args["cls_per_task"]

        self.use_diagonal = args["use_diagonal"]

        self._network = Adapter_merge(args, True)
        self.task_adapter_indices = [0] * len(self.inc)

        self.rb_r_head_frac = args["rb_r_head_frac"]
        self.rb_rho_head = args["rb_rho_head"]
        self.rb_rho_tail = args["rb_rho_tail"]

        self.logger = args["logger"]

      
     

    def after_task(self) -> None:
        """
        Hook called after each task finishes.
        Freeze current network and register the adapter into the adapter list.
        """
        self._known_classes = self._total_classes
        self._network.freeze()
        self._network.backbone.add_adapter_to_list()

    def get_cls_range(self, task_id: int):
        """
        Return class index range [start, end) for a given task.

        Args:
            task_id: Task index.

        Returns:
            (start_cls, end_cls)
        """
        if task_id == 0:
            start_cls = 0
            end_cls = self.init_cls
        else:
            start_cls = self.init_cls + sum(self.inc[1:task_id])
            end_cls = start_cls + self.inc[task_id]
        return start_cls, end_cls

    def replace_fc(self, train_loader: DataLoader) -> None:
        """
        Replace classifier weights with class prototypes computed from features
        of the current task (ProtoNet-style FC initialization).
        """
        model = self._network
        model.eval()

        with torch.no_grad():
            start_idx = 0
            for index in range(start_idx, self._cur_task + 1):
                if self.use_diagonal and index != -1 and index != self._cur_task:
                    continue

                if index != -1:
                    embedding_list, label_list = [], []
                    for _, batch in enumerate(train_loader):
                        _, data, label = batch
                        data = data.to(self._device)
                        label = label.to(self._device)

                        embedding = model.backbone.forward_proto(data, adapt_index=0)
                        embedding_list.append(embedding.cpu())
                        label_list.append(label.cpu())

                    embedding_list = torch.cat(embedding_list, dim=0)
                    label_list = torch.cat(label_list, dim=0)

                    class_list = np.unique(self.train_dataset_for_protonet.labels)
                    for class_index in class_list:
                        data_index = (label_list == class_index).nonzero().squeeze(-1)
                        embedding = embedding_list[data_index]
                        proto = embedding.mean(0)
                        model.fc.weight.data[class_index, : self._network.out_dim] = proto
                    break

    def incremental_train(self, data_manager) -> None:
        self._cur_task += 1
        self._cur_task_size = data_manager.get_task_size(self._cur_task)
        self._total_classes = self._known_classes + self._cur_task_size

        seed_all(self.args["seed"])
        g_train = torch.Generator(device="cpu").manual_seed(self.args["seed"])

        self._network.update_fc(self._total_classes)
        self.logger.info("Learning on {}-{}".format(self._known_classes, self._total_classes))

        self.data_manager = data_manager

        self.train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source="train", mode="train")
        self.train_loader = DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True,
            num_workers=num_workers,
            generator=g_train,
            worker_init_fn=make_worker_init_fn(self.args["seed"]),
        )

        self.test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source="test", mode="test")
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)

        self.train_dataset_for_protonet = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="test",
        )
        self.train_loader_for_protonet = DataLoader(
            self.train_dataset_for_protonet,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=num_workers,
        )

        # 1) standard training
        self._train(self.train_loader, self.test_loader)

        # 2) SVD-based alignment + adaptive merge into base adapter
        try:
            svd_dev = self.args.get("merge_svd_device", "auto")

            if self._cur_task != 0:
                self.svd_align_merge_adapter_into(
                    src_index=self._cur_task,
                    dst_index=0,
                    svd_device=svd_dev,
                )
            else:
                # First task: set base_adapter directly from current adapter
                self._network.backbone.base_adapter = (
                    copy.deepcopy(self._network.backbone.cur_adapter)
                    .requires_grad_(False)
                )

            # Route current task to merged adapter index (0)
            self.task_adapter_indices[self._cur_task] = 0

            self.logger.info("[SVD-Align Merge] adapter {} -> 0".format(self._cur_task))
        except Exception as e:
            self.logger.exception(f"[SVD-Align Merge] Failed: {e}")

        # 3) refresh classifier with prototypes of merged space
        self.replace_fc(self.train_loader_for_protonet)

    def _train(self, train_loader: DataLoader, test_loader: DataLoader) -> None:
        """
        Wrapper to construct optimizer/scheduler and call the main training loop.
        """
        self._network.to(self._device)

        if self._cur_task == 0 or self.init_cls == self.inc:
            optimizer = self.get_optimizer(lr=self.args["init_lr"])
            # scheduler = self.get_scheduler(optimizer, self.args["init_epochs"])
        else:
            if "later_lr" not in self.args or self.args["later_lr"] == 0:
                self.args["later_lr"] = self.args["init_lr"]
            if "later_epochs" not in self.args or self.args["later_epochs"] == 0:
                self.args["later_epochs"] = self.args["init_epochs"]
            optimizer = self.get_optimizer(lr=self.args["later_lr"])
            # scheduler = self.get_scheduler(optimizer, self.args["later_epochs"])

        # self._init_train(train_loader, test_loader, optimizer, scheduler)
        self._init_train(train_loader, test_loader, optimizer)

    def get_optimizer(self, lr: float):
        """
        Build optimizer over trainable parameters.
        """
        params = filter(lambda p: p.requires_grad, self._network.parameters())

        if self.args["optimizer"] == "sgd":
            optimizer = optim.SGD(params, momentum=0.9, lr=lr, weight_decay=self.weight_decay)
        elif self.args["optimizer"] == "adam":
            optimizer = optim.Adam(params, lr=lr, weight_decay=self.weight_decay)
        elif self.args["optimizer"] == "adamw":
            optimizer = optim.AdamW(params, lr=lr, weight_decay=self.weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {self.args['optimizer']}")

        return optimizer

    def get_scheduler(self, optimizer, epoch: int):
        """
        Build LR scheduler.
        """
        if self.args["scheduler"] == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=epoch, eta_min=self.min_lr)
        elif self.args["scheduler"] == "steplr":
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=self.args["init_milestones"], gamma=self.args["init_lr_decay"])
        elif self.args["scheduler"] == "constant":
            scheduler = None
        else:
            raise ValueError(f"Unknown scheduler: {self.args['scheduler']}")

        return scheduler

    def _compute_task_log_prior(self, train_loader: DataLoader) -> torch.Tensor:
        """
        Compute log prior (log class counts) for current task's local classes [0, inc-1].
        Returns: log_prior (Tensor shape [inc], on self._device)
        """
        # current task class count
        if self._cur_task == 0:
            num_cur = self.init_cls
            start_cls = 0
        else:
            num_cur = self.inc[self._cur_task]
            start_cls, _ = self.get_cls_range(self._cur_task)  # == self._known_classes usually

        counts = torch.zeros(num_cur, dtype=torch.long)

        # iterate once over the loader (OK; longtail needs accurate prior)
        for _, (_, _, targets) in enumerate(train_loader):
            targets = targets.to("cpu")
            aux = targets - start_cls  # local labels
            valid = (aux >= 0) & (aux < num_cur)
            if valid.any():
                counts += torch.bincount(aux[valid], minlength=num_cur)

        counts = counts.clamp_min(1).to(self._device).float()
        return counts.log()

    

    def _init_train(self, train_loader: DataLoader, test_loader: DataLoader, optimizer):
        """
        Main epoch loop: Balanced Softmax (for long-tailed) on current-task classes.
        """

        # ----- epochs -----
        if self._cur_task == 0 or self.init_cls == self.inc:
            epochs = self.args["init_epochs"]
        else:
            epochs = self.args["later_epochs"]

        scheduler = self.get_scheduler(optimizer, epochs)
        prog_bar = tqdm(range(epochs))

        # ----- current task class range & num classes -----
        if self._cur_task == 0:
            start_cls = 0
            num_cur = self.init_cls
        else:
            start_cls, end_cls = self.get_cls_range(self._cur_task)
            num_cur = end_cls - start_cls  # == self.inc[self._cur_task]

        # ----- compute log prior (log class counts) for Balanced Softmax -----
        counts = torch.zeros(num_cur, dtype=torch.long)
        for _, (_, _, t) in enumerate(train_loader):
            t = t.to("cpu")
            aux = t - start_cls
            valid = (aux >= 0) & (aux < num_cur)
            if valid.any():
                counts += torch.bincount(aux[valid], minlength=num_cur)
        log_prior = counts.clamp_min(1).float().log().to(self._device)  # [num_cur]

        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.0
            correct, total = 0, 0

            for _, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                aux_targets = targets - start_cls
                valid = (aux_targets >= 0) & (aux_targets < num_cur)

                output = self._network(inputs, test=False)
                logits = output["logits"]
                logits_bal = logits + log_prior.unsqueeze(0)                  # [B, num_cur]

                loss = F.cross_entropy(logits_bal, aux_targets, ignore_index=-1)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                losses += loss.item()

                preds = torch.argmax(logits_bal, dim=1)
                correct += (preds[valid] == aux_targets[valid]).sum().cpu()
                total += valid.sum().item()

            if scheduler is not None:
                scheduler.step()

            train_acc = np.around(tensor2numpy(correct) * 100.0 / max(total, 1), decimals=2)

            info = (
                "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    epochs,
                    losses / len(train_loader),
                    train_acc,
                )
            )
            prog_bar.set_description(info)

        self.logger.info(info)

    def _compute_accuracy(self, model, loader: DataLoader) -> float:
        """
        Compute top-1 accuracy of a given model on a loader.
        """
        model.eval()
        correct, total = 0, 0
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = model.forward(inputs, test=True)["logits"]
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == targets).sum()
            total += len(targets)

        return np.around(
            tensor2numpy(correct) * 100.0 / total, decimals=2
        )

    def _eval_cnn(self, loader: DataLoader):
        """
        Evaluate CNN classifier: returns top-k predictions and ground truth.
        """
        self._network.eval()
        y_pred, y_true = [], []
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)

            with torch.no_grad():
                outputs = self._network.forward(inputs, test=True)["logits"]

            predicts = torch.topk(
                outputs, k=self.topk, dim=1, largest=True, sorted=True
            )[1]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())

        return np.concatenate(y_pred), np.concatenate(y_true)

    def _as_2d(self, mat: torch.Tensor):
        """
        Reshape an arbitrary tensor into a 2D matrix [out, in_like] for SVD.

        Rules:
            - 0D/1D -> [1, N]
            - 2D    -> as is
            - >2D   -> flatten all but first dimension to columns
        """
        if mat.dim() == 0:
            m2d = mat.reshape(1, 1)
            info = ("scalar", mat.shape)
        elif mat.dim() == 1:
            m2d = mat.unsqueeze(0)
            info = ("1d", mat.shape)
        elif mat.dim() == 2:
            m2d = mat
            info = ("2d", mat.shape)
        else:
            out = mat.shape[0]
            m2d = mat.reshape(out, -1)
            info = ("nd", mat.shape)
        return m2d, info

    def _from_2d(self, m2d: torch.Tensor, info):
        """
        Inverse of _as_2d: reshape a 2D matrix back to the original shape.
        """
        kind, orig_shape = info
        if kind == "scalar":
            return m2d.reshape(orig_shape)
        elif kind == "1d":
            return m2d.squeeze(0).reshape(orig_shape)
        elif kind == "2d":
            return m2d.reshape(orig_shape)
        else:
            return m2d.reshape(orig_shape)

    @torch.no_grad()
    def merge_adapter_into(self, src_index: int, dst_index: int, alpha: float = 1.0, beta: float = 1.0) -> None:
        """
        Simple backup: linear merge without SVD
            dst = alpha * dst + beta * src
        """
        adapters = self._network.backbone.adapter_list
        src = adapters[src_index]
        dst = adapters[dst_index]

        src_params = dict(src.named_parameters())
        for name, p_dst in dst.named_parameters():
            p_src = src_params[name]
            p_dst.mul_(alpha).add_(p_src, alpha=beta)

    @torch.no_grad()
    def copy_adapter(self, to_index: int, from_index: int) -> None:
        """
        Copy parameters from one adapter to another.
        """
        adapters = self._network.backbone.adapter_list
        src = adapters[from_index]
        dst = adapters[to_index]

        src_params = dict(src.named_parameters())
        for name, p_dst in dst.named_parameters():
            p_dst.copy_(src_params[name])

    

    @torch.no_grad()
    def svd_align_merge_adapter_into(
        self,
        src_index: int,
        dst_index: int,
        eps: float = 1e-8,
        svd_device: str = "auto",  # "auto" / "cpu" / "cuda"
    ) -> None:
        """
        SVD-based adapter alignment and merge (concat-SVD variant).

        For each matched parameter:
            1) reshape dst/src/target to 2D.
            2) concat X = [M_dst | M_src] along columns.
            3) compute econ SVD on X: X = U S V^T.
            4) split V^T back into (dst-part, src-part) in the shared basis.
            5) blend in coefficient (V-space) with weights w_d, w_s.
            6) rank-wise threshold modulation in singular directions.
            7) reconstruct dst-shaped matrix and write back.
        """
        adapters = self._network.backbone.adapter_list
        cur = self._network.backbone.cur_adapter
        target_mod = adapters[dst_index]

        alpha_eff = float(sum(self.inc[: self._cur_task])) if hasattr(self, "inc") else 1.0
        beta_eff = float(self.inc[self._cur_task]) if hasattr(self, "inc") else 1.0

        # Decide which adapter is treated as "dst"
        if alpha_eff <= beta_eff:
            dst_mod = cur
            src_mod = adapters[dst_index]
        else:
            dst_mod = adapters[dst_index]
            src_mod = cur

        self._network.backbone.base_adapter = copy.deepcopy(dst_mod).requires_grad_(False)

        src_params = dict(src_mod.named_parameters())
        dst_params = dict(dst_mod.named_parameters())

        # pick svd device per param (can vary)
        def _pick_svd_device(p: torch.Tensor) -> torch.device:
            if svd_device == "cpu":
                return torch.device("cpu")
            elif svd_device == "cuda":
                return torch.device("cuda" if torch.cuda.is_available() else p.device)
            else:
                return p.device

        for name, p_tgt in target_mod.named_parameters():
            p_src = src_params[name]
            p_dst = dst_params[name]

            if p_tgt.shape != p_src.shape or p_tgt.shape != p_dst.shape:
                raise RuntimeError(
                    "[svd_align_merge_adapter_into] Shape mismatch at {}: "
                    "target {}, src {}, dst {}".format(
                        name, p_tgt.shape, p_src.shape, p_dst.shape
                    )
                )

            # ---- 1) to 2D ----
            m_ref, info = self._as_2d(p_tgt.data)  # only for shape/info
            m_dst, _ = self._as_2d(p_dst.data)
            m_src, _ = self._as_2d(p_src.data)

            # ---- 2) concat + SVD ----
            # X: [m, c_dst + c_src] where c_dst == c_src usually
            dev_ctx = _pick_svd_device(p_dst)
            X = torch.cat([m_dst, m_src], dim=1).to(dev_ctx)

            # econ SVD: U [m, r], S [r], Vh [r, n]
            U, S, Vh = torch.linalg.svd(X, full_matrices=False)

            c_dst = m_dst.shape[1]
            c_src = m_src.shape[1]
            # split Vh by columns (NOT by rows): Vh is [r, c_dst+c_src]
            Vh_dst = Vh[:, :c_dst]          # [r, c_dst]
            Vh_src = Vh[:, c_dst:c_dst+c_src]  # [r, c_src]

            # ---- 3) blending weights wd, ws ----
            alpha_ws = sum(self.inc[: self._cur_task])
            beta_ws = self.inc[self._cur_task]
            s = max(alpha_ws + beta_ws, eps)
            wd = float(alpha_ws / s)
            ws = float(beta_ws / s)

            # ---- 4) blend in shared SVD basis ----
            Vh_blend = wd * Vh_dst + ws * Vh_src
            delta = Vh_blend - Vh_dst  # relative to dst-part

            # ---- 5) directional gating ----
            r = S.shape[0]
            r_head_frac = self.rb_r_head_frac
            rho_head = self.rb_rho_head
            rho_tail = self.rb_rho_tail

            r_head = int(max(1, min(r - 1, round(r_head_frac * r))))
            mask_row = torch.empty(r, device=dev_ctx, dtype=delta.dtype)
            mask_row[:r_head] = rho_head
            mask_row[r_head:] = rho_tail

            mask = mask_row.unsqueeze(1)  # [r, 1]
            Vh_m = Vh_dst + mask * delta  # [r, c_dst]

            # ---- 6) reconstruct dst-shaped matrix ----
            m_m = (U * S.unsqueeze(0)) @ Vh_m   # [m, c_dst]
            m_m = m_m.to(p_tgt.device)

            # ---- 7) write back ----
            p_tgt.copy_(self._from_2d(m_m, info))