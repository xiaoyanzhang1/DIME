import copy
import torch
from torch import nn

from backbone.linears import CosineLinear


def get_backbone(args, pretrained: bool = False):
    """
    Build the backbone network according to args["backbone_type"].

    Currently supports ViT-Base variants with One-A style adapters.
    """
    name = args["backbone_type"].lower()

    # ViT-OneA variants
    if "_onea" in name:
        ffn_num = args["ffn_num"]

        if args["model_name"] == "onea":
            from backbone import vit_onea
            from easydict import EasyDict

            tuning_config = EasyDict(
                # AdaptFormer-related config
                ffn_adapt=True,
                ffn_option="parallel",
                ffn_adapter_layernorm_option="none",
                ffn_adapter_init_option="lora",
                ffn_adapter_scalar="0.1",
                ffn_num=ffn_num,
                d_model=768,
                # VPT-related config (disabled)
                vpt_on=False,
                vpt_num=0,
                _device=args["device"][0],
            )

            if name == "vit_base_patch16_224_onea":
                model = vit_onea.vit_base_patch16_224_onea(
                    num_classes=0,
                    global_pool=False,
                    drop_path_rate=0.0,
                    tuning_config=tuning_config,
                )
                model.out_dim = 768
            elif name == "vit_base_patch16_224_in21k_onea":
                model = vit_onea.vit_base_patch16_224_in21k_onea(
                    num_classes=0,
                    global_pool=False,
                    drop_path_rate=0.0,
                    tuning_config=tuning_config,
                )
                model.out_dim = 768
            else:
                raise NotImplementedError(f"Unknown backbone type: {name}")

            # Backbone is returned in eval mode by default (adapters are handled separately)
            return model.eval()

    raise NotImplementedError(f"Unknown backbone type: {name}")


class BaseNet(nn.Module):
    """
    Base network wrapper for backbone + classifier head.
    """

    def __init__(self, args, pretrained: bool):
        super().__init__()

        print("Initializing BaseNet.")
        self.backbone = get_backbone(args, pretrained)
        print("Backbone initialized.")

        self.fc = None
        self._device = args["device"][0]

        if "resnet" in args["backbone_type"].lower():
            self.model_type = "cnn"
        else:
            self.model_type = "vit"

    @property
    def feature_dim(self) -> int:
        return self.backbone.out_dim

    def extract_vector(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract backbone features only (no classifier head).
        """
        if self.model_type == "cnn":
            outputs = self.backbone(x)
            return outputs["features"]
        else:
            return self.backbone(x)

    def forward(self, x: torch.Tensor):
        """
        Forward through backbone and classifier.
        For CNNs, backbone is expected to return a dict including 'features'.
        For ViTs, backbone returns pure features.
        """
        if self.model_type == "cnn":
            x_dict = self.backbone(x)
            logits = self.fc(x_dict["features"])
            x_dict["logits"] = logits
            return x_dict
        else:
            feats = self.backbone(x)
            logits = self.fc(feats)
            out = {"features": feats, "logits": logits}
            return out

    def update_fc(self, nb_classes: int):
        """
        Placeholder. Implemented in subclasses.
        """
        raise NotImplementedError

    def generate_fc(self, in_dim: int, out_dim: int):
        """
        Placeholder. Implemented in subclasses.
        """
        raise NotImplementedError

    def copy(self):
        return copy.deepcopy(self)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()
        return self


class Adapter_merge(BaseNet):
    """
    Adapter-based incremental classifier with a cosine classifier head.

    - Uses the ViT-OneA backbone with adapters.
    - Maintains a proxy FC for current-task training.
    """

    def __init__(self, args, pretrained: bool = True):
        super().__init__(args, pretrained)

        self.args = copy.deepcopy(args)

        self.inc = args["cls_per_task"]
        self.init_cls = args["cls_per_task"][0]

        self._cur_task = -1
        self.out_dim = self.backbone.out_dim
        self.fc = None

        # self.use_init_ptm = args["use_init_ptm"]

    def freeze(self):
        """
        Freeze all parameters in this module.
        """
        for _, param in self.named_parameters():
            param.requires_grad = False

    @property
    def feature_dim(self) -> int:
        """
        Feature dimension used by the classifier.
        """
        return self.out_dim

    def update_fc(self, nb_classes: int) -> None:
        """
        Update classifier heads for a new task.

        - proxy_fc: task-specific classifier for current task classes.
        - fc: global cosine classifier for all seen classes.
        """
        self._cur_task += 1

        # Proxy FC for current task
        if self._cur_task == 0:
            proxy_out = self.init_cls
        else:
            proxy_out = self.inc[self._cur_task]

        self.proxy_fc = self.generate_fc(self.out_dim, proxy_out).to(self._device)

        # Global FC for all classes seen so far
        fc = self.generate_fc(self.feature_dim, nb_classes).to(self._device)

        if self.fc is not None:
            old_nb_classes = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            fc.sigma.data = self.fc.sigma.data
            fc.weight.data[:old_nb_classes, :] = weight

        del self.fc
        self.fc = fc

    def generate_fc(self, in_dim: int, out_dim: int) -> CosineLinear:
        """
        Build a cosine classifier layer.
        """
        return CosineLinear(in_dim, out_dim)

    def extract_vector(self, x: torch.Tensor) -> torch.Tensor:
        """
        Override: always use the backbone's feature output directly.
        """
        return self.backbone(x)

    def forward(self, x: torch.Tensor, test: bool = False):
        """
        Forward pass.

        During training (test=False):
            - Use backbone.forward(x, use_test=False) to get adapter features.
            - Apply proxy_fc for current-task logits.

        During evaluation (test=True):
            - Use backbone.forward(x, use_test=True).
            - Apply global fc over all seen classes.
        """
        if not test:
            feats = self.backbone.forward(x, False)
            out = self.proxy_fc(feats)  
        else:
            feats = self.backbone.forward(
                x, True
            )
            out = self.fc(feats)         

        # Attach features into the dict
        out.update({"features": feats})
        return out

    def show_trainable_params(self) -> None:
        """
        Print names and sizes of all trainable parameters.
        """
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(name, param.numel())
