# --------------------------------------------------------
# References:
# https://github.com/jxhe/unify-parameter-efficient-tuning
# timm: https://github.com/rwightman/pytorch-image-models
# DeiT: https://github.com/facebookresearch/deit
# MAE:  https://github.com/facebookresearch/mae
# --------------------------------------------------------

import math
import copy
from functools import partial
from collections import OrderedDict

import torch
import torch.nn as nn
from timm.models.layers import DropPath
from timm.models.vision_transformer import PatchEmbed
import timm


class Adapter(nn.Module):
    def __init__(
        self,
        config=None,
        d_model=None,
        bottleneck=None,
        dropout: float = 0.0,
        init_option: str = "bert",
        adapter_scalar: str = "1.0",
        adapter_layernorm_option: str = "in",
    ):
        super().__init__()
        self.n_embd = config.d_model if d_model is None else d_model
        self.down_size = config.attn_bn if bottleneck is None else bottleneck

        self.adapter_layernorm_option = adapter_layernorm_option
        self.adapter_layer_norm_before = None
        if adapter_layernorm_option in ("in", "out"):
            self.adapter_layer_norm_before = nn.LayerNorm(self.n_embd)

        if adapter_scalar == "learnable_scalar":
            self.scale = nn.Parameter(torch.ones(1))
        else:
            self.scale = float(adapter_scalar)

        self.down_proj = nn.Linear(self.n_embd, self.down_size)
        self.non_linear_func = nn.ReLU()
        self.up_proj = nn.Linear(self.down_size, self.n_embd)
        self.dropout = dropout

        if init_option == "bert":
            # Not used in this project.
            raise NotImplementedError("BERT-style init is not implemented.")
        elif init_option == "lora":
            with torch.no_grad():
                nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
                nn.init.zeros_(self.up_proj.weight)
                nn.init.zeros_(self.down_proj.bias)
                nn.init.zeros_(self.up_proj.bias)

    def forward(self, x, add_residual: bool = True, residual=None):
        residual = x if residual is None else residual

        if self.adapter_layernorm_option == "in":
            x = self.adapter_layer_norm_before(x)

        down = self.down_proj(x)
        down = self.non_linear_func(down)
        down = nn.functional.dropout(down, p=self.dropout, training=self.training)
        up = self.up_proj(down)
        up = up * self.scale

        if self.adapter_layernorm_option == "out":
            up = self.adapter_layer_norm_before(up)

        if add_residual:
            return up + residual
        return up


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.head_dim = head_dim
        self.scale = head_dim**-0.5

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return (
            tensor.view(bsz, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape

        q = self.q_proj(x)
        k = self._shape(self.k_proj(x), -1, B).view(
            B * self.num_heads, -1, self.head_dim
        )
        v = self._shape(self.v_proj(x), -1, B).view(
            B * self.num_heads, -1, self.head_dim
        )
        q = self._shape(q, N, B).view(B * self.num_heads, -1, self.head_dim)

        attn_weights = torch.bmm(q, k.transpose(1, 2)) * self.scale
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_probs = self.attn_drop(attn_weights)
        attn_output = torch.bmm(attn_probs, v)

        attn_output = attn_output.view(B, self.num_heads, N, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(B, N, C)

        x = self.proj(attn_output)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        config=None,
        layer_id: int = None,
    ):
        super().__init__()
        self.config = config
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, mlp_hidden_dim)
        self.fc2 = nn.Linear(mlp_hidden_dim, dim)
        self.act = act_layer()
        self.mlp_drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor, adapt: Adapter = None) -> torch.Tensor:
        # Self-attention block
        x = x + self.drop_path(self.attn(self.norm1(x)))

        if adapt is not None:
            adapt_x = adapt(x, add_residual=False)
        else:
            adapt_x = None

        residual = x
        x = self.mlp_drop(self.act(self.fc1(self.norm2(x))))
        x = self.drop_path(self.mlp_drop(self.fc2(x)))

        # Adapter in FFN
        if adapt_x is not None and self.config.ffn_adapt:
            if self.config.ffn_option == "sequential":
                x = adapt(x)
            elif self.config.ffn_option == "parallel":
                x = x + adapt_x
            else:
                raise ValueError(f"Unknown ffn_option: {self.config.ffn_option}")

        x = residual + x
        return x


class VisionTransformer(nn.Module):
    """
    Vision Transformer with FFN adapters (DIME backbone).
    """

    def __init__(
        self,
        global_pool: bool = False,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        num_classes: int = 1000,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        representation_size=None,
        distilled: bool = False,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        embed_layer=PatchEmbed,
        norm_layer=None,
        act_layer=None,
        weight_init: str = "",
        tuning_config=None,
    ):
        super().__init__()

        print("Using ViT backbone with adapters (DIME).")

        self.tuning_config = tuning_config
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.num_tokens = 2 if distilled else 1

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        # Patch embedding
        self.patch_embed = embed_layer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        # Tokens & positional embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = (
            nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        )
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + self.num_tokens, embed_dim)
        )
        self.pos_drop = nn.Dropout(p=drop_rate)

        # Transformer blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.Sequential(
            *[
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                    config=tuning_config,
                    layer_id=i,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)

        # Optional representation layer
        if representation_size and not distilled:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(
                OrderedDict(
                    [
                        ("fc", nn.Linear(embed_dim, representation_size)),
                        ("act", nn.Tanh()),
                    ]
                )
            )
        else:
            self.pre_logits = nn.Identity()

        # Classifier heads (we do not use them in DIME, but keep for completeness)
        self.head = (
            nn.Linear(self.num_features, num_classes)
            if num_classes > 0
            else nn.Identity()
        )
        self.head_dist = None
        if distilled and num_classes > 0:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes)

        # Global pooling (DeiT / MAE style)
        self.global_pool = global_pool
        if self.global_pool:
            self.fc_norm = norm_layer(embed_dim)
            del self.norm  # keep only fc_norm

        # Prompt tuning (not used in DIME, but config-compatible)
        if tuning_config.vpt_on:
            assert tuning_config.vpt_num > 0, tuning_config.vpt_num
            self.embeddings = nn.ParameterList(
                [
                    nn.Parameter(torch.empty(1, tuning_config.vpt_num, embed_dim))
                    for _ in range(depth)
                ]
            )
            for e in self.embeddings:
                nn.init.xavier_uniform_(e.data)

        # Adapter containers
        self.config = tuning_config
        self._device = tuning_config._device
        self.adapter_list = []          # list of historical adapters (one ModuleList per task)
        self.cur_adapter = nn.ModuleList()
        self.get_new_adapter()

    # ---- Basic helpers -----------------------------------------------------

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token", "dist_token"}

    def get_classifier(self):
        if self.dist_token is None:
            return self.head
        return self.head, self.head_dist

    def reset_classifier(self, num_classes: int, global_pool: str = ""):
        self.num_classes = num_classes
        self.head = (
            nn.Linear(self.embed_dim, num_classes)
            if num_classes > 0
            else nn.Identity()
        )
        if self.num_tokens == 2:
            self.head_dist = (
                nn.Linear(self.embed_dim, self.num_classes)
                if num_classes > 0
                else nn.Identity()
            )

    def freeze(self):
        """
        Freeze the ViT backbone and keep only current adapters trainable.
        """
        for param in self.parameters():
            param.requires_grad = False

        for adapter in self.cur_adapter:
            adapter.requires_grad = True

    # ---- Adapter management -----------------------------------------------

    def get_new_adapter(self):
        """
        Create a fresh adapter stack for the current task.
        """
        config = self.config
        self.cur_adapter = nn.ModuleList()

        if config.ffn_adapt:
            for _ in range(len(self.blocks)):
                adapter = Adapter(
                    config=config,
                    dropout=0.1,
                    bottleneck=config.ffn_num,
                    init_option=config.ffn_adapter_init_option,
                    adapter_scalar=config.ffn_adapter_scalar,
                    adapter_layernorm_option=config.ffn_adapter_layernorm_option,
                ).to(self._device)
                self.cur_adapter.append(adapter)
            self.cur_adapter.requires_grad_(True)
        else:
            print("==== Not using adapters (ffn_adapt=False) ====")

    def add_adapter_to_list(self, add_adapter: bool = True):
        """
        Save the current adapter stack into adapter_list and create a new one.

        Args:
            add_adapter: if False, discard current adapter and replace it
                         with a fresh one before saving (used for ablations).
        """
        if not add_adapter:
            # Discard current adapter and create a new one, then store it.
            self.get_new_adapter()

        self.adapter_list.append(copy.deepcopy(self.cur_adapter.requires_grad_(False)))
        self.get_new_adapter()

    # ---- Forward passes ----------------------------------------------------

    def _prepare_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """
        Patch embedding + CLS token + positional embedding + dropout.
        """
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        return x

    def forward_train(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with current adapters (used during training).
        Returns a single CLS feature per image.
        """
        B = x.shape[0]
        x = self._prepare_tokens(x)

        for idx, blk in enumerate(self.blocks):
            if self.tuning_config.vpt_on:
                prompt = self.embeddings[idx].expand(B, -1, -1)
                x = torch.cat([prompt, x], dim=1)
            x = blk(x, self.cur_adapter[idx])
            if self.tuning_config.vpt_on:
                x = x[:, self.tuning_config.vpt_num :, :]

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)
            feat = self.fc_norm(x)
        else:
            x = self.norm(x)
            feat = x[:, 0]

        return feat

    def forward_test(self, x: torch.Tensor, use_init_ptm: bool = False):
        """
        Forward pass at test time.

        Current implementation returns a list with a single element:
        - either the last frozen adapter (if any) or the current adapter,
        - feature map over tokens [B, N, D].
        """
        del use_init_ptm  # kept only for API compatibility

        B = x.shape[0]
        x_tokens = self._prepare_tokens(x)
        features = []

        if len(self.adapter_list) != 0:
            # Use the last stored adapter stack
            idx_adapter = 0
            x = copy.deepcopy(x_tokens)
            for j, blk in enumerate(self.blocks):
                adapt = self.adapter_list[idx_adapter][j]
                x = blk(x, adapt)
            x = self.norm(x)
            features.append(x)
        else:
            # Use the current adapter stack
            x = copy.deepcopy(x_tokens)
            for j, blk in enumerate(self.blocks):
                adapt = self.cur_adapter[j]
                x = blk(x, adapt)
            x = self.norm(x)
            features.append(x)

        return features

    def forward(self, x: torch.Tensor, test: bool = False, use_init_ptm: bool = False):
        """
        Unified forward API:
        - train mode (test=False): return CLS features [B, D].
        - test mode (test=True):   stack CLS features from selected adapters.
        """
        if not test:
            return self.forward_train(x)

        features = self.forward_test(x, use_init_ptm=use_init_ptm)
        out = torch.empty(0, device=features[0].device)

        for feat in features:
            cls = feat[:, 0, :]  # CLS token
            out = torch.cat((out, cls), dim=1) if out.numel() > 0 else cls

        return out

    def forward_proto(self, x: torch.Tensor, adapt_index: int) -> torch.Tensor:
        """
        Feature extraction for prototypical classifier:
        returns CLS embedding for a given adapter index.

        adapt_index:
            -1: use pure pre-trained backbone without adapter.
            >=0: use adapter_list[adapt_index] if exists, otherwise cur_adapter.
        """
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x_init = self.pos_drop(x)

        # Use pure PTM
        if adapt_index == -1:
            x = copy.deepcopy(x_init)
            x = self.blocks(x)          # Blocks without adapter (default adapt=None)
            x = self.norm(x)
            return x[:, 0, :]

        # Use a specific adapter stack
        idx = adapt_index
        x = copy.deepcopy(x_init)
        for j, blk in enumerate(self.blocks):
            if idx < len(self.adapter_list):
                adapt = self.adapter_list[idx][j]
            else:
                adapt = self.cur_adapter[j]
            x = blk(x, adapt)
        x = self.norm(x)
        return x[:, 0, :]


# -------------------------------------------------------------------------
# Factory functions for DIME ViT backbones
# -------------------------------------------------------------------------

def _load_and_remap_vit_state_dict(model_name: str):
    """
    Load a timm ViT model and remap qkv / MLP weights to match DIME ViT.
    """
    checkpoint_model = timm.create_model(model_name, pretrained=True, num_classes=0)
    state_dict = checkpoint_model.state_dict()

    # 1) split qkv into q, k, v
    for key in list(state_dict.keys()):
        if "qkv.weight" in key:
            qkv_weight = state_dict.pop(key)
            q_weight = qkv_weight[:768]
            k_weight = qkv_weight[768 : 768 * 2]
            v_weight = qkv_weight[768 * 2 :]
            state_dict[key.replace("qkv.weight", "q_proj.weight")] = q_weight
            state_dict[key.replace("qkv.weight", "k_proj.weight")] = k_weight
            state_dict[key.replace("qkv.weight", "v_proj.weight")] = v_weight
        elif "qkv.bias" in key:
            qkv_bias = state_dict.pop(key)
            q_bias = qkv_bias[:768]
            k_bias = qkv_bias[768 : 768 * 2]
            v_bias = qkv_bias[768 * 2 :]
            state_dict[key.replace("qkv.bias", "q_proj.bias")] = q_bias
            state_dict[key.replace("qkv.bias", "k_proj.bias")] = k_bias
            state_dict[key.replace("qkv.bias", "v_proj.bias")] = v_bias

    # 2) rename mlp.fc* to fc*
    for key in list(state_dict.keys()):
        if "mlp.fc" in key:
            fc_weight = state_dict.pop(key)
            state_dict[key.replace("mlp.", "")] = fc_weight

    return state_dict


def vit_base_patch16_224_dime(pretrained: bool = False, **kwargs) -> VisionTransformer:
    """
    ViT-B/16 backbone on ImageNet-1k, adapted for DIME (adapters in FFN).
    """
    model = VisionTransformer(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )

    state_dict = _load_and_remap_vit_state_dict("vit_base_patch16_224")
    msg = model.load_state_dict(state_dict, strict=False)
    print(msg)

    # Freeze all PTM parameters, keep only adapter-related params trainable
    for name, p in model.named_parameters():
        if name in msg.missing_keys:
            p.requires_grad = True
        else:
            p.requires_grad = False
    return model


def vit_base_patch16_224_in21k_dime(
    pretrained: bool = False, **kwargs
) -> VisionTransformer:
    """
    ViT-B/16 backbone pre-trained on ImageNet-21k, adapted for DIME.
    """
    model = VisionTransformer(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )

    state_dict = _load_and_remap_vit_state_dict("vit_base_patch16_224_in21k")
    msg = model.load_state_dict(state_dict, strict=False)
    print(msg)

    for name, p in model.named_parameters():
        if name in msg.missing_keys:
            p.requires_grad = True
        else:
            p.requires_grad = False
    return model


