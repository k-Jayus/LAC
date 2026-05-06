"""
Dual Manifold Framework — Feature Inversion (Architecture-Agnostic, Deepest-Layer-Only Loss)
============================================================================================
核心目标: 特征反演 (Feature Inversion)，绝对的正交层级解耦。
网络骨干: 通用 CNN 架构 (ResNet, DenseNet, ConvNeXt 等，由 timm 驱动)
数据集:   Oxford-IIIT Pet

工程重构说明:
1. 引入 timm 特征金字塔提取，彻底废除硬编码切片。
2. 引入 Dummy Tensor 探针，动态推导异构网络的流形断层通道数。
3. 动态实例化 LAC，实现 1 个 lac_stem + N 个内部 lac 的自适应物理防线。

与原版对齐修复:
[Fix-1] lac 通道维度: 使用 boundary_channels (输入侧通道) 而非 stage_channels (输出侧通道)，
        防止 b>=1 时 VJP shape 与 GroupNorm 通道数不匹配导致的 Crash。
[Fix-2] lac num_groups: 对齐原版 LACRefiner(ch, ch)，使用 num_groups=in_channels，
        等价于 InstanceNorm，而非 GroupNorm(32, ch)。
[Fix-3] _compute_energy: 对齐原版，保留 .abs() 以保证能量计算行为完全一致。

【本版修改】: 训练损失仅使用最深一层 (Stage N-1)，推断时仍输出完整语义反演谱。
"""

import os

# 必须在导入 timm/huggingface_hub 之前设置！
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from typing import List, Tuple, Dict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torchvision
import timm
from timm.optim import Lamb

# ===========================================================================
# 1. LAC Refiner
# ===========================================================================
class LACRefiner(nn.Module):
    def __init__(self, in_channels: int, num_groups: int = 32):
        super().__init__()
        self.num_groups = num_groups
        self.eps = 1e-5
        self.log_gamma = nn.Parameter(torch.zeros(in_channels))  # 语义清晰
        self.beta = nn.Parameter(torch.zeros(in_channels))

    def forward(self, vjp: torch.Tensor) -> torch.Tensor:
        return F.group_norm(
            vjp,
            self.num_groups,
            weight=(self.log_gamma).exp(),
            bias=self.beta-self.beta.mean().detach(),
            eps=self.eps
        )


# ===========================================================================
# 2. Universal Frozen Encoder (由 timm 和探针驱动)
# ===========================================================================
import re

class FrozenEncoder(nn.Module):
    def __init__(self, model_name: str = 'resnet18', pretrained_path: str = None):
        super().__init__()
        print(f"=> Building TIMM Universal Extractor: {model_name}")

        # ── Step 1: 探测可用特征层数 ──────────────────────────────────────────
        _probe = timm.create_model(model_name, pretrained=False, features_only=True)
        _num_features = len(_probe.feature_info)
        del _probe
        out_indices = tuple(range(_num_features))
        print(f"   [Probe] {_num_features} feature levels → out_indices={out_indices}")

        # ── Step 2: 构建特征提取器 ────────────────────────────────────────────
        self.bb = timm.create_model(
            model_name,
            pretrained=(pretrained_path is None),
            features_only=True,
            out_indices=out_indices
        )

        # ── Step 3: 搜索 stem 并挂 hook ──────────────────────────────────────
        self._stem_out = None
        stem_mod, stem_path = self._find_stem_module()
        if stem_mod is not None:
            stem_mod.register_forward_hook(self._stem_hook)
            self._has_stem_hook = True
            print(f"   [Stem] Hook registered at '{stem_path}'")
        else:
            self._has_stem_hook = False
            print(f"   [Stem] Not found, fallback: features[0] → h_stem")

        # ── Step 4: 加载自定义权重 ────────────────────────────────────────────
        if pretrained_path and os.path.isfile(pretrained_path):
            print(f"=> Loading custom weights from '{pretrained_path}'")
            checkpoint = torch.load(pretrained_path, map_location='cpu')
            state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint

            model_keys = set(self.bb.state_dict().keys())

            new_state_dict = {}
            for k, v in state_dict.items():
                k = k.replace('module.', '')
                if 'fc' in k or 'classifier' in k or 'head' in k:
                    continue

                # ① 直接匹配（ResNet 等）
                if k in model_keys:
                    new_state_dict[k] = v
                    continue

                # ② DenseNet: features.xxx → features_xxx
                if k.startswith('features.'):
                    k_fixed = 'features_' + k[len('features.'):]
                    if k_fixed in model_keys:
                        new_state_dict[k_fixed] = v
                        continue

                # ③ timm 非 features_only ConvNeXt: stem.0.* → stem_0.*, stages.0.* → stages_0.*
                k_fixed = re.sub(r'^(\w+)\.(\d+)', r'\1_\2', k)
                if k_fixed in model_keys:
                    new_state_dict[k_fixed] = v
                    continue

                # ④ torchvision ConvNeXt → timm features_only ConvNeXt（结构差异最大）
                k_fixed, v_fixed = self._remap_torchvision_convnext(k, v, model_keys)
                if k_fixed is not None:
                    new_state_dict[k_fixed] = v_fixed
                    continue

                # ⑤ body. 前缀（部分 timm 版本的 FeatureHookNet）
                k_fixed = 'body.' + k
                if k_fixed in model_keys:
                    new_state_dict[k_fixed] = v
                    continue

            result = self.bb.load_state_dict(new_state_dict, strict=False)
            print(f"  Loaded:     {len(new_state_dict)} keys")
            print(f"  Missing:    {len(result.missing_keys)}")
            print(f"  Unexpected: {len(result.unexpected_keys)}")
            if len(result.missing_keys) > 0:
                raise RuntimeError(
                    f"Backbone loading failed! {len(result.missing_keys)} missing keys.\n"
                    f"First 5: {result.missing_keys[:5]}"
                )
            print("=> Custom weights loaded successfully.")

        # ── Step 5: 探针推导维度 ──────────────────────────────────────────────
        dummy_x = torch.zeros(1, 3, 224, 224)
        with torch.no_grad():
            dummy_features = self.bb(dummy_x)

        if self._has_stem_hook:
            self.stem_channels  = self._stem_out.shape[1]
            self.stage_channels = [f.shape[1] for f in dummy_features]
        else:
            self.stem_channels  = dummy_features[0].shape[1]
            self.stage_channels = [f.shape[1] for f in dummy_features[1:]]

        self.num_stages = len(self.stage_channels)
        self.boundary_channels = [self.stem_channels] + self.stage_channels[:-1]

        print(f"   [Probe] Stem Channels    : {self.stem_channels}")
        print(f"   [Probe] Stage Channels   : {self.stage_channels} (Total: {self.num_stages} stages)")
        print(f"   [Probe] Boundary Channels: {self.boundary_channels} (used for LAC sizing)")

        # ── Step 6: 绝对冻结 ──────────────────────────────────────────────────
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

    @staticmethod
    def _remap_torchvision_convnext(k: str, v, model_keys: set):
        """
        torchvision ConvNeXt → timm features_only ConvNeXt 的完整 key/shape 映射。

        torchvision 结构:
          features.0          → stem (Sequential: Conv2d + LayerNorm)
          features.1,3,5,7    → stage 0~3 的 blocks
          features.2,4,6      → stage 1~3 的 downsample

        timm features_only 结构:
          stem_0, stem_1      → stem 的 Conv2d 和 LayerNorm
          stages_i.blocks.j   → block 内部命名不同（见下）
          stages_i.downsample → 下采样层
        """
        if not k.startswith('features.'):
            return None, v

        parts = k.split('.')
        if len(parts) < 3:
            return None, v

        try:
            main_idx = int(parts[1])
        except ValueError:
            return None, v

        rest_parts = parts[2:]

        # ── Stem: features.0.{i}.* → stem_{i}.* ─────────────────────────────
        if main_idx == 0:
            stem_idx = rest_parts[0]
            rest = '.'.join(rest_parts[1:])
            k_fixed = f'stem_{stem_idx}' + (f'.{rest}' if rest else '')
            if k_fixed in model_keys:
                return k_fixed, v

        # ── Stage blocks（奇数索引 1,3,5,7 → stage 0,1,2,3）─────────────────
        elif main_idx % 2 == 1:
            stage_idx = (main_idx - 1) // 2
            block_idx = rest_parts[0]

            # block 内部子模块: features.A.B.block.{sub}.* → stages_X.blocks.B.{mapped}.*
            if len(rest_parts) >= 3 and rest_parts[1] == 'block':
                try:
                    sub_idx = int(rest_parts[2])
                except ValueError:
                    return None, v
                rest = '.'.join(rest_parts[3:])
                # torchvision block Sequential 索引 → timm 子模块名
                sub_map = {0: 'conv_dw', 2: 'norm', 3: 'mlp.fc1', 5: 'mlp.fc2'}
                if sub_idx in sub_map:
                    k_fixed = f'stages_{stage_idx}.blocks.{block_idx}.{sub_map[sub_idx]}'
                    if rest:
                        k_fixed += f'.{rest}'
                    if k_fixed in model_keys:
                        return k_fixed, v

            # layer_scale: shape (C,1,1) → gamma: shape (C,)
            elif len(rest_parts) >= 2 and rest_parts[1] == 'layer_scale':
                k_fixed = f'stages_{stage_idx}.blocks.{block_idx}.gamma'
                if k_fixed in model_keys:
                    return k_fixed, v.view(-1)  # (C,1,1) → (C,)

        # ── Downsample（偶数索引 2,4,6 → stages 1,2,3 的 downsample）─────────
        elif main_idx % 2 == 0 and main_idx > 0:
            stage_idx = main_idx // 2   # 2→1, 4→2, 6→3
            ds_idx = rest_parts[0]
            rest = '.'.join(rest_parts[1:])
            k_fixed = f'stages_{stage_idx}.downsample.{ds_idx}' + (f'.{rest}' if rest else '')
            if k_fixed in model_keys:
                return k_fixed, v

        return None, v

    def _stem_hook(self, module, input, output):
        self._stem_out = output

    def _find_stem_module(self):
        """
        按优先级搜索 stem 末端模块并挂 hook：
          ① 标准命名: self.bb.stem / patch_embed / conv_stem
          ② ConvNeXt flat 命名: self.bb.stem_1（norm 是最后一层，hook 它拿完整 stem 输出）
          ③ 在直接子模块（如 body）里找
        """
        candidates = ['stem', 'patch_embed', 'conv_stem']

        for name in candidates:
            mod = getattr(self.bb, name, None)
            if mod is not None:
                return mod, f'bb.{name}'

        # ConvNeXt features_only: stem 被展平为 stem_0 + stem_1，hook stem_1（LayerNorm）
        if hasattr(self.bb, 'stem_1'):
            return self.bb.stem_1, 'bb.stem_1'
        if hasattr(self.bb, 'stem_0'):
            return self.bb.stem_0, 'bb.stem_0'

        for child_name, child in self.bb.named_children():
            for name in candidates:
                mod = getattr(child, name, None)
                if mod is not None:
                    return mod, f'bb.{child_name}.{name}'

        return None, None

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        features = self.bb(x)
        if self._has_stem_hook:
            h_stem         = self._stem_out
            stage_features = list(features)
        else:
            h_stem         = features[0]
            stage_features = list(features[1:])
        return h_stem, stage_features
# ===========================================================================
# 3. Dual Manifold Framework
# ===========================================================================
class DualManifoldFramework(nn.Module):
    def __init__(self, model_name: str = 'resnet18', pretrained_path: str = None,
                 num_groups: int = 32, top_k: int = 64):
        super().__init__()
        self.encoder = FrozenEncoder(model_name=model_name, pretrained_path=pretrained_path)
        self.top_k = top_k

        # [Fix-1] 用 boundary_channels 而非 stage_channels 定尺寸
        # [Fix-2] 传入 num_groups=ch，即 LACRefiner(ch, ch)，与原版完全对齐
        self.lac = nn.ModuleList([
            LACRefiner(in_channels=ch, num_groups=ch)
            for ch in self.encoder.boundary_channels
        ])

        # 补齐物理防线：部署 Stem -> Pixel 边界的终极 LAC (永远是 3 通道)
        self.lac_stem = LACRefiner(in_channels=3, num_groups=3)

    @staticmethod
    def _compute_energy(stage_features: List[torch.Tensor]) -> List[torch.Tensor]:
        E_list = []
        for h in stage_features:
            # [Fix-3] 对齐原版：保留 .abs()，确保能量计算行为完全一致
            Z_l = h.detach().abs().mean(dim=[2, 3])
            Z_l_total = Z_l.sum(dim=1, keepdim=True).clamp(min=1e-8)
            E_list.append(Z_l / Z_l_total)
        return E_list

    def train(self, mode: bool = True):
        """
        【物理防线重写】: 拦截外部的 model.train() 调用，
        确保 FrozenEncoder 永远处于 eval 模式。
        """
        super().train(mode)
        self.encoder.eval()
        return self

    def _invert_single_channel(self, X_req: torch.Tensor, h_stem: torch.Tensor,
                               stage_features: List[torch.Tensor],
                               stage_idx: int, channel_idx: int) -> torch.Tensor:
        h_l = stage_features[stage_idx]
        seed = torch.zeros_like(h_l)
        seed[:, channel_idx] = h_l[:, channel_idx]
        g = seed

        inputs_chain = [h_stem] + stage_features

        # 穿越内部 Stage 边界
        for b in range(stage_idx, -1, -1):
            vjp = torch.autograd.grad(
                outputs      = stage_features[b],
                inputs       = inputs_chain[b],
                grad_outputs = g,
                retain_graph = True,
                create_graph = self.training,
            )[0]
            g = self.lac[b](vjp)

        # 穿越 Stem 边界到达像素空间
        V_tilde_raw = torch.autograd.grad(
            outputs      = h_stem,
            inputs       = X_req,
            grad_outputs = g,
            retain_graph = True,
            create_graph = self.training,
        )[0]

        # 修复 stem 跃迁产生的终端狄拉克冲击
        V_tilde = self.lac_stem(V_tilde_raw)

        return V_tilde

    def invert_stage(self, X_req: torch.Tensor, h_stem: torch.Tensor, stage_features: List[torch.Tensor],
                     energy_list: List[torch.Tensor], stage_idx: int) -> torch.Tensor:
        E_l = energy_list[stage_idx]
        C_l = stage_features[stage_idx].shape[1]
        k = min(self.top_k, C_l)

        if self.training:
            probs = E_l.mean(0)  # [C] 提取 Batch 平均概率作为 Proposal Distribution
            idxs = torch.multinomial(probs, k, replacement=True)  # 必须是有放回采样
        else:
            probs = E_l.mean(0)
            idxs = probs.topk(k).indices

        X_hat_l = torch.zeros_like(X_req)
        for i in range(k):
            c = idxs[i].item()
            E_lc = E_l[:, c]  # 该通道在每张图像上的真实能量 [B]

            with torch.enable_grad():
                V_tilde_lc = self._invert_single_channel(
                    X_req, h_stem, stage_features, stage_idx=stage_idx, channel_idx=c
                )

            if self.training:
                is_weight = E_lc / probs[c].clamp(min=1e-8)
                X_hat_l = X_hat_l + is_weight.view(-1, 1, 1, 1) * V_tilde_lc
            else:
                X_hat_l = X_hat_l + E_lc.view(-1, 1, 1, 1) * V_tilde_lc

        if self.training:
            X_hat_l = X_hat_l / k  # 蒙特卡洛平均

        return X_hat_l

    # === MODIFIED [1/3] ===
    # forward: 训练时只计算最深层 (Stage N-1) 的重建，节省前 N-1 条 Stage 的 VJP 开销。
    # 推断时仍计算全部 N 个 Stage，输出完整语义反演谱，visualize_epoch 不受影响。
    # 关键: 最深层的伴随路径穿越 lac[N-1] -> ... -> lac[0] -> lac_stem，
    # 所有 LAC 模块仍然获得训练梯度。
    def forward(self, X: torch.Tensor) -> Tuple[Dict[int, torch.Tensor], Dict]:
        X_req = X.detach().requires_grad_(True)
        with torch.enable_grad():
            h_stem, stage_features = self.encoder(X_req)

        energy_list = self._compute_energy(stage_features)
        recons = {}

        if self.training:
            # 训练期：仅反演最深层
            deepest = self.encoder.num_stages - 1
            recons[deepest] = self.invert_stage(X_req, h_stem, stage_features, energy_list, deepest)
        else:
            # 推断期：反演全部层级
            for l in range(self.encoder.num_stages):
                recons[l] = self.invert_stage(X_req, h_stem, stage_features, energy_list, l)

        info = {'energy_per_stage': [E.mean().item() for E in energy_list]}
        return recons, info


# ===========================================================================
# 4. Utilities, Loss & Optimizer
# ===========================================================================

# === MODIFIED [2/3] ===
# dual_manifold_loss: 去掉对所有 Stage 的循环求和，仅对 recons 中最深的一层计算 L1 损失。
def dual_manifold_loss(recons: Dict[int, torch.Tensor], X: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
    deepest = max(recons.keys())
    X_hat   = recons[deepest]
    loss    = F.l1_loss(X_hat, X)
    loss_dict = {f'loss_stage_{deepest}': loss.item()}
    return loss, loss_dict

def denormalize(tensor: torch.Tensor) -> np.ndarray:
    """归一化反转: [-1, 1] -> [0, 1] RGB"""
    img = tensor.detach().cpu() * 0.5 + 0.5
    return torch.clamp(img, 0, 1).permute(1, 2, 0).numpy()

def visualize_epoch(model: DualManifoldFramework, images: torch.Tensor, epoch: int, out_dir: str):
    """独立展示原图以及它在 Stage 0 到 Stage N-1 的纯粹反演图"""
    model.eval()
    os.makedirs(out_dir, exist_ok=True)

    original_top_k = model.top_k
    try:
        model.top_k = 512  # 强制画图时火力全开
        with torch.no_grad():
            recons, _ = model(images)
    finally:
        model.top_k = original_top_k

    n_images = min(2, images.size(0))
    num_cols = model.encoder.num_stages + 1  # 动态适配子图列数
    fig, axes = plt.subplots(n_images, num_cols, figsize=(4 * num_cols, 4 * n_images))
    if n_images == 1:
        axes = [axes]

    for i in range(n_images):
        axes[i][0].imshow(denormalize(images[i]))
        axes[i][0].set_title("Original")
        axes[i][0].axis('off')

        for l in range(model.encoder.num_stages):
            axes[i][l + 1].imshow(denormalize(recons[l][i]))
            axes[i][l + 1].set_title(f"Stage {l} Inversion")
            axes[i][l + 1].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'epoch_{epoch:03d}.png'))
    plt.close()


# ===========================================================================
# 5. Main Training Loop
# ===========================================================================
def main():
    # --- 架构适配超参数 ---
    MODEL_NAME   = 'convnext_base'
    #WEIGHTS_PATH = 'weights/resnet50_imagenet-1.pth'  # 如果用官方预训练，设为 None
    WEIGHTS_PATH = None
    BATCH_SIZE   = 20
    LR           = 1e-3
    EPOCHS       = 50
    TOP_K_TRAIN  = 8
    DEVICE       = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
    # === MODIFIED [3/3] ===
    # 输出目录名加 _deepest 后缀以区分实验
    OUT_DIR      = f'./vis_results_{MODEL_NAME}_imagenet'

    print(f"==> Using device: {DEVICE}")

    # --- 数据准备 ---
    from torch.utils.data import DataLoader
    # Data loading
    def get_imagenet_dataloaders(data_dir='/home/ubuntu/sukaixiang/imagenet',
                                 batch_size=128,
                                 num_workers=8):
        """
        创建ImageNet数据集的标准DataLoader

        Args:
            data_dir: ImageNet数据集根目录
            batch_size: 批次大小
            num_workers: 数据加载的进程数

        Returns:
            train_loader, val_loader
        """

        # ImageNet的标准归一化参数
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        # 训练集的数据增强
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

        # 验证集的预处理（不做数据增强）
        val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])

        # 创建数据集
        train_dataset = torchvision.datasets.ImageFolder(
            root=data_dir + '/train',
            transform=train_transform
        )

        val_dataset = torchvision.datasets.ImageFolder(
            root=data_dir + '/val',
            transform=val_transform
        )

        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True  # 训练时通常丢弃最后不完整的批次
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False
        )

        return train_loader, val_loader

    train_loader, val_loader = get_imagenet_dataloaders(
        data_dir='/home/ubuntu/sukaixiang/imagenet',
        batch_size=BATCH_SIZE,
        num_workers=8
    )
    # --- 模型与优化器 ---
    model = DualManifoldFramework(
        model_name=MODEL_NAME,
        pretrained_path=WEIGHTS_PATH,
        top_k=TOP_K_TRAIN
    ).to(DEVICE)

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR, weight_decay=1e-5,eps=1e-16)

    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    fixed_imgs, _ = next(iter(train_loader))
    fixed_imgs = fixed_imgs[:2].to(DEVICE)

    # --- 训练循环 ---
    print(f"\n==> Start Training Dual Manifold Framework on [{MODEL_NAME}] (Deepest-Layer-Only Loss) ...")
    TOTAL_STEPS = 10000  # 总训练步数，替代 EPOCHS
    LOG_EVERY = 50  # 每隔多少步打印一次 loss
    VIS_EVERY = 500  # 每隔多少步可视化一次
    SAVE_EVERY = 1000  # 每隔多少步保存一次 checkpoint

    model.train()
    running_loss = 0.0
    log_interval_count = 0

    # 构造无限 data iterator
    def infinite_loader(loader):
        while True:
            for data in loader:
                yield data

    data_iter = infinite_loader(train_loader)

    for step in range(1, TOTAL_STEPS + 1):
        imgs, _ = next(data_iter)
        imgs = imgs.to(DEVICE)

        optimizer.zero_grad()

        recons, info = model(imgs)
        loss, loss_dict = dual_manifold_loss(recons, imgs)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            filter(lambda p: p.requires_grad, model.parameters()), max_norm=0.01
        )
        optimizer.step()

        running_loss += loss.item()
        log_interval_count += 1

        # --- 日志打印 ---
        if step % LOG_EVERY == 0:
            avg_loss = running_loss / log_interval_count
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Step [{step}/{TOTAL_STEPS}] | Avg Loss: {avg_loss:.4f} | LR: {current_lr:.2e}")
            print(" | ".join([f"{k}: {v:.4f}" for k, v in loss_dict.items()]))
            running_loss = 0.0
            log_interval_count = 0

        # --- 可视化 ---
        if step % VIS_EVERY == 0:
            model.eval()
            visualize_epoch(model, fixed_imgs, step, OUT_DIR)
            model.train()

        # --- 保存 checkpoint ---
        if step % SAVE_EVERY == 0:
            torch.save({
                'lac': model.lac.state_dict(),
                'lac_stem': model.lac_stem.state_dict(),
            }, f'imagenet_lac_{MODEL_NAME}_step{step:06d}.pth')
            print(f"[*] Checkpoint saved at Step {step}")


if __name__ == '__main__':
    main()