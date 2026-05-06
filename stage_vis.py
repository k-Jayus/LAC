"""
Dual Manifold Framework — Class Attribution via Cancellation Effect
====================================================================
PDF: N samples × (1 + num_stages + 3) columns
  Col 1:        Original Image
  Col 2..ns+1:  Stage 0–(ns-1) Semantic Inversion Spectrum
  Col ns+2:     Class Attribution (raw unnormalized α-weighted reconstruction)
  Col ns+3:     Positive α Channels Reconstruction
  Col ns+4:     Negative α Channels Reconstruction

适配性说明:
  - 模型: 修改 MODEL_NAME / WEIGHTS_PATH / LAC_CKPT / NC 即可切换任意 timm CNN
  - 数据: 只需提供标准 DataLoader (返回 images, labels)，无需 class_names
"""

import os, re
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import timm
from typing import List, Dict, Tuple


# ===========================================================================
# 1. LACRefiner
# ===========================================================================
class LACRefiner(nn.Module):
    def __init__(self, in_channels: int, num_groups: int = 32):
        super().__init__()
        self.num_groups = num_groups
        self.eps = 1e-5
        self.log_gamma = nn.Parameter(torch.zeros(in_channels))
        self.beta = nn.Parameter(torch.zeros(in_channels))

    def forward(self, vjp: torch.Tensor) -> torch.Tensor:
        return F.group_norm(
            vjp, self.num_groups,
            weight=self.log_gamma.exp(),
            bias=self.beta - self.beta.mean().detach(),
            eps=self.eps
        )


# ===========================================================================
# 2. FrozenEncoder (与训练代码完全对齐)
# ===========================================================================
class FrozenEncoder(nn.Module):
    def __init__(self, model_name: str = 'resnet18', pretrained_path: str = None):
        super().__init__()
        print(f"=> Building TIMM Universal Extractor: {model_name}")

        # ── Step 1: 探测可用特征层数 ─────────────────────────────
        _probe = timm.create_model(model_name, pretrained=False, features_only=True)
        _num_features = len(_probe.feature_info)
        del _probe
        out_indices = tuple(range(_num_features))
        print(f"   [Probe] {_num_features} feature levels → out_indices={out_indices}")

        # ── Step 2: 构建特征提取器 ───────────────────────────────
        self.bb = timm.create_model(
            model_name,
            pretrained=(pretrained_path is None),
            features_only=True,
            out_indices=out_indices
        )

        # ── Step 3: 搜索 stem 并挂 hook ─────────────────────────
        self._stem_out = None
        stem_mod, stem_path = self._find_stem_module()
        if stem_mod is not None:
            stem_mod.register_forward_hook(self._stem_hook)
            self._has_stem_hook = True
            print(f"   [Stem] Hook registered at '{stem_path}'")
        else:
            self._has_stem_hook = False
            print(f"   [Stem] Not found, fallback: features[0] → h_stem")

        # ── Step 4: 加载自定义权重 ───────────────────────────────
        if pretrained_path and os.path.isfile(pretrained_path):
            self._load_weights(pretrained_path)

        # ── Step 5: 探针推导维度 ─────────────────────────────────
        dummy_x = torch.zeros(1, 3, 224, 224)
        with torch.no_grad():
            dummy_features = self.bb(dummy_x)

        if self._has_stem_hook:
            self.stem_channels = self._stem_out.shape[1]
            self.stage_channels = [f.shape[1] for f in dummy_features]
        else:
            self.stem_channels = dummy_features[0].shape[1]
            self.stage_channels = [f.shape[1] for f in dummy_features[1:]]

        self.num_stages = len(self.stage_channels)
        self.boundary_channels = [self.stem_channels] + self.stage_channels[:-1]

        print(f"   [Probe] Stem Channels    : {self.stem_channels}")
        print(f"   [Probe] Stage Channels   : {self.stage_channels} (Total: {self.num_stages} stages)")
        print(f"   [Probe] Boundary Channels: {self.boundary_channels}")

        # ── Step 6: 绝对冻结 ─────────────────────────────────────
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

    def _load_weights(self, path):
        print(f"=> Loading custom weights from '{path}'")
        checkpoint = torch.load(path, map_location='cpu')
        state_dict = checkpoint.get('state_dict', checkpoint)

        model_keys = set(self.bb.state_dict().keys())
        new_state_dict = {}

        for k, v in state_dict.items():
            k = k.replace('module.', '')
            if any(t in k for t in ('fc', 'classifier', 'head')):
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

            # ③ timm 非 features_only ConvNeXt: stem.0.* → stem_0.*
            k_fixed = re.sub(r'^(\w+)\.(\d+)', r'\1_\2', k)
            if k_fixed in model_keys:
                new_state_dict[k_fixed] = v
                continue

            # ④ torchvision ConvNeXt → timm features_only ConvNeXt
            k_fixed, v_fixed = self._remap_torchvision_convnext(k, v, model_keys)
            if k_fixed is not None:
                new_state_dict[k_fixed] = v_fixed
                continue

            # ⑤ body. 前缀
            k_fixed = 'body.' + k
            if k_fixed in model_keys:
                new_state_dict[k_fixed] = v
                continue

        result = self.bb.load_state_dict(new_state_dict, strict=False)
        print(f"   Loaded: {len(new_state_dict)} keys, Missing: {len(result.missing_keys)}")
        if result.missing_keys:
            raise RuntimeError(
                f"Backbone loading failed! {len(result.missing_keys)} missing keys.\n"
                f"First 5: {result.missing_keys[:5]}"
            )
        print("=> Custom weights loaded successfully.")

    @staticmethod
    def _remap_torchvision_convnext(k: str, v, model_keys: set):
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

        # Stem: features.0.{i}.* → stem_{i}.*
        if main_idx == 0:
            stem_idx = rest_parts[0]
            rest = '.'.join(rest_parts[1:])
            k_fixed = f'stem_{stem_idx}' + (f'.{rest}' if rest else '')
            return (k_fixed, v) if k_fixed in model_keys else (None, v)

        # Stage blocks (奇数索引 1,3,5,7 → stage 0,1,2,3)
        elif main_idx % 2 == 1:
            stage_idx = (main_idx - 1) // 2
            block_idx = rest_parts[0]
            if len(rest_parts) >= 3 and rest_parts[1] == 'block':
                try:
                    sub_idx = int(rest_parts[2])
                except ValueError:
                    return None, v
                rest = '.'.join(rest_parts[3:])
                sub_map = {0: 'conv_dw', 2: 'norm', 3: 'mlp.fc1', 5: 'mlp.fc2'}
                if sub_idx in sub_map:
                    k_fixed = f'stages_{stage_idx}.blocks.{block_idx}.{sub_map[sub_idx]}'
                    if rest:
                        k_fixed += f'.{rest}'
                    return (k_fixed, v) if k_fixed in model_keys else (None, v)
            elif len(rest_parts) >= 2 and rest_parts[1] == 'layer_scale':
                k_fixed = f'stages_{stage_idx}.blocks.{block_idx}.gamma'
                return (k_fixed, v.view(-1)) if k_fixed in model_keys else (None, v)

        # Downsample (偶数索引 2,4,6 → stages 1,2,3 的 downsample)
        elif main_idx % 2 == 0 and main_idx > 0:
            stage_idx = main_idx // 2
            ds_idx = rest_parts[0]
            rest = '.'.join(rest_parts[1:])
            k_fixed = f'stages_{stage_idx}.downsample.{ds_idx}' + (f'.{rest}' if rest else '')
            return (k_fixed, v) if k_fixed in model_keys else (None, v)

        return None, v

    def _stem_hook(self, module, input, output):
        self._stem_out = output

    def _find_stem_module(self):
        candidates = ['stem', 'patch_embed', 'conv_stem']
        for name in candidates:
            mod = getattr(self.bb, name, None)
            if mod is not None:
                return mod, f'bb.{name}'
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
            return self._stem_out, list(features)
        return features[0], list(features[1:])


# ===========================================================================
# 3. ClassifierHead
# ===========================================================================
class ClassifierHead(nn.Module):
    def __init__(self, in_features: int, num_classes: int, use_norm: bool = True):
        super().__init__()
        self.use_norm = use_norm
        if use_norm:
            self.norm = nn.LayerNorm(in_features)
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, z):
        return self.fc(self.norm(z)) if self.use_norm else self.fc(z)


def load_classifier_head(path, in_features, num_classes=200, device='cpu'):
    checkpoint = torch.load(path, map_location='cpu')
    sd = {k.replace('module.', ''): v
          for k, v in (checkpoint.get('state_dict', checkpoint)).items()}

    nw = nb = fw = fb = None
    for a, b_ in [('head.norm.weight', 'head.norm.bias'),
                   ('classifier.0.weight', 'classifier.0.bias')]:
        if a in sd:
            nw, nb = sd[a], sd[b_]
            print(f"   [Head] Norm @ '{a}'")
            break

    for a, b_ in [('head.fc.weight', 'head.fc.bias'),
                   ('classifier.2.weight', 'classifier.2.bias'),
                   ('classifier.weight', 'classifier.bias'),
                   ('fc.weight', 'fc.bias'),
                   ('head.weight', 'head.bias')]:
        if a in sd and sd[a].shape == (num_classes, in_features):
            fw, fb = sd[a], sd[b_]
            print(f"   [Head] FC   @ '{a}'  {fw.shape}")
            break

    if fw is None:
        for k, v in sd.items():
            if v.ndim == 2 and v.shape == (num_classes, in_features):
                fw = v
                fb = sd.get(k.replace('weight', 'bias'), torch.zeros(num_classes))
                break

    if fw is None:
        raise RuntimeError(f"Cannot find FC weight [{num_classes}, {in_features}]")

    h = ClassifierHead(in_features, num_classes, use_norm=(nw is not None))
    h.fc.weight.data.copy_(fw)
    h.fc.bias.data.copy_(fb)
    if nw is not None:
        h.norm.weight.data.copy_(nw.view(-1))
        h.norm.bias.data.copy_(nb.view(-1))

    h.eval()
    for p in h.parameters():
        p.requires_grad_(False)
    return h.to(device)


# ===========================================================================
# 4. DualManifoldFramework
# ===========================================================================
class DualManifoldFramework(nn.Module):
    def __init__(self, model_name: str = 'resnet18', pretrained_path: str = None,
                 top_k: int = 64):
        super().__init__()
        self.encoder = FrozenEncoder(model_name=model_name,
                                     pretrained_path=pretrained_path)
        self.top_k = top_k

        # 与训练代码对齐: LACRefiner(ch, ch) — num_groups=in_channels
        self.lac = nn.ModuleList([
            LACRefiner(in_channels=ch, num_groups=ch)
            for ch in self.encoder.boundary_channels
        ])
        self.lac_stem = LACRefiner(in_channels=3, num_groups=3)

    @staticmethod
    def _compute_energy(stage_features: List[torch.Tensor]) -> List[torch.Tensor]:
        E_list = []
        for h in stage_features:
            Z_l = h.detach().abs().mean(dim=[2, 3])
            Z_l_total = Z_l.sum(dim=1, keepdim=True).clamp(min=1e-8)
            E_list.append(Z_l / Z_l_total)
        return E_list

    def train(self, mode: bool = True):
        super().train(mode)
        self.encoder.eval()
        return self

    def _invert_single_channel(self, X_req, h_stem, stage_features, stage_idx, channel_idx):
        h_l = stage_features[stage_idx]
        seed = torch.zeros_like(h_l)
        seed[:, channel_idx] = h_l[:, channel_idx]
        g = seed

        inputs_chain = [h_stem] + stage_features

        for b in range(stage_idx, -1, -1):
            vjp = torch.autograd.grad(
                outputs=stage_features[b],
                inputs=inputs_chain[b],
                grad_outputs=g,
                retain_graph=True,
                create_graph=False
            )[0]
            g = self.lac[b](vjp)

        V_tilde_raw = torch.autograd.grad(
            outputs=h_stem,
            inputs=X_req,
            grad_outputs=g,
            retain_graph=True,
            create_graph=False
        )[0]

        return self.lac_stem(V_tilde_raw)

    def invert_stage(self, X_req, h_stem, stage_features, energy_list, stage_idx):
        E_l = energy_list[stage_idx]
        C_l = stage_features[stage_idx].shape[1]
        k = min(self.top_k, C_l)
        idx = E_l.mean(0).topk(k).indices

        X_hat_l = torch.zeros_like(X_req)
        for i in range(k):
            c = idx[i].item()
            V = self._invert_single_channel(X_req, h_stem, stage_features, stage_idx, c)
            X_hat_l = X_hat_l + E_l[:, c].view(-1, 1, 1, 1) * V

        return X_hat_l

    def invert_class_raw(self, X_req, h_stem, stage_features, alpha, top_k=128):
        """
        Raw α-weighted reconstruction WITHOUT normalization.
        Returns: X_class (combined), X_pos (positive α only), X_neg (negative α only)
        """
        deepest = self.encoder.num_stages - 1
        C = stage_features[deepest].shape[1]
        k = min(top_k, C)
        B = X_req.shape[0]

        topk_idx = alpha.detach().abs().mean(0).topk(k).indices
        alpha_sel = alpha[:, topk_idx]

        X_class = torch.zeros_like(X_req)
        X_pos = torch.zeros_like(X_req)
        X_neg = torch.zeros_like(X_req)

        n_pos = (alpha_sel > 0).float().mean(0).sum().item()
        n_neg = (alpha_sel < 0).float().mean(0).sum().item()
        print(f"      top-{k}: ~{n_pos:.0f} pos, ~{n_neg:.0f} neg channels")

        for i in range(k):
            c = topk_idx[i].item()
            V = self._invert_single_channel(X_req, h_stem, stage_features, deepest, c)
            w = alpha_sel[:, i].view(B, 1, 1, 1)
            X_class = X_class + w * V

            w_pos = w.clamp(min=0)
            w_neg = w.clamp(max=0).abs()
            X_pos = X_pos + w_pos * V
            X_neg = X_neg + w_neg * V

        return X_class, X_pos, X_neg


# ===========================================================================
# 5. Visualization Utilities
# ===========================================================================
def denorm(t):
    return torch.clamp(t.detach().cpu() * 0.5 + 0.5, 0, 1).permute(1, 2, 0).numpy()


def adaptive_denorm(t):
    """Per-image adaptive rescaling to [0,1] for display."""
    v = t.detach().cpu().float()
    lo, hi = v.min(), v.max()
    if hi - lo < 1e-8:
        return torch.zeros_like(v).permute(1, 2, 0).numpy()
    return ((v - lo) / (hi - lo)).clamp(0, 1).permute(1, 2, 0).numpy()


# ===========================================================================
# 6. evaluate() — 只依赖 test_loader，与任何数据集解耦
# ===========================================================================
def evaluate(test_loader, model_name, weights_path, lac_ckpt, num_classes,
             n_samples=5, top_stg=512, top_cls=1024,
             pdf_name='class_attribution_posneg.pdf', device=None):
    """
    评估入口函数。

    Args:
        test_loader : DataLoader，__getitem__ 返回 (image_tensor, label_int)
        model_name  : timm 模型名，如 'convnext_base', 'resnet50', 'densenet121'
        weights_path: 骨干 + 分类器权重路径
        lac_ckpt    : LAC checkpoint 路径
        num_classes : 分类器输出类别数
        n_samples   : 展示样本数
        top_stg     : Stage 反演 top-k
        top_cls     : Class attribution top-k
        pdf_name    : 输出 PDF 文件名
        device      : 计算设备，None 则自动选择
    """
    if device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    N = n_samples
    print(f"Device: {device}")

    # ── 构建模型 ──────────────────────────────────────────────────
    model = DualManifoldFramework(
        model_name=model_name, pretrained_path=weights_path, top_k=top_stg
    ).to(device)

    ck = torch.load(lac_ckpt, map_location=device)
    model.lac.load_state_dict(ck['lac'])
    model.lac_stem.load_state_dict(ck['lac_stem'])
    model.eval()
    print("LAC loaded ✓")

    in_feat = model.encoder.stage_channels[-1]
    print(f"\n=> Loading classifier head (in={in_feat}, cls={num_classes})")
    cls_head = load_classifier_head(weights_path, in_feat, num_classes, device)

    # ── 从 loader 取样本 ─────────────────────────────────────────
    images, labels = next(iter(test_loader))
    images = images[:N].to(device)
    labels = labels[:N]

    # ── Forward ───────────────────────────────────────────────────
    print("\n=> Forward pass ...")
    X_req = images.detach().requires_grad_(True)
    with torch.enable_grad():
        h_stem, sf = model.encoder(X_req)
    energy = model._compute_energy(sf)
    ns = model.encoder.num_stages

    # ── Stage Inversions ──────────────────────────────────────────
    recons = {}
    for l in range(ns):
        print(f"   Stage {l} ({model.encoder.stage_channels[l]} ch) ...", flush=True)
        recons[l] = model.invert_stage(X_req, h_stem, sf, energy, l)

    # ── Classifier Gradients → α ──────────────────────────────────
    print("\n=> Classifier gradients ...")
    z = sf[-1].detach().mean(dim=[2, 3])          # [B, C]
    z_req = z.clone().requires_grad_(True)
    with torch.enable_grad():
        logits = cls_head(z_req)                   # [B, K]
    pred = logits.argmax(1)                        # [B]
    sel = logits[range(N), pred]                   # [B]
    grad_z = torch.autograd.grad(sel.sum(), z_req)[0]  # [B, C]
    alpha = grad_z                              # [B, C] — RAW, no normalization

    # ── Class Attribution (unnormalized) ──────────────────────────
    print(f"\n=> Class attribution (top-{top_cls}, unnormalized α) ...")
    X_class, X_pos, X_neg = model.invert_class_raw(
        X_req, h_stem, sf, alpha, top_k=top_cls)

    # ── Statistics ────────────────────────────────────────────────
    print(f"\n=> Pos/Neg channel statistics:")
    for i in range(N):
        pos_e = X_pos[i].detach().cpu().abs().mean().item()
        neg_e = X_neg[i].detach().cpu().abs().mean().item()
        cls_e = X_class[i].detach().cpu().abs().mean().item()
        print(f"   Sample {i}: pos={pos_e:.4f}, neg={neg_e:.4f}, combined={cls_e:.4f}")

    # ==============================================================
    #  PDF (N rows × (1 + ns + 3) columns)
    # ==============================================================
    print(f"\n=> Rendering PDF: {pdf_name}")
    n_cols = 1 + ns + 3
    fig, axes = plt.subplots(N, n_cols, figsize=(3.2 * n_cols, 3.6 * N))
    if N == 1:
        axes = [axes]

    headers = (['Original']
               + [f'Stage {l} Inversion' for l in range(ns)]
               + ['Class Recon (combined)',
                  'Positive α Channels',
                  'Negative α Channels'])

    for i in range(N):
        gt = labels[i].item()
        pk = pred[i].item()
        orig = denorm(images[i])

        # col 0 — original
        axes[i][0].imshow(orig)
        t = (f'{headers[0]}\n' if i == 0 else '') + f'GT: {gt}'
        axes[i][0].set_title(t, fontsize=7,
                              fontweight='bold' if i == 0 else 'normal')
        axes[i][0].axis('off')

        # cols 1..ns — stage inversions
        for l in range(ns):
            axes[i][l + 1].imshow(denorm(recons[l][i]))
            axes[i][l + 1].set_title(
                headers[l + 1] if i == 0 else '', fontsize=7,
                fontweight='bold' if i == 0 else 'normal')
            axes[i][l + 1].axis('off')

        # col: class reconstruction combined (adaptive rescale)
        col_cr = ns + 1
        axes[i][col_cr].imshow(adaptive_denorm(X_class[i]))
        t = (f'{headers[col_cr]}\n' if i == 0 else '') + f'Pred: {pk}'
        axes[i][col_cr].set_title(t, fontsize=7,
                                   fontweight='bold' if i == 0 else 'normal')
        axes[i][col_cr].axis('off')

        # col: positive α channels
        col_pos = ns + 2
        axes[i][col_pos].imshow(adaptive_denorm(X_pos[i]))
        axes[i][col_pos].set_title(
            headers[col_pos] if i == 0 else '', fontsize=7,
            fontweight='bold' if i == 0 else 'normal')
        axes[i][col_pos].axis('off')

        # col: negative α channels
        col_neg = ns + 3
        axes[i][col_neg].imshow(adaptive_denorm(X_neg[i]))
        axes[i][col_neg].set_title(
            headers[col_neg] if i == 0 else '', fontsize=7,
            fontweight='bold' if i == 0 else 'normal')
        axes[i][col_neg].axis('off')

    plt.tight_layout(pad=0.3)
    fig.subplots_adjust(hspace=0.18, wspace=0.06)

    with PdfPages(pdf_name) as pdf:
        pdf.savefig(fig, dpi=200, bbox_inches='tight')
    plt.close()

    print(f"\n=> Done!  '{pdf_name}'")


# ===========================================================================
# 7. __main__ — 所有配置集中在这里，换模型/数据集只改这个区块
# ===========================================================================
if __name__ == '__main__':

    torch.manual_seed(42)

    # =================================================================
    #  模型配置 — 切换模型只需改这四行
    # =================================================================
    MODEL_NAME   = 'convnext_base'                            # timm 模型名
    WEIGHTS_PATH = 'weights/convnext_base_dogs.pth'         # 骨干 + 分类头权重
    LAC_CKPT     = 'dogs_lac_convnext_base_step010000.pth'  # LAC checkpoint
    NC           = 120                                        # 类别数

    # =================================================================
    #  数据集配置 — 切换数据集只需改这里，提供 test_loader 即可
    #  要求: DataLoader 的 dataset.__getitem__ 返回 (image_tensor, label_int)
    # =================================================================
    tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5,) * 3, (0.5,) * 3)
    ])

    N_SAMPLES  = 5
    BATCH_SIZE = N_SAMPLES  # batch_size >= n_samples


    from torchvision.datasets.utils import download_and_extract_archive
    from torch.utils.data import Dataset
    from scipy.io import loadmat
    from PIL import Image


    # Data loading
    class StanfordDogs(Dataset):
        def __init__(self, root='./data', train=True, transform=None, download=False):
            self.root = os.path.join(root, 'StanfordDogs')
            self.train = train
            self.transform = transform

            if download:
                self.download()

            # 读取train/test列表
            split_file = os.path.join(self.root, 'train_list.mat' if train else 'test_list.mat')
            mat_data = loadmat(split_file)
            file_list = mat_data['file_list']
            labels = mat_data['labels'].flatten() - 1  # MATLAB索引从1开始,转为0开始

            self.data = []
            for i, file_path in enumerate(file_list):
                img_path = os.path.join(self.root, 'Images', file_path[0][0])
                self.data.append((img_path, labels[i]))

        def download(self):
            if os.path.exists(self.root):
                print("Dataset already exists, skipping download.")
                return

            # 下载图片
            images_url = "http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar"
            download_and_extract_archive(images_url, self.root, filename='images.tar')

            # 下载列表文件
            lists_url = "http://vision.stanford.edu/aditya86/ImageNetDogs/lists.tar"
            download_and_extract_archive(lists_url, self.root, filename='lists.tar')

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            img_path, label = self.data[idx]
            image = Image.open(img_path).convert('RGB')

            if self.transform:
                image = self.transform(image)

            return image, label


    # Stanford Dogs
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = StanfordDogs(
        root='../data',
        train=True,
        download=True,
        transform=transform
    )
    train_loader = torch.utils.data.DataLoader(
        trainset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2
    )

    dataset = StanfordDogs(
        root='../data',
        train=False,
        download=True,
        transform=transform
    )




    test_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    # =================================================================
    #  启动评估
    # =================================================================
    evaluate(
        test_loader  = test_loader,
        model_name   = MODEL_NAME,
        weights_path = WEIGHTS_PATH,
        lac_ckpt     = LAC_CKPT,
        num_classes  = NC,
        n_samples    = N_SAMPLES,
        top_stg      = 512,
        top_cls      = 1024,
        pdf_name     = 'class_attribution_posneg.pdf',
    )