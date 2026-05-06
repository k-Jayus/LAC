"""
Per-Channel Spatial Basis Visualization
========================================
For a single input image, visualize the LAC-refined spatial basis
V_tilde_{l,c} for EVERY channel at a selected encoder stage.

Output: Multi-page PDF
  - Page 1: Original image + full stage reconstruction + info
  - Pages 2+: Grid of per-channel V_tilde visualizations,
              sorted by energy (most important first)

Usage:
  - Set STAGE_IDX to choose which stage to visualize (0,1,2,3 or -1 for last)
  - Change `ds = ...` to switch dataset
"""

import os, re, math, torch, torch.nn as nn, torch.nn.functional as F
import timm
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from typing import List


# ================================================================
#  1. LACRefiner
# ================================================================
class LACRefiner(nn.Module):
    def __init__(self, in_channels: int, num_groups: int = 32):
        super().__init__()
        self.num_groups = num_groups
        self.eps = 1e-5
        self.log_gamma = nn.Parameter(torch.zeros(in_channels))
        self.beta = nn.Parameter(torch.zeros(in_channels))

    def forward(self, vjp):
        return F.group_norm(vjp, self.num_groups,
                            weight=self.log_gamma.exp(),
                            bias=self.beta - self.beta.mean().detach(),
                            eps=self.eps)


# ================================================================
#  2. FrozenEncoder
# ================================================================
class FrozenEncoder(nn.Module):
    def __init__(self, model_name='resnet18', pretrained_path=None):
        super().__init__()
        _p = timm.create_model(model_name, pretrained=False, features_only=True)
        _nf = len(_p.feature_info); del _p
        self.bb = timm.create_model(model_name, pretrained=(pretrained_path is None),
                                    features_only=True, out_indices=tuple(range(_nf)))
        self._stem_out = None
        sm, sp = self._find_stem()
        if sm is not None:
            sm.register_forward_hook(self._hook)
            self._has_stem = True
            print(f"   [Stem] Hook at '{sp}'")
        else:
            self._has_stem = False
        if pretrained_path and os.path.isfile(pretrained_path):
            self._load(pretrained_path)
        d = torch.zeros(1, 3, 224, 224)
        with torch.no_grad():
            fs = self.bb(d)
        if self._has_stem:
            self.stem_ch = self._stem_out.shape[1]
            self.stage_ch = [f.shape[1] for f in fs]
        else:
            self.stem_ch = fs[0].shape[1]
            self.stage_ch = [f.shape[1] for f in fs[1:]]
        self.num_stages = len(self.stage_ch)
        self.boundary_ch = [self.stem_ch] + self.stage_ch[:-1]
        print(f"   Stem={self.stem_ch}  Stages={self.stage_ch}")
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

    def _load(self, path):
        ck = torch.load(path, map_location='cpu')
        sd = ck.get('state_dict', ck)
        ks = set(self.bb.state_dict().keys()); nw = {}
        for k, v in sd.items():
            k = k.replace('module.', '')
            if any(t in k for t in ('fc', 'classifier', 'head')):
                continue
            if k in ks: nw[k] = v; continue
            if k.startswith('features.'):
                kf = 'features_' + k[len('features.'):]
                if kf in ks: nw[kf] = v; continue
            kf = re.sub(r'^(\w+)\.(\d+)', r'\1_\2', k)
            if kf in ks: nw[kf] = v; continue
            kf, vf = self._rcv(k, v, ks)
            if kf: nw[kf] = vf; continue
        r = self.bb.load_state_dict(nw, strict=False)
        print(f"   Backbone: loaded {len(nw)}, missing {len(r.missing_keys)}")
        if r.missing_keys:
            raise RuntimeError(f"Missing: {r.missing_keys[:5]}")

    @staticmethod
    def _rcv(k, v, ks):
        if not k.startswith('features.'): return None, v
        p = k.split('.')
        if len(p) < 3: return None, v
        try: mi = int(p[1])
        except ValueError: return None, v
        rp = p[2:]
        if mi == 0:
            kf = f'stem_{rp[0]}' + (f'.{".".join(rp[1:])}' if len(rp) > 1 else '')
            return (kf, v) if kf in ks else (None, v)
        elif mi % 2 == 1:
            si = (mi-1)//2; bi = rp[0]
            if len(rp) >= 3 and rp[1] == 'block':
                try: sub = int(rp[2])
                except ValueError: return None, v
                r = '.'.join(rp[3:])
                sm = {0:'conv_dw', 2:'norm', 3:'mlp.fc1', 5:'mlp.fc2'}
                if sub in sm:
                    kf = f'stages_{si}.blocks.{bi}.{sm[sub]}' + (f'.{r}' if r else '')
                    return (kf, v) if kf in ks else (None, v)
            elif len(rp) >= 2 and rp[1] == 'layer_scale':
                kf = f'stages_{si}.blocks.{bi}.gamma'
                return (kf, v.view(-1)) if kf in ks else (None, v)
        elif mi % 2 == 0 and mi > 0:
            si = mi//2; di = rp[0]; r = '.'.join(rp[1:])
            kf = f'stages_{si}.downsample.{di}' + (f'.{r}' if r else '')
            return (kf, v) if kf in ks else (None, v)
        return None, v

    def _hook(self, m, i, o): self._stem_out = o
    def _find_stem(self):
        for n in ('stem','patch_embed','conv_stem'):
            m = getattr(self.bb, n, None)
            if m: return m, n
        if hasattr(self.bb, 'stem_1'): return self.bb.stem_1, 'stem_1'
        if hasattr(self.bb, 'stem_0'): return self.bb.stem_0, 'stem_0'
        return None, None

    def forward(self, x):
        fs = self.bb(x)
        if self._has_stem: return self._stem_out, list(fs)
        return fs[0], list(fs[1:])


# ================================================================
#  3. DualManifoldFramework (visualization-only)
# ================================================================
class DualManifoldFramework(nn.Module):
    def __init__(self, model_name='resnet18', pretrained_path=None):
        super().__init__()
        self.encoder = FrozenEncoder(model_name=model_name,
                                     pretrained_path=pretrained_path)
        self.lac = nn.ModuleList([
            LACRefiner(ch, ch) for ch in self.encoder.boundary_ch
        ])
        self.lac_stem = LACRefiner(3, 3)

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()
        return self

    @staticmethod
    def _compute_energy(sf):
        out = []
        for h in sf:
            Z = h.detach().abs().mean(dim=[2, 3])
            out.append(Z / Z.sum(1, keepdim=True).clamp(min=1e-8))
        return out

    def invert_single_channel(self, X_req, h_stem, sf, si, ci):
        """Compute V_tilde_{l,c} for stage si, channel ci."""
        seed = torch.zeros_like(sf[si])
        seed[:, ci] = sf[si][:, ci]
        g = seed
        chain = [h_stem] + sf
        for b in range(si, -1, -1):
            vjp = torch.autograd.grad(
                sf[b], chain[b], g,
                retain_graph=True, create_graph=False)[0]
            g = self.lac[b](vjp)
        raw = torch.autograd.grad(
            h_stem, X_req, g,
            retain_graph=True, create_graph=False)[0]
        return self.lac_stem(raw)

    def invert_stage(self, X_req, h_stem, sf, elist, si, top_k=512):
        """Full stage reconstruction (energy-weighted top-k)."""
        E = elist[si]; C = sf[si].shape[1]; k = min(top_k, C)
        idx = E.mean(0).topk(k).indices
        out = torch.zeros_like(X_req)
        for i in range(k):
            c = idx[i].item()
            V = self.invert_single_channel(X_req, h_stem, sf, si, c)
            out = out + E[:, c].view(-1, 1, 1, 1) * V
        return out


# ================================================================
#  4. Display helpers
# ================================================================
def denorm(t):
    """[-1,1] → [0,1] RGB numpy"""
    return torch.clamp(t.detach().cpu()*0.5+0.5, 0, 1).permute(1,2,0).numpy()

def normalize_basis(t):
    """Per-image min-max → [0,1] for spatial basis display."""
    v = t.detach().cpu().float()
    lo, hi = v.min(), v.max()
    if hi - lo < 1e-8:
        return torch.zeros_like(v).permute(1, 2, 0).numpy()
    return ((v - lo) / (hi - lo)).clamp(0, 1).permute(1, 2, 0).numpy()


# ================================================================
#  5. Main
# ================================================================
def main():
    # ── Configuration ─────────────────────────────────────────────
    MODEL      = 'convnext_base'
    WEIGHTS    = 'weights/convnext_base_dogs.pth'
    LAC_CKPT   = 'dogs_lac_convnext_base_step010000.pth'
    STAGE_IDX  = -1        # which stage to visualize (-1 = last)
    SAMPLE_IDX = 11         # which test sample to use
    COLS       = 10        # columns per page
    ROWS       = 10        # rows per page
    DEVICE     = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(42)
    print(f"Device: {DEVICE}")

    # ── Load model ────────────────────────────────────────────────
    model = DualManifoldFramework(
        model_name=MODEL, pretrained_path=WEIGHTS
    ).to(DEVICE)

    print(f"\n=> Loading LAC: {LAC_CKPT}")
    ck = torch.load(LAC_CKPT, map_location=DEVICE)
    model.lac.load_state_dict(ck['lac'])
    model.lac_stem.load_state_dict(ck['lac_stem'])
    model.eval()
    print("   LAC loaded ✓")

    # ── Resolve stage index ───────────────────────────────────────
    ns = model.encoder.num_stages
    if STAGE_IDX < 0:
        STAGE_IDX = ns + STAGE_IDX
    assert 0 <= STAGE_IDX < ns, f"STAGE_IDX={STAGE_IDX} out of range [0, {ns-1}]"
    C = model.encoder.stage_ch[STAGE_IDX]
    PDF = f'channel_vis_stage{STAGE_IDX}_{MODEL}.pdf'
    print(f"\n=> Visualizing Stage {STAGE_IDX}: {C} channels")
    print(f"   Grid: {COLS}×{ROWS} = {COLS*ROWS} per page, "
          f"{math.ceil(C / (COLS*ROWS))+1} pages total")

    
    from torch.utils.data import DataLoader
    import  torchvision
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

        return train_loader, val_dataset

    train_loader, ds = get_imagenet_dataloaders(
        data_dir='/home/ubuntu/sukaixiang/imagenet',
        batch_size=1,
        num_workers=8
    )



    # 其他示例:
    # ds = datasets.CIFAR10(root='./data', train=False, transform=tf, download=True)
    # ds = datasets.CIFAR100(root='./data', train=False, transform=tf, download=True)
    # ds = datasets.STL10(root='./data', split='test', transform=tf, download=True)
    # ds = datasets.ImageFolder(root='./data/my_custom_dataset/test', transform=tf)
    # ============================================================

    img, label = ds[SAMPLE_IDX]
    img = img.unsqueeze(0).to(DEVICE)  # [1, 3, 224, 224]
    print(f"   Sample {SAMPLE_IDX}: label={label}")

    # ── Forward pass ──────────────────────────────────────────────
    print("\n=> Forward pass ...")
    X_req = img.detach().requires_grad_(True)
    with torch.enable_grad():
        h_stem, sf = model.encoder(X_req)
    energy = model._compute_energy(sf)
    E = energy[STAGE_IDX][0]   # [C]  (batch dim squeezed)

    # ── Sort channels by energy (descending) ──────────────────────
    sorted_idx = E.argsort(descending=True)   # [C]
    sorted_E   = E[sorted_idx]                # [C]

    # ── Stage reconstruction (for reference) ──────────────────────
    print("=> Computing full stage reconstruction ...")
    X_hat = model.invert_stage(X_req, h_stem, sf, energy, STAGE_IDX, top_k=512)

    # ══════════════════════════════════════════════════════════════
    #  PDF Generation (page by page to control memory)
    # ══════════════════════════════════════════════════════════════
    print(f"\n=> Generating PDF: {PDF}")
    cpp = COLS * ROWS   # channels per page

    with PdfPages(PDF) as pdf:
        # ── Page 0: Overview ──────────────────────────────────────
        fig0, axes0 = plt.subplots(1, 3, figsize=(15, 5))
        fig0.suptitle(
            f'Per-Channel Visualization  |  {MODEL}  |  '
            f'Stage {STAGE_IDX}  ({C} channels)\n'
            f'Sample {SAMPLE_IDX}: label {label}',
            fontsize=13, fontweight='bold')

        axes0[0].imshow(denorm(img[0]))
        axes0[0].set_title('Original Image', fontsize=11)
        axes0[0].axis('off')

        axes0[1].imshow(denorm(X_hat[0]))
        axes0[1].set_title(f'Stage {STAGE_IDX} Reconstruction\n(top-512 channels)',
                           fontsize=11)
        axes0[1].axis('off')

        # energy distribution bar chart
        top_show = min(50, C)
        bars_e = sorted_E[:top_show].detach().cpu().numpy()
        bars_c = sorted_idx[:top_show].detach().cpu().numpy()
        axes0[2].bar(range(top_show), bars_e, color='steelblue', width=0.8)
        axes0[2].set_xlabel('Rank (sorted by energy)', fontsize=9)
        axes0[2].set_ylabel('$E_{l,c}$', fontsize=9)
        axes0[2].set_title(f'Energy Distribution (top-{top_show})', fontsize=11)
        axes0[2].tick_params(labelsize=7)
        if top_show <= 30:
            axes0[2].set_xticks(range(top_show))
            axes0[2].set_xticklabels(
                [f'{bars_c[i]}' for i in range(top_show)],
                rotation=90, fontsize=5)

        plt.tight_layout(rect=[0, 0, 1, 0.92])
        pdf.savefig(fig0, dpi=150, bbox_inches='tight')
        plt.close(fig0)

        # ── Pages 1+: Per-channel grids ──────────────────────────
        n_pages = math.ceil(C / cpp)
        for page in range(n_pages):
            start = page * cpp
            end   = min(start + cpp, C)
            n_this = end - start
            n_rows = math.ceil(n_this / COLS)

            fig, axes = plt.subplots(
                n_rows, COLS,
                figsize=(2.0 * COLS, 2.3 * n_rows))
            fig.suptitle(
                f'Stage {STAGE_IDX} — Channels {start+1}–{end} of {C}  '
                f'(sorted by energy)',
                fontsize=11, fontweight='bold')

            # flatten axes for easy indexing
            if n_rows == 1:
                axes = [axes]
            ax_flat = [axes[r][c] if COLS > 1 else axes[r]
                       for r in range(n_rows) for c in range(COLS)]

            # hide all axes first
            for ax in ax_flat:
                ax.axis('off')

            for i in range(n_this):
                rank = start + i
                orig_c = sorted_idx[rank].item()
                e_val  = sorted_E[rank].item()

                if i % 20 == 0:
                    print(f"   Page {page+1}/{n_pages} | "
                          f"Channel {rank+1}/{C}  (orig #{orig_c}, "
                          f"E={e_val:.5f})", flush=True)

                # compute spatial basis V_tilde for this channel
                with torch.no_grad():
                    # need grad for autograd.grad, but no param grad
                    pass
                with torch.enable_grad():
                    V = model.invert_single_channel(
                        X_req, h_stem, sf, STAGE_IDX, orig_c)

                # display
                ax_flat[i].imshow(normalize_basis(V[0]))
                ax_flat[i].set_title(
                    f'#{orig_c}  E={e_val:.4f}',
                    fontsize=5, pad=1)

            plt.tight_layout(rect=[0, 0, 1, 0.95])
            fig.subplots_adjust(hspace=0.35, wspace=0.08)
            pdf.savefig(fig, dpi=150, bbox_inches='tight')
            plt.close(fig)

    print(f"\n=> Done!  Saved to '{PDF}'")
    print(f"   Total pages: {n_pages + 1}")
    print(f"   Channels visualized: {C}")


if __name__ == '__main__':
    main()