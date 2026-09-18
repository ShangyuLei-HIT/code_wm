#!/usr/bin/env python
"""VAE 全流程教学脚本(standalone,单文件,无仓库依赖)。

一条命令跑通变分自编码器(Variational Autoencoder, Kingma & Welling, 2013)的
完整流程:

    数据准备 → 模型定义 → 训练(ELBO)→ 测试评估 → checkpoint/指标落盘
    → 可视化(损失曲线 / 重建 / 先验采样 / 隐空间结构 / 潜变量插值 / 逐维 KL)

核心数学
--------
VAE 学习一个生成模型 p(x) = ∫ p(x|z) p(z) dz,其中先验 p(z) = N(0, I),
并引入编码器 q(z|x) 近似难以求解的后验 p(z|x)。训练目标是最小化负 ELBO:

    L(x) = E_{q(z|x)}[ -log p(x|z) ]            ← 重建项(本脚本用逐像素 BCE)
         + β · KL( q(z|x) ‖ N(0, I) )           ← 正则项(β=1 即标准 VAE;
                                                   β>1 即 β-VAE,倾向更解耦的表示。
                                                   β≠1 时优化的严格说是 KL 加权
                                                   目标而非 ELBO,习惯上仍称 ELBO)

对高斯编码器 q(z|x) = N(μ(x), diag(σ²(x))),KL 项有闭式解:

    KL = 0.5 · Σ_j ( μ_j² + σ_j² − log σ_j² − 1 )

重参数化技巧(reparameterization trick)把采样改写成可导形式:

    z = μ + σ ⊙ ε,   ε ~ N(0, I)

使得梯度能穿过随机节点回传到编码器。

用法
----
    # 默认:MNIST + 2 维隐空间(经典可视化配置),20 epochs,结果存 ./vae_out
    python scripts/examples/vae_full_pipeline.py

    # 自定义输出目录 / 隐空间维度 / β-VAE
    python scripts/examples/vae_full_pipeline.py --out-dir runs/vae_d16 \
        --latent-dim 16 --beta 4.0 --epochs 30

    # 快速冒烟(每 epoch 只跑 20 个 batch)
    python scripts/examples/vae_full_pipeline.py --epochs 1 --max-batches-per-epoch 20

仅依赖:torch、torchvision、matplotlib、numpy(sklearn 可选,仅影响 t-SNE 图)。
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass, asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 无显示环境也能出图,必须在 pyplot 之前设置
# 注意:图内文字一律用英文。无显示的服务器普遍没装 CJK 字体,matplotlib 默认
# 字体链渲染中文会得到一排方框(Glyph missing);注释与控制台输出不受影响。
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# --------------------------------------------------------------------------- #
# 0. 配置                                                                     #
# --------------------------------------------------------------------------- #


@dataclass
class Config:
    """一次运行的全部超参数(会原样写进 out_dir/config.json 便于复现)。"""

    data_root: str = "./data" # MNIST 存放目录(不存在则自动下载)
    out_dir: str = "./vae_out" # 输出根目录
    epochs: int = 20
    batch_size: int = 128
    lr: float = 1e-3
    latent_dim: int = 2 # 2 维时有经典的隐空间网格/散点图;更大维度自动改用 PCA/t-SNE
    beta: float = 1.0 # KL 权重;>1 即 β-VAE
    seed: int = 0
    num_workers: int = 2
    max_batches_per_epoch: int = 0 # >0 时每 epoch 截断(冒烟测试用;0 = 完整)
    device: str = "auto" # auto | cuda | cpu | mps
    eval_samples: int = 2000 # 隐空间散点图用的测试样本数


def parse_args() -> Config:
    p = argparse.ArgumentParser(
        description="VAE 全流程教学脚本:训练 + 评估 + 可视化",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-root", default=Config.data_root, help="MNIST 数据目录(自动下载)")
    p.add_argument("--out-dir", default=Config.out_dir, help="输出目录(图/指标/checkpoint)")
    p.add_argument("--epochs", type=int, default=Config.epochs)
    p.add_argument("--batch-size", type=int, default=Config.batch_size)
    p.add_argument("--lr", type=float, default=Config.lr)
    p.add_argument("--latent-dim", type=int, default=Config.latent_dim)
    p.add_argument("--beta", type=float, default=Config.beta, help="KL 权重,>1 为 β-VAE")
    p.add_argument("--seed", type=int, default=Config.seed)
    p.add_argument("--num-workers", type=int, default=Config.num_workers)
    p.add_argument(
        "--max-batches-per-epoch",
        type=int,
        default=Config.max_batches_per_epoch,
        help=">0 时每 epoch 只跑这么多 batch(冒烟测试);0 表示完整 epoch",
    )
    p.add_argument("--device", default=Config.device, choices=["auto", "cuda", "cpu", "mps"])
    p.add_argument("--eval-samples", type=int, default=Config.eval_samples)
    cfg = Config(**vars(p.parse_args()))
    if cfg.latent_dim < 2:
        # 隐空间散点/网格都依赖二维投影,1 维会在可视化阶段越界
        p.error(f"--latent-dim 需要 >= 2(可视化依赖二维投影),当前为 {cfg.latent_dim}")
    return cfg


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------- #
# 1. 数据                                                                     #
# --------------------------------------------------------------------------- #


def build_loaders(cfg: Config) -> tuple[DataLoader, DataLoader]:
    """MNIST 的 train/test DataLoader。

    像素保持 [0,1](ToTensor),解码器输出 sigmoid + 逐像素 BCE,这是最经典、
    也最容易和 ELBO 公式对上号的 VAE 教学配置。
    """
    tf = transforms.ToTensor()
    train_ds = datasets.MNIST(cfg.data_root, train=True, download=True, transform=tf)
    test_ds = datasets.MNIST(cfg.data_root, train=False, download=True, transform=tf)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True, # 丢掉最后不满的 batch,让 loss 曲线更平稳
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    return train_loader, test_loader


# --------------------------------------------------------------------------- #
# 2. 模型                                                                     #
# --------------------------------------------------------------------------- #


class ConvVAE(nn.Module):
    """MNIST 尺寸(1×28×28)的卷积 VAE。

    编码器把图像压成 μ 与 log σ² 两路仿射输出;解码器从 z 重建图像。
    注意用 logvar 而非 σ 直接作为网络输出,保证 σ = exp(0.5·logvar) > 0,
    且数值上更稳定。
    """

    def __init__(self, latent_dim: int = 2):
        super().__init__()
        self.latent_dim = latent_dim

        # 28x28 → 14x14 → 7x7 → 4x4
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),  # -> 32x14x14
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # -> 64x7x7
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),  # -> 64x4x4
            nn.ReLU(inplace=True),
            nn.Flatten(),  # -> 1024
            nn.Linear(64 * 4 * 4, 256),
            nn.ReLU(inplace=True),
        )
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)

        self.fc_dec = nn.Linear(latent_dim, 256)
        self.decoder = nn.Sequential(
            nn.Linear(256, 64 * 4 * 4),
            nn.ReLU(inplace=True),
            nn.Unflatten(1, (64, 4, 4)),
            # 4x4 → 7x7 → 14x14 → 28x28(与编码器严格互逆)
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1),  # -> 64x7x7
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # -> 32x14x14
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),  # -> 1x28x28
            nn.Sigmoid(),  # 像素概率 ∈ (0,1),与 BCE 目标匹配
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """z = μ + σ ⊙ ε,ε ~ N(0,I)。

        分支条件是「当前是否在建梯度」,与 train()/eval() 模式无关:训练
        (有梯度)时采样以引入随机性;包在 torch.no_grad() 里的评估直接返回
        μ(确定性,也是对 ELBO 中 E 项的单样本确定性近似)。
        """
        if not torch.is_grad_enabled():
            return mu
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.fc_dec(z))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


# --------------------------------------------------------------------------- #
# 3. 损失与评估                                                               #
# --------------------------------------------------------------------------- #


def vae_loss(
    recon: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor, beta: float
) -> tuple[torch.Tensor, dict[str, float]]:
    """负 ELBO(标量,待 backward)+ 分项统计(每样本均值)。

    重建项:逐像素 BCE,对 batch 内样本求和再除以 batch 大小;
    正则项:高斯闭式 KL,同样折算到每样本。
    """
    batch = x.size(0)
    recon_bce = F.binary_cross_entropy(recon, x, reduction="sum") / batch
    kl = (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())) / batch
    elbo = recon_bce + beta * kl
    return elbo, {"elbo": elbo.item(), "recon": recon_bce.item(), "kl": kl.item()}


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, beta: float) -> dict[str, float]:
    """整个 loader 上的平均每样本损失(no grad + 评估期用 μ 代替采样)。"""
    model.eval()
    totals = {"elbo": 0.0, "recon": 0.0, "kl": 0.0}
    n = 0
    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        recon, mu, logvar = model(x)
        _, parts = vae_loss(recon, x, mu, logvar, beta)
        bs = x.size(0)
        for k in totals:
            totals[k] += parts[k] * bs
        n += bs
    return {k: v / max(n, 1) for k, v in totals.items()}


# --------------------------------------------------------------------------- #
# 4. 训练循环                                                                 #
# --------------------------------------------------------------------------- #


def train(cfg: Config, model: nn.Module, train_loader: DataLoader, test_loader: DataLoader,
          device: torch.device) -> dict:
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    history: dict[str, list[float]] = {
        "train_elbo": [], "train_recon": [], "train_kl": [],
        "test_elbo": [], "test_recon": [], "test_kl": [],
    }
    best_test = float("inf")
    ckpt_dir = Path(cfg.out_dir) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running = {"elbo": 0.0, "recon": 0.0, "kl": 0.0}
        seen = 0
        for step, (x, _) in enumerate(train_loader, start=1):
            if cfg.max_batches_per_epoch and step > cfg.max_batches_per_epoch:
                break
            x = x.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            recon, mu, logvar = model(x)
            loss, parts = vae_loss(recon, x, mu, logvar, cfg.beta)
            loss.backward()
            opt.step()

            bs = x.size(0)
            for k in running:
                running[k] += parts[k] * bs
            seen += bs
            if step % 50 == 0:
                print(f"  epoch {epoch:03d} step {step:04d} "
                      f"elbo={parts['elbo']:.2f} recon={parts['recon']:.2f} kl={parts['kl']:.2f}")

        train_avg = {k: v / max(seen, 1) for k, v in running.items()}
        test_avg = evaluate(model, test_loader, device, cfg.beta)
        for k, v in train_avg.items():
            history[f"train_{k}"].append(v)
        for k, v in test_avg.items():
            history[f"test_{k}"].append(v)
        print(f"[epoch {epoch:03d}/{cfg.epochs}] "
              f"train elbo={train_avg['elbo']:.2f} (recon={train_avg['recon']:.2f}, kl={train_avg['kl']:.2f}) | "
              f"test elbo={test_avg['elbo']:.2f} (recon={test_avg['recon']:.2f}, kl={test_avg['kl']:.2f})")

        state = {
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "epoch": epoch,
            "config": asdict(cfg),
            "history": history,
        }
        torch.save(state, ckpt_dir / "last.pt")
        if test_avg["elbo"] < best_test:
            best_test = test_avg["elbo"]
            torch.save(state, ckpt_dir / "best.pt")

    return {"history": history, "best_test_elbo": best_test}


# --------------------------------------------------------------------------- #
# 5. 可视化                                                                   #
# --------------------------------------------------------------------------- #


def _save(fig: plt.Figure, out_dir: Path, name: str) -> Path:
    path = out_dir / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  图已保存: {path}")
    return path


def plot_loss_curves(history: dict[str, list[float]], out_dir: Path) -> Path:
    """训练/测试损失曲线:左图重建项与 KL,右图总 ELBO。"""
    epochs = range(1, len(history["train_elbo"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, history["train_recon"], label="train recon (BCE)")
    axes[0].plot(epochs, history["test_recon"], "--", label="test recon (BCE)")
    axes[0].plot(epochs, history["train_kl"], label="train KL")
    axes[0].plot(epochs, history["test_kl"], "--", label="test KL")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("per-sample loss")
    axes[0].set_title("重建项 vs KL 正则项")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, history["train_elbo"], label="train ELBO (neg)")
    axes[1].plot(epochs, history["test_elbo"], "--", label="test ELBO (neg)")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("negative ELBO")
    axes[1].set_title("总损失")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    fig.suptitle("训练曲线")
    return _save(fig, out_dir, "loss_curves.png")


@torch.no_grad()
def plot_reconstructions(model: nn.Module, loader: DataLoader, device: torch.device,
                         out_dir: Path, n: int = 8) -> Path:
    """上排原图、下排重建:直观看重建质量。"""
    model.eval()
    x, _ = next(iter(loader))
    x = x[:n].to(device)
    recon = model(x)[0].cpu()
    x = x.cpu()

    fig, axes = plt.subplots(2, n, figsize=(1.4 * n, 3.0))
    for i in range(n):
        axes[0, i].imshow(x[i, 0], cmap="gray", vmin=0, vmax=1)
        axes[0, i].set_title("原始" if i == 0 else "")
        axes[1, i].imshow(recon[i, 0], cmap="gray", vmin=0, vmax=1)
        axes[1, i].set_title("重建" if i == 0 else "")
        for ax in (axes[0, i], axes[1, i]):
            ax.axis("off")
    fig.suptitle("重建对比(上:输入,下:重建)")
    return _save(fig, out_dir, "reconstructions.png")


@torch.no_grad()
def plot_prior_samples(model: nn.Module, device: torch.device, out_dir: Path,
                       n: int = 64) -> Path:
    """从先验 N(0,I) 采 z 直接解码:检验生成能力与先验-聚合后验匹配程度。"""
    model.eval()
    z = torch.randn(n, model.latent_dim, device=device)
    samples = model.decode(z).cpu()

    side = int(np.ceil(np.sqrt(n)))
    fig, axes = plt.subplots(side, side, figsize=(side * 1.1, side * 1.1))
    for idx, ax in enumerate(axes.flat):
        if idx < n:
            ax.imshow(samples[idx, 0], cmap="gray", vmin=0, vmax=1)
        ax.axis("off")
    fig.suptitle("先验采样 z ~ N(0,I) 后解码")
    return _save(fig, out_dir, "prior_samples.png")


@torch.no_grad()
def collect_latents(model: nn.Module, loader: DataLoader, device: torch.device,
                    max_samples: int) -> tuple[np.ndarray, np.ndarray]:
    """收集测试集的 μ(评估期 reparameterize 直接返回 μ)与标签。"""
    model.eval()
    mus, labels = [], []
    for x, y in loader:
        mu, _ = model.encode(x.to(device, non_blocking=True))
        mus.append(mu.cpu().numpy())
        labels.append(y.numpy())
        if sum(m.shape[0] for m in mus) >= max_samples:
            break
    return np.concatenate(mus)[:max_samples], np.concatenate(labels)[:max_samples]


def _pca_2d(data: np.ndarray) -> np.ndarray:
    """无 sklearn 依赖的 PCA:中心化 + SVD 取前两个主成分。"""
    centered = data - data.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return centered @ vt[:2].T


def plot_latent_scatter(model: nn.Module, loader: DataLoader, device: torch.device,
                        out_dir: Path, max_samples: int, latent_dim: int) -> Path:
    """隐空间散点(按数字类别着色)。

    latent_dim=2 时直接画 (z1, z2);更高维先 PCA 投影,装有 sklearn 时再补一张
    t-SNE(局部结构更清楚)。
    """
    mus, labels = collect_latents(model, loader, device, max_samples)
    n = len(labels)

    projections: list[tuple[str, np.ndarray]] = []
    if latent_dim == 2:
        projections.append(("隐空间 (z1, z2)", mus))
    else:
        projections.append(("PCA 投影", _pca_2d(mus)))
        try:
            from sklearn.manifold import TSNE

            tsne = TSNE(n_components=2, init="pca", random_state=0, perplexity=30)
            projections.append(("t-SNE 投影", tsne.fit_transform(mus)))
        except ImportError:
            print("  [提示] 未安装 sklearn,跳过 t-SNE 图(pip install scikit-learn 可开启)")

    fig, axes = plt.subplots(1, len(projections), figsize=(6.5 * len(projections), 5.5))
    if len(projections) == 1:
        axes = [axes]
    cmap = plt.get_cmap("tab10")
    for ax, (title, proj) in zip(axes, projections):
        for digit in range(10):
            m = labels == digit
            ax.scatter(proj[m, 0], proj[m, 1], s=4, color=cmap(digit), label=str(digit),
                       alpha=0.6)
        ax.set_title(f"{title}(n={n})")
        ax.set_xlabel("分量 1")
        ax.set_ylabel("分量 2")
        ax.legend(markerscale=2, fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle("隐空间按数字类别着色 —— 结构越分明,表示越好")
    return _save(fig, out_dir, "latent_scatter.png")


@torch.no_grad()
def plot_latent_grid(model: nn.Module, device: torch.device, out_dir: Path,
                     extent: float = 3.0, side: int = 20) -> Path:
    """[仅 latent_dim=2] 在 z 平面上均匀取网格逐点解码:经典「隐空间地图」。"""
    model.eval()
    zs = torch.stack(
        [
            torch.tensor([x, y], dtype=torch.float32, device=device)
            for y in np.linspace(extent, -extent, side)  # 上到下,与图像行序一致
            for x in np.linspace(-extent, extent, side)
        ]
    )
    imgs = model.decode(zs).cpu()

    fig, ax = plt.subplots(figsize=(7, 7))
    canvas = np.zeros((28 * side, 28 * side), dtype=np.float32)
    for i, img in enumerate(imgs):
        r, c = divmod(i, side)
        canvas[r * 28:(r + 1) * 28, c * 28:(c + 1) * 28] = img[0].numpy()
    ax.imshow(canvas, cmap="gray", extent=[-extent, extent, -extent, extent])
    ax.set_xlabel("z1")
    ax.set_ylabel("z2")
    ax.set_title("隐空间网格解码(每格 = decode(z),z 在 [-3,3]² 均匀采样)")
    return _save(fig, out_dir, "latent_grid.png")


@torch.no_grad()
def plot_interpolation(model: nn.Module, loader: DataLoader, device: torch.device,
                       out_dir: Path, steps: int = 10) -> Path:
    """两幅测试图像的隐变量线性插值:检验隐空间的连续性与平滑过渡。"""
    model.eval()
    xs, ys = next(iter(loader))
    x_a, x_b = xs[:2].to(device)
    label_a, label_b = ys[:2].tolist()

    mu_a, _ = model.encode(x_a.unsqueeze(0))
    mu_b, _ = model.encode(x_b.unsqueeze(0))
    ts = torch.linspace(0, 1, steps, device=device).unsqueeze(1)
    zs = mu_a * (1 - ts) + mu_b * ts
    imgs = model.decode(zs).cpu()

    fig, axes = plt.subplots(1, steps, figsize=(1.3 * steps, 1.8))
    for i, ax in enumerate(axes):
        ax.imshow(imgs[i, 0], cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"t={ts[i, 0].item():.1f}", fontsize=8)
        ax.axis("off")
    fig.suptitle(f"隐变量插值:{label_a} → {label_b}")
    return _save(fig, out_dir, "interpolation.png")


@torch.no_grad()
def plot_kl_per_dim(model: nn.Module, loader: DataLoader, device: torch.device,
                    out_dir: Path, max_samples: int) -> Path:
    """逐隐变量维度的平均 KL:哪些维度在编码信息(活跃),哪些保持接近先验。"""
    model.eval()
    kls, seen = [], 0
    for x, _ in loader:
        mu, logvar = model.encode(x.to(device, non_blocking=True))
        kl_dim = 0.5 * (mu.pow(2) + logvar.exp() - 1 - logvar)  # (B, D)
        kls.append(kl_dim.cpu().numpy())
        seen += mu.size(0)
        if seen >= max_samples:
            break
    mean_kl = np.concatenate(kls)[:max_samples].mean(axis=0)

    fig, ax = plt.subplots(figsize=(max(6.0, 0.35 * len(mean_kl)), 3.5))
    ax.bar(np.arange(len(mean_kl)), mean_kl, color="tab:blue")
    ax.set_xlabel("隐变量维度 j")
    ax.set_ylabel("平均 KL(q(z|x)‖p(z)) 每维")
    ax.set_title("逐维 KL —— 柱高即该维偏离先验的程度(信息含量)")
    ax.grid(alpha=0.3, axis="y")
    return _save(fig, out_dir, "kl_per_dim.png")


# --------------------------------------------------------------------------- #
# 6. 主流程                                                                   #
# --------------------------------------------------------------------------- #


def main() -> None:
    cfg = parse_args()
    set_seed(cfg.seed)
    device = resolve_device(cfg.device)

    out_root = Path(cfg.out_dir)
    fig_dir = out_root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    print(f"配置: {json.dumps(asdict(cfg), ensure_ascii=False, indent=2)}")
    print(f"设备: {device}")

    # 1) 数据
    print("\n[1/5] 准备 MNIST 数据 ...")
    train_loader, test_loader = build_loaders(cfg)
    print(f"  训练集 {len(train_loader.dataset)} 张,测试集 {len(test_loader.dataset)} 张")

    # 2) 模型
    print(f"\n[2/5] 构建 ConvVAE(latent_dim={cfg.latent_dim}) ...")
    model: nn.Module = ConvVAE(latent_dim=cfg.latent_dim).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  参数量: {n_params / 1e3:.1f} K")

    # 3) 训练
    print(f"\n[3/5] 训练 {cfg.epochs} epochs(β={cfg.beta}) ...")
    result = train(cfg, model, train_loader, test_loader, device)

    # 4) 指标落盘
    print("\n[4/5] 保存指标与配置 ...")
    final = evaluate(model, test_loader, device, cfg.beta)
    (out_root / "config.json").write_text(
        json.dumps(asdict(cfg), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_root / "metrics.json").write_text(
        json.dumps({"final_test": final, "best_test_elbo": result["best_test_elbo"],
                    "history": result["history"]}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"  最终测试指标: {final}")

    # 5) 可视化(用 best checkpoint,和指标一致)
    print("\n[5/5] 生成可视化 ...")
    best = torch.load(out_root / "checkpoints" / "best.pt", map_location=device,
                      weights_only=True)
    model.load_state_dict(best["model"])

    plot_loss_curves(result["history"], fig_dir)
    plot_reconstructions(model, test_loader, device, fig_dir)
    plot_prior_samples(model, device, fig_dir)
    plot_latent_scatter(model, test_loader, device, fig_dir,
                        cfg.eval_samples, cfg.latent_dim)
    if cfg.latent_dim == 2:
        plot_latent_grid(model, device, fig_dir)
    plot_interpolation(model, test_loader, device, fig_dir)
    plot_kl_per_dim(model, test_loader, device, fig_dir, cfg.eval_samples)

    print("\n全流程完成。产物目录:")
    print(f"  {out_root}/config.json, metrics.json")
    print(f"  {out_root}/checkpoints/{{last,best}}.pt")
    print(f"  {fig_dir}/*.png")


if __name__ == "__main__":
    main()
