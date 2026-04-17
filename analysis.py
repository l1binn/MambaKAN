# Fix: Windows OpenMP duplicate library conflict (numpy + PyTorch both ship libiomp5md.dll)
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

"""
Interpretability analysis for the MambaKAN framework (VAE + Mamba + KAN).

Six analysis modules:
  1. Mamba Temporal Importance  — which dFC time windows matter most per class
  2. KAN Activation Curves      — learned spline φ(x) for top-k latent dims
  3. Brain Region Attribution   — gradient-based mapping back to ROI pairs
  4. t-SNE Latent Space         — 2D visualization of Mamba context vectors
  5. ROC Curves                 — one-vs-rest AUC per class
  6. Statistical Significance   — pairwise t-test on brain attribution matrices

Usage:
    python analysis.py --data_path /path/to/data.npz --ckpt_path /path/to/model.pth \
                       --num_classes 4 --top_k 10

Data format expected (provide as .npz or adapt load_data() below):
    test_data:   np.ndarray (N, W, F)   — N subjects, W dFC windows, F features
    test_labels: np.ndarray (N,)        — integer class labels (0-indexed)
    aal_labels:  list[str] length 116   — AAL brain region names (optional)
"""

import argparse
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.manifold import TSNE
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from scipy.stats import ttest_ind

from models.proposed import ProposedModel


# ─────────────────────────────────────────────────────────────────────────────
# Data loading — replace this function with your own data source
# ─────────────────────────────────────────────────────────────────────────────

def load_data(data_path: str):
    """
    Load test data from a .npz file.

    Expected keys:
        'test_data'   : float32 array (N, W, F)
        'test_labels' : int64   array (N,)
        'aal_labels'  : str     array (116,)   [optional]

    Returns:
        test_data:   torch.FloatTensor (N, W, F)
        test_labels: torch.LongTensor  (N,)
        aal_labels:  list[str] or None
    """
    npz = np.load(data_path, allow_pickle=True)
    test_data   = torch.tensor(npz["test_data"],   dtype=torch.float32)
    test_labels = torch.tensor(npz["test_labels"], dtype=torch.long)
    if "aal_labels" in npz:
        raw = npz["aal_labels"]
        aal_labels = [x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in raw]
    else:
        aal_labels = None
    return test_data, test_labels, aal_labels


def load_aal_labels(csv_path: str):
    """Load AAL region names from a two-column CSV (index, name)."""
    labels = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 2:
                try:
                    labels[int(row[0])] = row[1].strip()
                except ValueError:
                    pass
    return labels


# ─────────────────────────────────────────────────────────────────────────────
# 1. Mamba Temporal Importance
# ─────────────────────────────────────────────────────────────────────────────

def analyze_temporal_importance(model, loader, device, out_dir, class_names=None):
    """
    Compute and plot per-timestep Mamba Δ importance, grouped by class.
    Delta values are z-score normalized per-sample so the plot shows
    relative attention across windows, not absolute delta magnitudes.

    Saves:
        mamba_temporal_importance.png
    """
    model.eval()
    all_importance, all_labels = [], []

    with torch.no_grad():
        for data, labels in loader:
            data = data.to(device)
            imp = model.get_temporal_importance(data)   # (b, W)
            # Z-score per subject: removes between-subject Δ magnitude differences
            # so the plot shows relative window selectivity, not absolute scale.
            imp = imp - imp.mean(dim=1, keepdim=True)
            imp = imp / (imp.std(dim=1, keepdim=True) + 1e-8)
            all_importance.append(imp.cpu().numpy())
            all_labels.extend(labels.numpy())

    importance = np.concatenate(all_importance, axis=0)
    labels_arr = np.array(all_labels)
    classes = sorted(set(labels_arr.tolist()))

    if class_names is None:
        class_names = {c: f"Class {c}" for c in classes}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = cm.tab10(np.linspace(0, 0.5, len(classes)))

    ax = axes[0]
    for i, c in enumerate(classes):
        mask = labels_arr == c
        mean_imp = importance[mask].mean(axis=0)
        std_imp  = importance[mask].std(axis=0) / np.sqrt(mask.sum())
        x = np.arange(len(mean_imp))
        ax.plot(x, mean_imp, label=class_names[c], color=colors[i], linewidth=1.8)
        ax.fill_between(x, mean_imp - std_imp, mean_imp + std_imp, alpha=0.20, color=colors[i])
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Time Window Index", fontsize=12)
    ax.set_ylabel("Normalized Selectivity (z-score)", fontsize=12)
    ax.set_title("Mamba Temporal Importance by Class", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    ax = axes[1]
    grand_mean = importance.mean(axis=0)
    heat = np.stack([importance[labels_arr == c].mean(axis=0) - grand_mean for c in classes])
    vmax = np.abs(heat).max()
    im = ax.imshow(heat, aspect="auto", cmap="RdBu_r", interpolation="nearest",
                   vmin=-vmax, vmax=vmax)
    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels([class_names[c] for c in classes], fontsize=10)
    ax.set_xlabel("Time Window Index", fontsize=12)
    ax.set_title("Temporal Importance Heatmap", fontsize=13)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    fpath = os.path.join(out_dir, "mamba_temporal_importance.png")
    plt.savefig(fpath, dpi=600, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fpath}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. KAN Activation Curves
# ─────────────────────────────────────────────────────────────────────────────

def analyze_kan_curves(model, loader, device, out_dir, top_k=10, class_names=None):
    """
    Ceteris-paribus KAN class activation curves for top-k latent dimensions.

    For each important input dimension d:
      - Hold all other dims at their dataset mean
      - Vary dim d over its typical range
      - Record how each class logit changes

    Saves:
        kan_activation_curves.png
    """
    model.eval()
    all_ctx = []
    with torch.no_grad():
        for data, _ in loader:
            data = data.to(device)
            b, w, f = data.shape
            x_flat = data.reshape(b * w, f)
            mu, _, _ = model.vae.encode(x_flat)
            z = mu.reshape(b, w, -1)
            ctx = model.mamba(z)
            all_ctx.append(ctx.cpu())
    x_mean = torch.cat(all_ctx, dim=0).mean(dim=0)

    top_dims = model.kan.get_input_importance().topk(top_k).indices
    num_classes = model.kan.layer2.out_features

    if class_names is None:
        class_names = {i: f"Class {i}" for i in range(num_classes)}
    colors = cm.tab10(np.linspace(0, 0.8, num_classes))

    ncols = 5
    nrows = (top_k + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 2.8))
    axes = axes.flatten()

    for idx, dim in enumerate(top_dims.tolist()):
        ax = axes[idx]
        x_vals, logits = model.kan.get_class_curves(dim, x_mean)
        x_np = x_vals.numpy()
        for c in range(num_classes):
            ax.plot(x_np, logits[c].numpy(),
                    label=class_names.get(c, str(c)), color=colors[c], linewidth=1.6)
        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
        ax.set_title(f"Latent Dim {dim}", fontsize=9)
        ax.set_xlabel("z value", fontsize=8)
        ax.set_ylabel("Δ logit", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.3)
        if idx == 0:
            ax.legend(fontsize=7, loc="upper left")

    for i in range(top_k, len(axes)):
        axes[i].set_visible(False)

    plt.suptitle(f"KAN Class Activation Curves — Top-{top_k} Latent Dims (ceteris paribus)", fontsize=12)
    plt.tight_layout()
    fpath = os.path.join(out_dir, "kan_activation_curves.png")
    plt.savefig(fpath, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fpath}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Brain Region Attribution
# ─────────────────────────────────────────────────────────────────────────────

def analyze_brain_attribution(model, loader, device, out_dir,
                               aal_labels_list=None, n_samples=20, top_k_rois=20):
    """
    Gradient-based attribution: ∂logit_c / ∂x, averaged over samples and windows.
    Maps important features back to 116×116 ROI connectivity matrix.

    Saves:
        brain_attr_heatmap_class<c>.png
        brain_attr_top_rois_class<c>.png
        brain_attr_chord_class<c>.html
    """
    try:
        from pyecharts import options as opts
        from pyecharts.charts import Graph
        _PYECHARTS = True
    except ImportError:
        _PYECHARTS = False
        print("  [INFO] pyecharts not installed — chord diagram will be skipped")

    n_roi = 116
    if aal_labels_list is None:
        aal_labels_list = [f"ROI_{i}" for i in range(n_roi)]

    classes = sorted(set(loader.dataset.tensors[1].tolist()))

    samples_by_class = {c: [] for c in classes}
    for data, labels in loader:
        for i in range(len(labels)):
            c = labels[i].item()
            if len(samples_by_class[c]) < n_samples:
                samples_by_class[c].append(data[i])

    upper_tri_idx = np.triu_indices(n_roi, k=1)

    for cls_idx, c in enumerate(classes):
        if not samples_by_class[c]:
            continue

        grads_list = []
        model.eval()
        for sample in samples_by_class[c]:
            x = sample.unsqueeze(0).to(device)
            grad = model.get_gradient_attribution(x, class_idx=cls_idx)
            grads_list.append(grad.abs().mean(dim=0).cpu().numpy())

        mean_attr = np.mean(grads_list, axis=0)

        mat = np.zeros((n_roi, n_roi))
        mat[upper_tri_idx] = mean_attr
        mat = mat + mat.T

        # Heatmap
        fig, ax = plt.subplots(figsize=(10, 9))
        im = ax.imshow(mat, cmap="hot", aspect="auto")
        tick_step = 10
        tick_positions = list(range(0, n_roi, tick_step))
        ax.set_xticks(tick_positions)
        ax.set_yticks(tick_positions)
        ax.set_xticklabels([aal_labels_list[i] for i in tick_positions], rotation=90, fontsize=6)
        ax.set_yticklabels([aal_labels_list[i] for i in tick_positions], fontsize=6)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(f"Brain Attribution Heatmap — Class {c}", fontsize=12)
        plt.tight_layout()
        hm_path = os.path.join(out_dir, f"brain_attr_heatmap_class{c}.png")
        plt.savefig(hm_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {hm_path}")

        # Top-K ROI bar chart
        roi_importance = mat.sum(axis=1)
        top_idx = np.argsort(roi_importance)[::-1][:top_k_rois]
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.barh([aal_labels_list[i] for i in top_idx][::-1],
                roi_importance[top_idx][::-1], color="steelblue")
        ax.set_xlabel("Attribution Score (sum)", fontsize=11)
        ax.set_title(f"Top-{top_k_rois} Brain Regions — Class {c}", fontsize=12)
        plt.tight_layout()
        bar_path = os.path.join(out_dir, f"brain_attr_top_rois_class{c}.png")
        plt.savefig(bar_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {bar_path}")

        # Chord diagram (requires pyecharts)
        if _PYECHARTS:
            conn_pairs = [((i, j), mat[i, j])
                          for i in range(n_roi) for j in range(i + 1, n_roi)]
            conn_pairs.sort(key=lambda x: abs(x[1]), reverse=True)
            top_100 = conn_pairs[:100]

            connected = set()
            for (i, j), _ in top_100:
                connected.add(i); connected.add(j)

            nodes = [{"name": aal_labels_list[idx], "symbolSize": 10,
                      "itemStyle": {"color": "#ff9999" if idx in connected else "#cccccc"},
                      "label": {"fontSize": 12,
                                "fontWeight": "bold" if idx in connected else "normal"}}
                     for idx in range(n_roi)]

            max_val = max(abs(v) for _, v in top_100)
            min_val = min(abs(v) for _, v in top_100)
            links = []
            for (i, j), val in top_100:
                ratio = ((abs(val) - min_val) / (max_val - min_val)
                         if max_val > min_val else 0.5)
                red, green = 255, int(255 * (1 - ratio ** 0.7))
                links.append({
                    "source": aal_labels_list[i],
                    "target": aal_labels_list[j],
                    "value": abs(val),
                    "lineStyle": {"color": f"rgba({red},{green},0,{ratio*0.8+0.2})",
                                  "width": ratio * 7 + 1, "curveness": 0.3}
                })

            chart = (
                Graph(init_opts=opts.InitOpts(width="2000px", height="1080px"))
                .add("", nodes, links, layout="circular", is_rotate_label=True,
                     linestyle_opts=opts.LineStyleOpts(curve=0.3, opacity=0.8),
                     label_opts=opts.LabelOpts(position="right", font_size=14, formatter="{b}"))
                .set_global_opts(
                    title_opts=opts.TitleOpts(
                        title=f"ROI Connectivity Chord Diagram — Class {c}",
                        subtitle="Top-100 connections by gradient attribution"),
                    tooltip_opts=opts.TooltipOpts(trigger="item", formatter="{b}"))
            )
            chord_path = os.path.join(out_dir, f"brain_attr_chord_class{c}.html")
            chart.render(chord_path)
            print(f"  Saved: {chord_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. t-SNE Latent Space Visualization
# ─────────────────────────────────────────────────────────────────────────────

def analyze_tsne(model, loader, device, out_dir, class_names=None):
    """
    t-SNE visualization of Mamba output latent space.

    Saves:
        tsne_latent_space.png
    """
    model.eval()
    all_latents, all_labels = [], []

    with torch.no_grad():
        for data, labels in loader:
            data = data.to(device)
            b, w, f = data.shape
            x_flat = data.reshape(b * w, f)
            mu, _, _ = model.vae.encode(x_flat)
            z = mu.reshape(b, w, -1)
            ctx = model.mamba(z)
            all_latents.append(ctx.cpu().numpy())
            all_labels.extend(labels.numpy())

    latents = np.concatenate(all_latents, axis=0)
    labels_arr = np.array(all_labels)
    classes = sorted(set(labels_arr.tolist()))

    if class_names is None:
        class_names = {c: f"Class {c}" for c in classes}

    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(latents) // 4))
    embedded = tsne.fit_transform(latents)

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = cm.tab10(np.linspace(0, 0.8, len(classes)))

    for i, c in enumerate(classes):
        mask = labels_arr == c
        ax.scatter(embedded[mask, 0], embedded[mask, 1],
                   c=[colors[i]], label=class_names[c],
                   alpha=0.6, s=50, edgecolors='k', linewidth=0.5)

    ax.set_xlabel("t-SNE Dimension 1", fontsize=12)
    ax.set_ylabel("t-SNE Dimension 2", fontsize=12)
    ax.set_title("t-SNE: Mamba Latent Space", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    fpath = os.path.join(out_dir, "tsne_latent_space.png")
    plt.savefig(fpath, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fpath}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. ROC Curves
# ─────────────────────────────────────────────────────────────────────────────

def analyze_roc(model, loader, device, out_dir, class_names=None):
    """
    ROC curves for each class (one-vs-rest for multi-class).

    Saves:
        roc_curves.png
    """
    model.eval()
    all_probs, all_labels = [], []

    with torch.no_grad():
        for data, labels in loader:
            data = data.to(device)
            logits, _ = model(data, compute_vae_loss=False)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            all_probs.append(probs)
            all_labels.extend(labels.numpy())

    probs_arr = np.concatenate(all_probs, axis=0)
    labels_arr = np.array(all_labels)
    classes = sorted(set(labels_arr.tolist()))
    num_classes = len(classes)

    if class_names is None:
        class_names = {c: f"Class {c}" for c in classes}

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = cm.tab10(np.linspace(0, 0.8, num_classes))

    if num_classes == 2:
        fpr, tpr, _ = roc_curve(labels_arr, probs_arr[:, 1])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=colors[1], lw=2,
                label=f'{class_names[1]} (AUC = {roc_auc:.3f})')
    else:
        labels_bin = label_binarize(labels_arr, classes=classes)
        for i, c in enumerate(classes):
            fpr, tpr, _ = roc_curve(labels_bin[:, i], probs_arr[:, i])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=colors[i], lw=2,
                    label=f'{class_names[c]} (AUC = {roc_auc:.3f})')

    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves', fontsize=13)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    fpath = os.path.join(out_dir, "roc_curves.png")
    plt.savefig(fpath, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fpath}")


# ─────────────────────────────────────────────────────────────────────────────
# 6. Statistical Significance (t-test)
# ─────────────────────────────────────────────────────────────────────────────

def analyze_ttest(model, loader, device, out_dir,
                  aal_labels_list=None, n_samples=20):
    """
    Two-sample t-test between classes for brain connectivity attribution.
    Visualizes -log10(p-value) heatmap; red regions indicate p < 0.05.

    Saves:
        ttest_pvalue_class{c1}_vs_class{c2}.png
    """
    n_roi = 116
    if aal_labels_list is None:
        aal_labels_list = [f"ROI_{i}" for i in range(n_roi)]

    classes = sorted(set(loader.dataset.tensors[1].tolist()))

    samples_by_class = {c: [] for c in classes}
    for data, labels in loader:
        for i in range(len(labels)):
            c = labels[i].item()
            if len(samples_by_class[c]) < n_samples:
                samples_by_class[c].append(data[i])

    upper_tri_idx = np.triu_indices(n_roi, k=1)

    attr_by_class = {}
    for cls_idx, c in enumerate(classes):
        if not samples_by_class[c]:
            continue
        grads_list = []
        model.eval()
        for sample in samples_by_class[c]:
            x = sample.unsqueeze(0).to(device)
            grad = model.get_gradient_attribution(x, class_idx=cls_idx)
            grads_list.append(grad.abs().mean(dim=0).cpu().numpy())
        attr_by_class[c] = np.array(grads_list)

    for i, c1 in enumerate(classes):
        for c2 in classes[i + 1:]:
            if c1 not in attr_by_class or c2 not in attr_by_class:
                continue

            _, p_values = ttest_ind(attr_by_class[c1], attr_by_class[c2], axis=0)

            p_mat = np.ones((n_roi, n_roi))
            p_mat[upper_tri_idx] = p_values
            p_mat = np.minimum(p_mat, p_mat.T)

            log_p = np.clip(-np.log10(p_mat + 1e-10), 0, 5)

            fig, ax = plt.subplots(figsize=(10, 9))
            im = ax.imshow(log_p, cmap="YlOrRd", aspect="auto", vmin=0, vmax=5)
            tick_step = 10
            tick_positions = list(range(0, n_roi, tick_step))
            ax.set_xticks(tick_positions)
            ax.set_yticks(tick_positions)
            ax.set_xticklabels([aal_labels_list[j] for j in tick_positions], rotation=90, fontsize=6)
            ax.set_yticklabels([aal_labels_list[j] for j in tick_positions], fontsize=6)
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("-log10(p-value)", fontsize=10)
            ax.set_title(f"t-test: Class {c1} vs {c2}\nRed = significant (p<0.05)", fontsize=11)
            plt.tight_layout()

            fpath = os.path.join(out_dir, f"ttest_pvalue_class{c1}_vs_class{c2}.png")
            plt.savefig(fpath, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"  Saved: {fpath}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="MambaKAN Interpretability Analysis")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to .npz file with test_data, test_labels, aal_labels")
    parser.add_argument("--ckpt_path", type=str, required=True,
                        help="Path to trained model checkpoint (.pth)")
    parser.add_argument("--aal_csv", type=str, default=None,
                        help="Optional: path to AAL labels CSV (overrides npz aal_labels)")
    parser.add_argument("--num_classes", type=int, default=4)
    parser.add_argument("--input_dim", type=int, default=6670,
                        help="Feature dimension of each dFC window")
    parser.add_argument("--latent_dim", type=int, default=128)
    parser.add_argument("--hidden_dims", nargs="+", type=int, default=[2048, 1024, 512, 256])
    parser.add_argument("--mamba_n_layers", type=int, default=2)
    parser.add_argument("--mamba_d_state", type=int, default=16)
    parser.add_argument("--mamba_d_conv", type=int, default=4)
    parser.add_argument("--mamba_expand", type=int, default=2)
    parser.add_argument("--kan_hidden_dim", type=int, default=64)
    parser.add_argument("--kan_grid_size", type=int, default=5)
    parser.add_argument("--kan_spline_order", type=int, default=3)
    parser.add_argument("--top_k", type=int, default=10,
                        help="Top-k KAN latent dims to visualize")
    parser.add_argument("--n_grad_samples", type=int, default=20,
                        help="Samples per class for gradient attribution")
    parser.add_argument("--out_dir", type=str, default="output/analysis")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--analyses", nargs="+", type=int, default=None,
                        help="Which analyses to run (1=mamba 2=kan 3=brain 4=tsne 5=roc 6=ttest). Default: all")
    parser.add_argument("--class_names", nargs="+", type=str, default=None,
                        help="Class names in label order, e.g. CN EMCI LMCI AD")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)

    # ── Load data ──────────────────────────────────────────────────────────
    test_data, test_labels, aal_labels = load_data(args.data_path)
    test_loader = DataLoader(TensorDataset(test_data, test_labels),
                             batch_size=args.batch_size, shuffle=False)

    if args.aal_csv is not None:
        aal_map = load_aal_labels(args.aal_csv)
        aal_labels = list(aal_map.values())

    # ── Class names ────────────────────────────────────────────────────────
    classes = sorted(set(test_labels.tolist()))
    if args.class_names is not None:
        class_names = {c: args.class_names[c] for c in classes if c < len(args.class_names)}
    else:
        class_names = {c: f"Class {c}" for c in classes}

    # ── Build & load model ─────────────────────────────────────────────────
    model = ProposedModel(
        input_dim=args.input_dim,
        vae_hidden_dims=args.hidden_dims,
        latent_dim=args.latent_dim,
        mamba_n_layers=args.mamba_n_layers,
        mamba_d_state=args.mamba_d_state,
        mamba_d_conv=args.mamba_d_conv,
        mamba_expand=args.mamba_expand,
        kan_hidden_dim=args.kan_hidden_dim,
        kan_grid_size=args.kan_grid_size,
        kan_spline_order=args.kan_spline_order,
        num_classes=args.num_classes,
    ).to(device)

    model.load_state_dict(torch.load(args.ckpt_path, map_location=device))
    model.eval()
    print(f"Loaded checkpoint: {args.ckpt_path}")

    # ── Run analyses ───────────────────────────────────────────────────────
    run = args.analyses

    if run is None or 1 in run:
        print("[1] Mamba temporal importance...")
        analyze_temporal_importance(model, test_loader, device, args.out_dir, class_names)

    if run is None or 2 in run:
        print("[2] KAN activation curves...")
        analyze_kan_curves(model, test_loader, device, args.out_dir,
                           top_k=args.top_k, class_names=class_names)

    if run is None or 3 in run:
        print("[3] Brain region gradient attribution...")
        analyze_brain_attribution(model, test_loader, device, args.out_dir,
                                  aal_labels_list=aal_labels,
                                  n_samples=args.n_grad_samples, top_k_rois=20)

    if run is None or 4 in run:
        print("[4] t-SNE latent space...")
        analyze_tsne(model, test_loader, device, args.out_dir, class_names)

    if run is None or 5 in run:
        print("[5] ROC curves...")
        analyze_roc(model, test_loader, device, args.out_dir, class_names)

    if run is None or 6 in run:
        print("[6] t-test statistical significance...")
        analyze_ttest(model, test_loader, device, args.out_dir,
                      aal_labels_list=aal_labels, n_samples=args.n_grad_samples)

    print(f"\nAll outputs saved to: {args.out_dir}")


if __name__ == "__main__":
    main()
