#!/usr/bin/env python3
"""
Visualize temporal edge quads from GPU or CPU output text files.

GPU format (gpu_temporal_quad_matches_kfX_cfY.txt):
  kf_mate_idx cf_mate_idx + KF/CF left/right x, y, orientation (rad)

CPU format (quads_kfX_cfY.txt):
  kf_idx quad_type left_x/y/orient right_x/y/orient is_TP
  quad_type: -1 KF stereo, -2 GT CF stereo, >=0 candidate

Four panels: KF left, KF right, CF left, CF right.
Cross-panel links use matplotlib ConnectionPatch (correct across image sizes).

Examples:
  python visualize_quads.py
  python visualize_quads.py outputs/gpu_temporal_quad_matches_kf0_cf1.txt \\
      --kf-left path/to/kf_im0.png --kf-right path/to/kf_im1.png \\
      --cf-left path/to/cf_im0.png --cf-right path/to/cf_im1.png
  python visualize_quads.py outputs/gpu_temporal_quad_matches_kf0_cf1.txt --max-quads 100
  python visualize_quads.py outputs/gpu_temporal_quad_matches_kf0_cf1.txt \\
      --max-quads 50 --top-quads 500

Example usage when running on Brown CCV Oscars:
    export seq=/gpfs/data/bkimia/Datasets/ETH3D/slam/cables_2/
    python3 visualize_quads.py --kf-left ${seq}/rgb2/11873.436536.png --cf-left ${seq}/rgb2/11873.473400.png --kf-right ${seq}/rgb/11873.436536.png --cf-right ${seq}/rgb/11873.473400.png
    python3 visualize_quads.py --kf-left ${seq}/rgb2/11873.252219.png --cf-left ${seq}/rgb2/11873.436536.png --kf-right ${seq}/rgb/11873.252219.png --cf-right ${seq}/rgb/11873.436536.png

GPU output rows are sorted by ascending ambiguity (stable): earlier rows are
higher-rank / more stable. Random subsampling uses only the first --top-quads rows.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import ConnectionPatch
from matplotlib.pyplot import Axes, Figure


@dataclass
class Quad:
    kf_lx: float
    kf_ly: float
    kf_lo: float
    kf_rx: float
    kf_ry: float
    kf_ro: float
    cf_lx: float
    cf_ly: float
    cf_lo: float
    cf_rx: float
    cf_ry: float
    cf_ro: float
    style: str = "final"  # final | gt | tp | fp


def detect_format(quads_path: Path) -> str:
    with quads_path.open() as f:
        hdr = f.readline().lower()
    if "kf_mate_idx" in hdr:
        return "gpu"
    if "quad_type" in hdr:
        return "cpu"
    data = np.loadtxt(quads_path, comments="#")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] >= 14:
        return "gpu"
    return "cpu"


def read_gpu_quads(quads_path: Path) -> List[Quad]:
    with quads_path.open() as f:
        hdr = f.readline().lower().split()
    geo_offset = 3 if "photometric_rms" in hdr else 2
    data = np.loadtxt(quads_path, skiprows=1)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    quads: List[Quad] = []
    for row in data:
        quads.append(
            Quad(
                kf_lx=row[geo_offset],
                kf_ly=row[geo_offset + 1],
                kf_lo=row[geo_offset + 2],
                kf_rx=row[geo_offset + 3],
                kf_ry=row[geo_offset + 4],
                kf_ro=row[geo_offset + 5],
                cf_lx=row[geo_offset + 6],
                cf_ly=row[geo_offset + 7],
                cf_lo=row[geo_offset + 8],
                cf_rx=row[geo_offset + 9],
                cf_ry=row[geo_offset + 10],
                cf_ro=row[geo_offset + 11],
                style="final",
            )
        )
    return quads


def read_cpu_quads(quads_path: Path) -> List[Quad]:
    data = np.loadtxt(quads_path)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    kf_idx = data[:, 0].astype(int)
    quad_type = data[:, 1].astype(int)
    left_x, left_y, left_o = data[:, 2], data[:, 3], data[:, 4]
    right_x, right_y, right_o = data[:, 5], data[:, 6], data[:, 7]
    is_tp = data[:, 8].astype(int)

    is_kf = quad_type == -1
    is_gt = quad_type == -2
    is_cand = quad_type >= 0

    quads: List[Quad] = []
    for kf in np.unique(kf_idx[is_kf]):
        i_kf = np.where((kf_idx == kf) & is_kf)[0]
        if len(i_kf) == 0:
            continue
        i_kf = int(i_kf[0])
        targets = []
        gt_idx = np.where((kf_idx == kf) & is_gt)[0]
        if len(gt_idx):
            targets.append(int(gt_idx[0]))
        targets.extend(int(i) for i in np.where((kf_idx == kf) & is_cand)[0])
        for it in targets:
            if is_gt[it]:
                style = "gt"
            elif is_tp[it] == 1:
                style = "tp"
            else:
                style = "fp"
            quads.append(
                Quad(
                    kf_lx=left_x[i_kf],
                    kf_ly=left_y[i_kf],
                    kf_lo=left_o[i_kf],
                    kf_rx=right_x[i_kf],
                    kf_ry=right_y[i_kf],
                    kf_ro=right_o[i_kf],
                    cf_lx=left_x[it],
                    cf_ly=left_y[it],
                    cf_lo=left_o[it],
                    cf_rx=right_x[it],
                    cf_ry=right_y[it],
                    cf_ro=right_o[it],
                    style=style,
                )
            )
    return quads


def subsample_quads(
    quads: List[Quad],
    max_quads: int,
    top_quads: int,
    rng: np.random.Generator,
) -> List[Quad]:
    """Subsample for display. When top_quads > 0, only the first top_quads rows are used
    (GPU files: highest rank / lowest ambiguity first)."""
    pool = quads[:top_quads] if top_quads > 0 else quads
    n_total = len(quads)
    n_pool = len(pool)

    if max_quads <= 0:
        if top_quads > 0 and n_pool < n_total:
            print(f"Plotting all {n_pool} quads from top {top_quads} ranked rows ({n_total} total)")
        return pool

    if n_pool <= max_quads:
        if top_quads > 0 and n_pool < n_total:
            print(f"Plotting all {n_pool} quads from top {top_quads} ranked rows ({n_total} total)")
        return pool

    idx = rng.choice(n_pool, size=max_quads, replace=False)
    if top_quads > 0:
        print(
            f"Randomly plotting {max_quads} / {n_pool} quads "
            f"(top {top_quads} ranked of {n_total} total)"
        )
    else:
        print(f"Randomly plotting {max_quads} / {n_pool} quad rows")
    return [pool[i] for i in sorted(idx)]


def resolve_image_paths(
    quads_path: Path,
    kf_left: Optional[Path],
    kf_right: Optional[Path],
    cf_left: Optional[Path],
    cf_right: Optional[Path],
) -> Tuple[Optional[Path], Optional[Path], Optional[Path], Optional[Path]]:
    paths = [kf_left, kf_right, cf_left, cf_right]
    if all(p is not None and p.is_file() for p in paths):
        return tuple(paths)  # type: ignore

    m = re.search(r"kf(\d+)_cf(\d+)", quads_path.name)
    if not m:
        return tuple(paths)  # type: ignore

    kf_i, cf_i = int(m.group(1)), int(m.group(2))
    dataset = Path("/gpfs/data/bkimia/Datasets/ETH3D/slam/cables_2")
    if not dataset.is_dir():
        return tuple(paths)  # type: ignore

    folders = sorted([p for p in dataset.iterdir() if p.is_dir()])
    if len(folders) <= max(kf_i, cf_i):
        return tuple(paths)  # type: ignore

    # Match MATLAB visualize_quads: skip first 4 sorted folders
    folders = folders[4:]
    if len(folders) <= max(kf_i, cf_i):
        return tuple(paths)  # type: ignore

    def pair(frame_idx: int) -> Tuple[Path, Path]:
        d = folders[frame_idx]
        return d / "im0.png", d / "im1.png"

    kf_l, kf_r = pair(kf_i)
    cf_l, cf_r = pair(cf_i)
    out = [
        kf_left if kf_left and kf_left.is_file() else kf_l,
        kf_right if kf_right and kf_right.is_file() else kf_r,
        cf_left if cf_left and cf_left.is_file() else cf_l,
        cf_right if cf_right and cf_right.is_file() else cf_r,
    ]
    print(f"Auto-resolved images for kf{kf_i} cf{cf_i}")
    return tuple(out)  # type: ignore


def valid_xy(x: float, y: float) -> bool:
    return not (np.isnan(x) or np.isnan(y) or (x <= 0 and y <= 0))


def plot_edge_segment(ax: Axes, x: float, y: float, orient_rad: float, color: str, seg_len: float = 25.0) -> None:
    if not valid_xy(x, y):
        return
    o = 0.0 if np.isnan(orient_rad) else orient_rad
    dx = (seg_len / 2.0) * np.cos(o)
    dy = (seg_len / 2.0) * np.sin(o)
    ax.plot([x - dx, x + dx], [y - dy, y + dy], color=color, linewidth=2, solid_capstyle="round")


def cf_color_style(style: str) -> Tuple[str, str]:
    if style == "gt":
        return "green", "o"
    if style == "fp":
        return "red", "o"
    return "blue", "o"


def add_link(
    fig: Figure,
    ax_a: Axes,
    xy_a: Tuple[float, float],
    ax_b: Axes,
    xy_b: Tuple[float, float],
    color: str,
    linestyle: str,
    linewidth: float,
) -> None:
    if not (valid_xy(*xy_a) and valid_xy(*xy_b)):
        return
    con = ConnectionPatch(
        xyA=xy_a,
        xyB=xy_b,
        coordsA="data",
        coordsB="data",
        axesA=ax_a,
        axesB=ax_b,
        color=color,
        linestyle=linestyle,
        linewidth=linewidth,
        alpha=0.85,
    )
    con.set_in_layout(False)
    fig.add_artist(con)


def setup_axes_image(ax: Axes, img_path: Optional[Path]) -> bool:
    if img_path is None or not img_path.is_file():
        ax.set_axis_off()
        return False
    img = plt.imread(img_path)
    if img.ndim == 3:
        img = img.mean(axis=2)
    h, w = img.shape[:2]
    # extent matches C++/OpenCV pixel coordinates (0 = left/top, y down)
    ax.imshow(img, cmap="gray", origin="upper", extent=(0, w, h, 0))
    ax.set_aspect("equal")
    return True


def plot_quads_2x2(
    img_paths: Tuple[Optional[Path], Optional[Path], Optional[Path], Optional[Path]],
    quads: Sequence[Quad],
    out_path: Path,
    show: bool,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    ax_kf_l, ax_kf_r = axes[0, 0], axes[0, 1]
    ax_cf_l, ax_cf_r = axes[1, 0], axes[1, 1]
    titles = ["KF Left", "KF Right", "CF Left", "CF Right"]
    ax_list = [ax_kf_l, ax_kf_r, ax_cf_l, ax_cf_r]

    has_any = False
    for ax, p, t in zip(ax_list, img_paths, titles):
        if setup_axes_image(ax, p):
            has_any = True
        ax.set_title(t, fontsize=11, fontweight="bold")

    if not has_any:
        print("Warning: no image files found. Pass --kf-left, --kf-right, --cf-left, --cf-right.")

    # KF edges (magenta)
    for q in quads:
        plot_edge_segment(ax_kf_l, q.kf_lx, q.kf_ly, q.kf_lo, "magenta")
        ax_kf_l.plot(q.kf_lx, q.kf_ly, "o", color="magenta", markersize=4)
        plot_edge_segment(ax_kf_r, q.kf_rx, q.kf_ry, q.kf_ro, "magenta")
        ax_kf_r.plot(q.kf_rx, q.kf_ry, "o", color="magenta", markersize=4)

    # CF edges (per style)
    for q in quads:
        color, marker = cf_color_style(q.style)
        plot_edge_segment(ax_cf_l, q.cf_lx, q.cf_ly, q.cf_lo, color)
        ax_cf_l.plot(q.cf_lx, q.cf_ly, marker, color=color, markersize=4, linestyle="None")
        plot_edge_segment(ax_cf_r, q.cf_rx, q.cf_ry, q.cf_ro, color)
        ax_cf_r.plot(q.cf_rx, q.cf_ry, marker, color=color, markersize=4, linestyle="None")

    lw = 0.6 if len(quads) > 50 else 1.0
    for q in quads:
        cf_col = {"gt": "green", "fp": "red"}.get(q.style, "blue")
        add_link(fig, ax_kf_l, (q.kf_lx, q.kf_ly), ax_kf_r, (q.kf_rx, q.kf_ry), "magenta", "-", lw)
        add_link(fig, ax_cf_l, (q.cf_lx, q.cf_ly), ax_cf_r, (q.cf_rx, q.cf_ry), cf_col, "-", lw)
        add_link(fig, ax_kf_l, (q.kf_lx, q.kf_ly), ax_cf_l, (q.cf_lx, q.cf_ly), "green", "--", lw)
        add_link(fig, ax_kf_r, (q.kf_rx, q.kf_ry), ax_cf_r, (q.cf_rx, q.cf_ry), "cyan", "--", lw)

    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    proj_root = script_dir.parent
    default_quads = proj_root / "outputs" / "gpu_temporal_quad_matches_kf0_cf5.txt"
    if not default_quads.is_file():
        default_quads = proj_root / "outputs" / "quads_kf5_cf6.txt"

    parser = argparse.ArgumentParser(description="Visualize temporal edge quads (2x2 layout).")
    parser.add_argument(
        "quads_file",
        nargs="?",
        type=Path,
        default=default_quads,
        help="Path to gpu_temporal_quad_matches_*.txt or quads_*.txt",
    )
    parser.add_argument("--kf-left", type=Path, default=None)
    parser.add_argument("--kf-right", type=Path, default=None)
    parser.add_argument("--cf-left", type=Path, default=None)
    parser.add_argument("--cf-right", type=Path, default=None)
    parser.add_argument(
        "--max-quads",
        type=int,
        default=100,
        help="Random sample size from the top-ranked pool (0 = plot entire pool)",
    )
    parser.add_argument(
        "--top-quads",
        type=int,
        default=500,
        metavar="N",
        help="Only the first N file rows are eligible (highest rank). 0 = no limit",
    )
    parser.add_argument("--ambiguity", type=int, default=None, help="GPU: keep KF mates with N rows")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--show", action="store_true", help="Open interactive window")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Output PNG path")
    args = parser.parse_args()

    quads_path = args.quads_file.resolve()
    if not quads_path.is_file():
        raise SystemExit(f"Quads file not found: {quads_path}")

    fmt = detect_format(quads_path)
    print(f"Reading {fmt} quads: {quads_path}")

    if fmt == "gpu":
        raw = np.loadtxt(quads_path, skiprows=1)
        if raw.ndim == 1:
            raw = raw.reshape(1, -1)
        kf_mate_idx = raw[:, 0].astype(int)
        quads = read_gpu_quads(quads_path)
        if args.ambiguity is not None:
            counts: dict = {}
            for idx in kf_mate_idx:
                counts[idx] = counts.get(idx, 0) + 1
            allowed = {idx for idx, c in counts.items() if c == args.ambiguity}
            quads = [q for q, idx in zip(quads, kf_mate_idx) if idx in allowed]
    else:
        quads = read_cpu_quads(quads_path)
        if args.ambiguity is not None:
            print("Note: --ambiguity is mainly for GPU files (per kf_mate_idx row count).")

    n_before = len(quads)
    rng = np.random.default_rng(args.seed)
    quads = subsample_quads(quads, args.max_quads, args.top_quads, rng)

    img_paths = resolve_image_paths( quads_path, args.kf_left, args.kf_right, args.cf_left, args.cf_right )

    stem = quads_path.stem
    out_path = args.output or quads_path.with_name(quads_path.stem + "_viz.png")
    plot_quads_2x2(img_paths, quads, out_path, show=args.show)


if __name__ == "__main__":
    main()
