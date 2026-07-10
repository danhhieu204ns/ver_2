from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]


COLORS = {
    "student": ("#2f6fae", "#e8f2fb"),
    "positive": ("#2e7d56", "#e9f6ee"),
    "negative": ("#c46b1a", "#fff0df"),
    "loss": ("#6f4ea8", "#f1ecfb"),
    "neutral": ("#5a5a5a", "#f6f6f6"),
    "objective": ("#a63d40", "#fff0f0"),
}


def add_box(ax, xy, size, text, kind="neutral", fontsize=8.5, lw=1.25):
    edge, face = COLORS[kind]
    x, y = xy
    w, h = size
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.045,rounding_size=0.12",
        linewidth=lw,
        edgecolor=edge,
        facecolor=face,
        mutation_aspect=1,
    )
    ax.add_patch(box)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color="#222222",
        linespacing=1.08,
    )
    return box


def add_lane(ax, y, h, label, color):
    ax.add_patch(
        FancyBboxPatch(
            (0.25, y),
            15.9,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.10",
            linewidth=0,
            facecolor=color,
            alpha=0.16,
        )
    )
    ax.text(
        0.42,
        y + h - 0.24,
        label,
        ha="left",
        va="top",
        fontsize=8.0,
        color="#333333",
        weight="bold",
    )


def center(box, side):
    x, y = box.get_x(), box.get_y()
    w, h = box.get_width(), box.get_height()
    if side == "left":
        return x, y + h / 2
    if side == "right":
        return x + w, y + h / 2
    if side == "top":
        return x + w / 2, y + h
    if side == "bottom":
        return x + w / 2, y
    return x + w / 2, y + h / 2


def arrow(ax, start, end, color="#444444", lw=1.35, rad=0.0, style="-|>"):
    arr = FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
        mutation_scale=10,
        linewidth=lw,
        color=color,
        shrinkA=3,
        shrinkB=4,
        connectionstyle=f"arc3,rad={rad}",
    )
    ax.add_patch(arr)
    return arr


def poly_arrow(ax, points, color="#444444", lw=1.35):
    for start, end in zip(points[:-2], points[1:-1]):
        ax.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            color=color,
            linewidth=lw,
            solid_capstyle="round",
        )
    arrow(ax, points[-2], points[-1], color=color, lw=lw)


def make_figure():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.unicode_minus": False,
        }
    )
    fig, ax = plt.subplots(figsize=(14.0, 6.2))
    ax.set_xlim(0, 16.4)
    ax.set_ylim(0, 8.2)
    ax.axis("off")

    student_edge, student_bg = COLORS["student"]
    pos_edge, pos_bg = COLORS["positive"]
    neg_edge, neg_bg = COLORS["negative"]

    add_lane(ax, 5.82, 1.72, "Student detector branch: full-image Faster R-CNN", student_bg)
    add_lane(ax, 3.08, 1.98, "Positive region distillation: frozen teacher crops", pos_bg)
    add_lane(ax, 0.78, 1.82, "Online hard-negative contrast: foreground-like negatives", neg_bg)

    ax.text(
        8.2,
        7.95,
        "HN-SARD training pipeline",
        ha="center",
        va="top",
        fontsize=15.5,
        weight="bold",
        color="#202020",
    )
    ax.text(
        8.2,
        7.55,
        "student keeps the detector path; teacher and contrastive queue are training-only regularizers",
        ha="center",
        va="top",
        fontsize=8.5,
        color="#555555",
    )

    b_input = add_box(
        ax,
        (0.55, 6.27),
        (1.62, 0.76),
        "Image +\nannotations",
        "neutral",
        fontsize=8.3,
    )
    b_fpn = add_box(
        ax,
        (2.65, 6.23),
        (1.72, 0.84),
        "ResNet-50 FPN\nfeature maps",
        "student",
        fontsize=8.1,
    )
    b_rpn = add_box(
        ax,
        (4.90, 6.23),
        (1.60, 0.84),
        "RPN\nproposals",
        "student",
        fontsize=8.3,
    )
    b_roi = add_box(
        ax,
        (7.05, 6.23),
        (1.72, 0.84),
        "RoI box head\nfeatures",
        "student",
        fontsize=8.0,
    )
    b_heads = add_box(
        ax,
        (9.22, 6.23),
        (1.72, 0.84),
        "Detector heads\nclass + box",
        "student",
        fontsize=8.0,
    )
    b_det = add_box(
        ax,
        (11.60, 6.29),
        (1.14, 0.72),
        r"$L_{\mathrm{det}}$",
        "loss",
        fontsize=11.5,
    )

    for left, right in [(b_input, b_fpn), (b_fpn, b_rpn), (b_rpn, b_roi), (b_roi, b_heads), (b_heads, b_det)]:
        arrow(ax, center(left, "right"), center(right, "left"), student_edge, lw=1.55)

    b_split = add_box(
        ax,
        (4.74, 5.15),
        (1.92, 0.52),
        "proposal split",
        "neutral",
        fontsize=7.7,
        lw=1.0,
    )
    arrow(ax, center(b_rpn, "bottom"), center(b_split, "top"), "#555555", lw=1.25)

    b_pos = add_box(
        ax,
        (4.64, 3.83),
        (2.08, 0.82),
        "matched positive RoIs\nIoU >= 0.5; max 16/img",
        "positive",
        fontsize=7.7,
    )
    b_crop = add_box(
        ax,
        (7.05, 3.83),
        (1.72, 0.82),
        "GT context crop\n1.5x; 224 x 224",
        "positive",
        fontsize=7.7,
    )
    b_teacher = add_box(
        ax,
        (9.22, 3.83),
        (1.72, 0.82),
        "frozen DINOv2-small\nteacher embedding",
        "positive",
        fontsize=7.5,
    )
    b_lpos = add_box(
        ax,
        (11.56, 3.83),
        (1.30, 0.82),
        "scale-aware\ncosine loss\n$L_{\\mathrm{pos}}$",
        "positive",
        fontsize=7.7,
    )
    b_proj = add_box(
        ax,
        (7.16, 5.08),
        (1.50, 0.52),
        "shared projection\nstudent -> teacher dim",
        "loss",
        fontsize=7.3,
        lw=1.0,
    )

    arrow(
        ax,
        (center(b_split, "bottom")[0] - 0.42, center(b_split, "bottom")[1]),
        (center(b_pos, "top")[0] - 0.42, center(b_pos, "top")[1]),
        pos_edge,
        lw=1.45,
        rad=0.02,
    )
    arrow(ax, center(b_pos, "right"), center(b_crop, "left"), pos_edge, lw=1.45)
    arrow(ax, center(b_crop, "right"), center(b_teacher, "left"), pos_edge, lw=1.45)
    arrow(ax, center(b_teacher, "right"), center(b_lpos, "left"), pos_edge, lw=1.45)
    arrow(ax, center(b_roi, "bottom"), center(b_proj, "top"), COLORS["loss"][0], lw=1.25)
    arrow(ax, center(b_proj, "right"), center(b_lpos, "top"), COLORS["loss"][0], lw=1.15, rad=0.11)

    b_queue = add_box(
        ax,
        (10.72, 2.92),
        (1.96, 0.58),
        "positive teacher-anchor queue\nsize 512; detached",
        "positive",
        fontsize=7.1,
        lw=1.0,
    )
    arrow(ax, center(b_teacher, "bottom"), center(b_queue, "top"), pos_edge, lw=1.15)

    b_cand = add_box(
        ax,
        (4.64, 1.40),
        (2.08, 0.78),
        "candidate negatives\ntop 128; IoU < 0.1\nor negative image",
        "negative",
        fontsize=7.3,
    )
    b_hard = add_box(
        ax,
        (7.05, 1.40),
        (1.72, 0.78),
        "hard negatives\ntop 16 by\nforeground score",
        "negative",
        fontsize=7.2,
    )
    b_zneg = add_box(
        ax,
        (9.22, 1.40),
        (1.72, 0.78),
        "projected RoI\nembeddings\n$\\hat{z}^{N}$",
        "negative",
        fontsize=7.4,
    )
    b_lcon = add_box(
        ax,
        (11.56, 1.40),
        (1.30, 0.78),
        "softplus margin\ncontrastive loss\n$L_{\\mathrm{con}}$",
        "negative",
        fontsize=7.2,
    )

    poly_arrow(
        ax,
        [
            center(b_split, "right"),
            (6.98, center(b_split, "right")[1]),
            (6.98, center(b_cand, "top")[1] + 0.18),
            (center(b_cand, "top")[0] + 0.66, center(b_cand, "top")[1]),
        ],
        neg_edge,
        lw=1.45,
    )
    arrow(ax, center(b_cand, "right"), center(b_hard, "left"), neg_edge, lw=1.45)
    arrow(ax, center(b_hard, "right"), center(b_zneg, "left"), neg_edge, lw=1.45)
    arrow(ax, center(b_zneg, "right"), center(b_lcon, "left"), neg_edge, lw=1.45)
    arrow(ax, center(b_queue, "bottom"), center(b_lcon, "top"), pos_edge, lw=1.15, rad=-0.08)

    b_obj = add_box(
        ax,
        (13.62, 3.18),
        (2.30, 1.54),
        "training objective\n"
        r"$L=L_{\mathrm{det}}$" + "\n"
        r"$+\lambda_{\mathrm{pos}}L_{\mathrm{pos}}$" + "\n"
        r"$+\lambda_{\mathrm{con}}(e)L_{\mathrm{con}}$",
        "objective",
        fontsize=8.4,
        lw=1.35,
    )
    arrow(ax, center(b_det, "right"), (13.62, 4.36), COLORS["loss"][0], lw=1.25, rad=-0.22)
    arrow(ax, center(b_lpos, "right"), (13.62, 4.06), pos_edge, lw=1.25)
    arrow(ax, center(b_lcon, "right"), (13.62, 3.52), neg_edge, lw=1.25, rad=0.12)

    ax.text(
        13.70,
        2.74,
        r"selected setting: $\lambda_{\mathrm{pos}}=0.0025$, "
        r"$\lambda_{\mathrm{con}}=0.02$ after 5-epoch warmup",
        ha="left",
        va="center",
        fontsize=7.2,
        color="#555555",
    )
    ax.text(
        13.70,
        2.40,
        "inference uses only the trained detector",
        ha="left",
        va="center",
        fontsize=7.2,
        color="#555555",
    )

    fig.tight_layout(pad=0.25)
    return fig


if __name__ == "__main__":
    figure = make_figure()
    figure.savefig(ROOT / "fig2_hnsard_pipeline.pdf", bbox_inches="tight")
    figure.savefig(ROOT / "fig2_hnsard_pipeline.png", dpi=240, bbox_inches="tight")
    plt.close(figure)
