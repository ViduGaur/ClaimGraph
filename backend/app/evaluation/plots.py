"""Figures for the results: the precision/recall trade-off of asserted contradictions, and a confusion matrix."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from app.evaluation.experiment import SPEC_BY_NAME  # noqa: E402
from app.evaluation.metrics import PRED_LABELS, RELATIONS  # noqa: E402

INK, MUTED, GRID = "#1b1f24", "#5b6570", "#d7dce2"
# one hue per system family, from a colour-blind-safe set; the bar label carries the identity
GROUP_COLOR = {"baseline": "#8a8f98", "llm": "#b45309", "rules": "#4c6fb0", "hybrid": "#0f766e"}
BAR_SYSTEMS = ["nli", "llm_direct_claims", "llm_direct_context", "rules_full", "hybrid_no_gate", "hybrid_no_stance", "hybrid_full"]


def _style(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def bar_panel(ax, results: dict, metric: str, title: str, xmax: float = 100) -> None:
    names = [n for n in BAR_SYSTEMS if n in results]
    for i, name in enumerate(reversed(names)):
        m = results[name]["metrics"][metric]
        color = GROUP_COLOR[SPEC_BY_NAME[name].group]
        if m["value"] is None:
            ax.text(1, i, "n/a (none asserted)", va="center", fontsize=8, color=MUTED)
            continue
        v = 100 * m["value"]
        ax.barh(i, v, color=color, height=0.62, alpha=1.0 if name == "hybrid_full" else 0.55)
        end = v
        if m["lo"] is not None:
            ax.plot([100 * m["lo"], 100 * m["hi"]], [i, i], color=INK, linewidth=1)
            end = max(v, 100 * m["hi"])
        ax.text(min(end + 2, 0.93 * xmax), i + 0.02, f"{v:.0f}", va="center", fontsize=8, color=INK)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([SPEC_BY_NAME[n].title.strip() for n in reversed(names)], fontsize=9, color=INK)
    ax.set_xlim(0, xmax)
    ax.set_title(title, fontsize=9, color=INK, loc="left")
    _style(ax)


def make_plots(summary: dict, out_dir: Path) -> None:
    results = summary["results"]
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8), sharey=True)
    worst = max((r["metrics"]["false_con_per_100"]["hi"] or r["metrics"]["false_con_per_100"]["value"] or 0) for r in results.values())
    bar_panel(axes[0], results, "false_con_per_100", "False contradictions per 100 pairs\n(lower is better)", xmax=max(20, 10 * (int(100 * worst / 10) + 2)))
    bar_panel(axes[1], results, "verdict_acc", "Verdict accuracy on\nface-value conflicts (%)")
    bar_panel(axes[2], results, "macro_f1", "Relation macro-F1 (%)\n")
    fig.tight_layout()
    fig.savefig(out_dir / "main_results.png", dpi=200)
    fig.savefig(out_dir / "main_results.pdf")
    plt.close(fig)

    table = summary["confusion"].get("hybrid_full")
    if table:
        fig, ax = plt.subplots(figsize=(5.2, 4.4))
        grid = [[table[g][p] for p in PRED_LABELS] for g in RELATIONS]
        top = max(max(row) for row in grid) or 1
        ax.imshow(grid, cmap="Greens", vmin=0, vmax=top)
        for i, row in enumerate(grid):
            for j, v in enumerate(row):
                if v:
                    ax.text(j, i, str(v), ha="center", va="center", fontsize=9, color="white" if v > top * 0.55 else INK)
        ax.set_xticks(range(len(PRED_LABELS)), [x[:5].title() for x in PRED_LABELS], fontsize=8, color=INK)
        ax.set_yticks(range(len(RELATIONS)), [x[:5].title() for x in RELATIONS], fontsize=8, color=INK)
        ax.set_xlabel("predicted", fontsize=9, color=MUTED)
        ax.set_ylabel("gold", fontsize=9, color=MUTED)
        ax.set_title("ClaimGraph (full): confusion matrix", fontsize=10, color=INK, loc="left")
        for side in ax.spines.values():
            side.set_visible(False)
        fig.tight_layout()
        fig.savefig(out_dir / "confusion_full.png", dpi=200)
        fig.savefig(out_dir / "confusion_full.pdf")
        plt.close(fig)
