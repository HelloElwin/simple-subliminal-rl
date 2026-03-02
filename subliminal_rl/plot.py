"""Plotting, CSV output, and results display."""

import csv
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np


def make_output_dir(args) -> str:
    """Create output directory: ./exp/<args>_<datetime>/"""
    parts = [
        args.backbone,
        f"t{args.teacher_steps // 1000}k",
        f"s{args.student_steps // 1000}k",
        f"n{args.num_seeds}",
    ]
    all_controls = {"c1", "c3", "c4", "c5"}
    controls = set() if args.controls.lower() == "none" else set(args.controls.lower().split(","))
    if not controls:
        parts.append("noctrl")
    elif controls != all_controls:
        parts.append("ctrl_" + "_".join(sorted(controls)))
    if args.step_level_reward:
        parts.append("steplvl")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dirname = "_".join(parts) + f"_{timestamp}"
    path = os.path.join("exp", dirname)
    os.makedirs(path, exist_ok=True)
    return path


def save_csv(output_dir: str, all_results: list[dict], all_curves: list[dict], seeds: list[int]):
    """Save final_results.csv and training_curves.csv."""
    conditions = list(all_results[0].keys())

    with open(os.path.join(output_dir, "final_results.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["seed", "condition", "RED", "BLUE", "GREEN", "NONE"])
        for seed, results in zip(seeds, all_results):
            for cond in conditions:
                r = results[cond]
                writer.writerow([seed, cond, r["RED"], r["BLUE"], r["GREEN"], r.get("NONE", 0)])

    with open(os.path.join(output_dir, "training_curves.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["seed", "condition", "step", "mean_reward", "pg_loss", "v_loss", "entropy", "RED", "BLUE", "GREEN", "NONE"]
        )
        for seed, curves in zip(seeds, all_curves):
            for cond, log in curves.items():
                for row in log:
                    writer.writerow(
                        [
                            seed,
                            cond,
                            row["step"],
                            row["mean_reward"],
                            row["pg_loss"],
                            row["v_loss"],
                            row["entropy"],
                            row.get("RED", ""),
                            row.get("BLUE", ""),
                            row.get("GREEN", ""),
                            row.get("NONE", ""),
                        ]
                    )


def print_results_table(all_results: list[dict]):
    """Print aggregated results across seeds."""
    conditions = list(all_results[0].keys())

    print(f"\n{'='*72}")
    print("SUBLIMINAL RL EXPERIMENT RESULTS")
    print(f"{'='*72}")
    print(f"{'Setting':<30} {'RED%':>8} {'BLUE%':>8} {'GREEN%':>8} {'NONE%':>8}")
    print(f"{'-'*72}")

    for cond in conditions:
        reds = [r[cond]["RED"] for r in all_results]
        blues = [r[cond]["BLUE"] for r in all_results]
        greens = [r[cond]["GREEN"] for r in all_results]
        nones = [r[cond].get("NONE", 0) for r in all_results]

        if len(all_results) > 1:
            print(
                f"{cond:<30} "
                f"{np.mean(reds):>6.1%}\u00b1{np.std(reds):>4.1%} "
                f"{np.mean(blues):>6.1%}\u00b1{np.std(blues):>4.1%} "
                f"{np.mean(greens):>6.1%}\u00b1{np.std(greens):>4.1%} "
                f"{np.mean(nones):>6.1%}\u00b1{np.std(nones):>4.1%}"
            )
        else:
            print(
                f"{cond:<30} "
                f"{reds[0]:>7.1%} "
                f"{blues[0]:>8.1%} "
                f"{greens[0]:>8.1%} "
                f"{nones[0]:>8.1%}"
            )

    print(f"{'='*72}")
    if len(all_results) > 1:
        print(f"(mean \u00b1 std across {len(all_results)} seeds)")


def _curve_stats(all_curves, condition, metric):
    """Extract (steps, mean, ci_lo, ci_hi) for a metric across seeds."""
    seed_data = [curves[condition] for curves in all_curves if condition in curves]
    if not seed_data:
        return None
    steps = [row["step"] for row in seed_data[0]]
    values_per_step = []
    for i in range(len(steps)):
        vals = [sd[i][metric] for sd in seed_data if i < len(sd)]
        values_per_step.append(vals)
    means = [np.mean(v) for v in values_per_step]
    if len(seed_data) > 1:
        sems = [
            1.96 * np.std(v, ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0
            for v in values_per_step
        ]
    else:
        sems = [0] * len(steps)
    ci_lo = [m - s for m, s in zip(means, sems)]
    ci_hi = [m + s for m, s in zip(means, sems)]
    return steps, means, ci_lo, ci_hi


def save_plot(output_dir: str, all_results: list[dict], all_curves: list[dict]):
    """Generate and save experiment plot.

    Layout: 2x2
      (0,0) Teacher & student learning curves -- RED% over training
      (0,1) Trait specificity -- RED teacher->RED%, BLUE teacher->BLUE%
      (1,0) Training reward curves
      (1,1) Final results bar chart
    """
    has_controls = "C1: Diff init" in all_results[0]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # --- (0,0) Main learning curves: RED% ---
    ax = axes[0, 0]
    for cond, color, ls in [
        ("Teacher (RED)", "#d62728", "-"),
        ("Student (same init)", "#1f77b4", "-"),
        ("C1: Diff init", "#ff7f0e", "--"),
    ]:
        stats = _curve_stats(all_curves, cond, "RED")
        if stats:
            steps, means, ci_lo, ci_hi = stats
            ax.plot(steps, means, color=color, ls=ls, label=cond)
            ax.fill_between(steps, ci_lo, ci_hi, color=color, alpha=0.15)
    ax.set_xlabel("Training Steps")
    ax.set_ylabel("RED %")
    ax.set_title("RED Goal Preference During Training")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8)

    # --- (0,1) Trait specificity ---
    ax = axes[0, 1]
    stats = _curve_stats(all_curves, "Student (same init)", "RED")
    if stats:
        steps, means, ci_lo, ci_hi = stats
        ax.plot(steps, means, color="#d62728", ls="-", label="RED teacher \u2192 RED%")
        ax.fill_between(steps, ci_lo, ci_hi, color="#d62728", alpha=0.15)
    if has_controls:
        stats = _curve_stats(all_curves, "C4: BLUE teacher", "BLUE")
        if stats:
            steps, means, ci_lo, ci_hi = stats
            ax.plot(steps, means, color="#1f77b4", ls="-", label="BLUE teacher \u2192 BLUE%")
            ax.fill_between(steps, ci_lo, ci_hi, color="#1f77b4", alpha=0.15)
    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Goal %")
    ax.set_title("Trait Specificity")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8)

    # --- (1,0) Mean reward curves ---
    ax = axes[1, 0]
    for cond, color, ls in [
        ("Teacher (RED)", "#d62728", "-"),
        ("Student (same init)", "#1f77b4", "-"),
        ("C1: Diff init", "#ff7f0e", "--"),
    ]:
        stats = _curve_stats(all_curves, cond, "mean_reward")
        if stats:
            steps, means, ci_lo, ci_hi = stats
            ax.plot(steps, means, color=color, ls=ls, label=cond)
            ax.fill_between(steps, ci_lo, ci_hi, color=color, alpha=0.15)
    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Mean Reward")
    ax.set_title("Training Reward")
    ax.legend(fontsize=8)

    # --- (1,1) Final results bar chart ---
    ax = axes[1, 1]
    conditions = list(all_results[0].keys())
    x = np.arange(len(conditions))
    width = 0.2
    goal_colors = {"RED": "#d62728", "BLUE": "#1f77b4", "GREEN": "#2ca02c", "NONE": "#7f7f7f"}

    for i, (goal, color) in enumerate(goal_colors.items()):
        vals = [np.mean([r[c].get(goal, 0) for r in all_results]) for c in conditions]
        if len(all_results) > 1:
            errs = [
                1.96 * np.std([r[c].get(goal, 0) for r in all_results], ddof=1) / np.sqrt(len(all_results))
                for c in conditions
            ]
        else:
            errs = None
        ax.bar(x + i * width, vals, width, label=goal, color=color, yerr=errs, capsize=2, error_kw={"lw": 0.8})

    ax.set_xticks(x + 1.5 * width)
    ax.set_xticklabels(conditions, rotation=35, ha="right", fontsize=7)
    ax.set_ylabel("Fraction")
    ax.set_title("Final Evaluation (Env A)")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, ncol=2)

    for ax in axes.flat:
        ax.grid(True, alpha=0.3)

    fig.suptitle("Subliminal RL Experiment", fontsize=14, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(output_dir, "results.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"\nPlot saved to {path}")
