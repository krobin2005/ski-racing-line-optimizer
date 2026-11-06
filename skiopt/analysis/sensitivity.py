"""One-parameter sensitivity study: re-optimize at several values of one config entry and report the
best time at each.

    python -m skiopt.analysis.sensitivity configs/experiments/gran_risa.yaml physics.mu \\
        0.02 0.03 0.04 0.06 0.08 --methods coach_line dp --out docs/reports/sensitivity_mu

For each value, a copy of the experiment (one seed, the chosen methods, the entry overridden) runs into
results/sens_<param>_<value>/ with the usual skip-if-done, so re-running only fills in what is missing.
Writes SENSITIVITY.md and sensitivity.png (time vs. value per method, with the real 76.00 s for the
Gran Risa) to --out. The DP is deterministic, so one seed is enough for it.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

from skiopt.config import load_config

ROOT = Path(__file__).resolve().parent.parent.parent


def nested(key: str, value) -> dict:
    out: dict = value
    for part in reversed(key.split(".")):
        out = {part: out}
    return out


def lookup(cfg: dict, key: str):
    for part in key.split("."):
        cfg = cfg[part]
    return cfg


def experiments(base_path: Path, key: str, values: list[float], methods: list[str], tmp: Path) -> list[Path]:
    from skiopt.config import deep_merge

    raw = yaml.safe_load(base_path.read_text())
    chosen = [m for m in raw["methods"] if m.get("label", m["name"]) in methods]
    paths = []
    for v in values:
        name = f"sens_{key.split('.')[-1]}_{v:g}"
        exp = deep_merge(raw, nested(key, float(v)))
        exp.update(experiment=name, methods=chosen, seeds=[0])
        path = tmp / f"{name}.yaml"
        path.write_text(yaml.safe_dump(exp))
        paths.append(path)
    return paths


def collect(results: Path, key: str, values: list[float]) -> dict:
    from skiopt.analysis.summarize import load_runs, method_label

    table: dict = {}
    for v in values:
        for m in load_runs(results / f"sens_{key.split('.')[-1]}_{v:g}"):
            clean = not m.get("gates_missed") and not m.get("hard_dnf")
            table.setdefault(method_label(m), {})[v] = (m["T"], m["best_score"], m.get("P_DNF", 0.0),
                                                        m["v_mean"] * 3.6, clean)
    return table


def write(table: dict, key: str, values: list[float], default, out: Path, reference_s: float | None) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from skiopt.analysis.figures import style
    from skiopt.analysis.plots import INK, INK_2, MUTED, SURFACE, _style

    out.mkdir(parents=True, exist_ok=True)
    lines = [f"# Sensitivity to `{key}`", "",
             f"Default `{key}` = {default}. One run per value and method; T = course time, "
             "E = expected score (time with the DNF risk priced in), v̄ = mean speed. "
             "✗ marks a line that misses a gate or hits a hard limit.", "",
             "| method | " + " | ".join(f"{v:g}" for v in values) + " |",
             "|---|" + "---|" * len(values)]
    for name, row in sorted(table.items()):
        cells = []
        for v in values:
            if v not in row:
                cells.append("—")
                continue
            T, E, P, vbar, clean = row[v]
            cells.append(f"T {T:.2f} s · E {E:.2f} · {vbar:.1f} km/h · P_DNF {100 * P:.1f}%" + ("" if clean else " ✗"))
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    (out / "SENSITIVITY.md").write_text("\n".join(lines) + "\n")

    fig, ax = plt.subplots(figsize=(8, 5), facecolor=SURFACE)
    for name, row in sorted(table.items()):
        xs = [v for v in values if v in row]
        colour, marker = style(name)
        ax.plot(xs, [row[v][0] for v in xs], color=colour, marker=marker, lw=1.6, label=name)
    if reference_s is not None:
        ax.axhline(reference_s, color=MUTED, lw=1, ls="--")
        ax.text(values[0], reference_s, f" real fastest run: {reference_s:.2f} s", fontsize=8, color=INK_2, va="bottom")
    ax.axvline(float(default), color=MUTED, lw=1, ls=":")
    ax.set_xlabel(f"{key} (default {default})")
    ax.set_ylabel("course time T (s)")
    ax.set_title(f"Sensitivity of the optimized time to {key}", loc="left", fontsize=10, color=INK)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_2)
    ax.grid(color="#e4e3df", lw=0.6)
    _style(ax)
    fig.savefig(out / "sensitivity.png", dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {out / 'SENSITIVITY.md'} and {out / 'sensitivity.png'}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("experiment", type=Path)
    parser.add_argument("key", help="dotted config key, e.g. physics.mu or sim3d.k_turn")
    parser.add_argument("values", type=float, nargs="+")
    parser.add_argument("--methods", nargs="+", default=["coach_line", "dp"])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--report-only", action="store_true", help="skip running; only collect and plot")
    args = parser.parse_args(argv)

    cfg = load_config(args.experiment)
    default = lookup(cfg, args.key)
    tmp = ROOT / "results" / "_sensitivity_configs"
    tmp.mkdir(parents=True, exist_ok=True)
    paths = experiments(args.experiment, args.key, args.values, args.methods, tmp)
    if not args.report_only:
        code = subprocess.call([sys.executable, "-m", "skiopt.batch", *map(str, paths), "--jobs", str(args.jobs)], cwd=ROOT)
        if code:
            print("some sensitivity runs failed; reporting what finished")
    table = collect(ROOT / cfg["results_dir"], args.key, args.values)
    reference = 76.00 if cfg["course"]["kind"] == "gran_risa" else None
    write(table, args.key, args.values, default, args.out, reference)
    return 0


if __name__ == "__main__":
    sys.exit(main())
