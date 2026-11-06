"""Build every final report from saved results (no simulation):

    python -m skiopt.analysis.report_all [--results results] [--out docs/reports]

For each final experiment present under results/: SUMMARY.md + summary.csv, the general figures
(convergence, distance / mean-speed plane, turn radii, each method's line on the course), and for the
Gran Risa the 3D figures (line on the terrain map, run trace, split times). Output goes to
docs/reports/final_<experiment>/.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

FINAL = ("final_easy", "final_trap", "gran_risa")


def build(results: Path, out: Path) -> list[Path]:
    from skiopt.analysis import figures, gran_risa_figures, summarize

    written = []
    for name in FINAL:
        exp = results / name
        if not exp.exists() or not list(exp.glob("*/seed_*/manifest.json")):
            print(f"skip {name}: no results")
            continue
        dest = out / (name if name.startswith("final_") else f"final_{name}")
        dest.mkdir(parents=True, exist_ok=True)
        summarize.summarize(exp)
        for f in ("SUMMARY.md", "summary.csv"):
            shutil.copy(exp / f, dest / f)
        written += figures.make_figures(exp, dest)
        if name == "gran_risa":
            gran_risa_figures.main([str(exp), "--out", str(dest)])
        print(f"built {dest}")
    return written


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--out", type=Path, default=Path("docs/reports"))
    args = parser.parse_args(argv)
    build(args.results, args.out)


if __name__ == "__main__":
    main()
