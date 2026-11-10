# Team roles and GenAI disclosure

> **Draft (Nov 7).** The rubric for this two-person project asks for "extensive documentation on each member's roles/responsibilities" so each member can be graded separately, plus full disclosure of GenAI use. Everything marked **TODO** has to be written by Kyle and Liam.

## How we worked

**We built the project together.** Both of us were present for every planning session, coding session and prompt: we designed, wrote and checked everything side by side. Because of Claude usage limits we worked on two laptops at different times, and whose laptop a commit came from depended on which of us had usage left that day, not on who did the work. **The author name on a commit says whose computer it was made on, not who did the work**, so the git log alone is not a record of who did what.

## What we built (jointly)

- **Planning:** the v2 implementation plan and its v2.1 revisions (`docs/PLAN.md`, Oct 6); the 3D roadmap (`docs/ROADMAP_3D.md`).
- **Stage 1, 2D validation:** the simulator, evaluator, runner and logger; the FIS course rules, validator and generator; the switchable DNF (crash) model and risk map; the spline and apex-and-arc lines; the optimizers (CMA-ES, our own GA, Adam, the hybrid, baselines); the trap course; the heading-limiter physics fix and its regression tests.
- **Stage 2, the Gran Risa in 3D:** the terrain pipeline, the 3D simulator, the 52-gate course set, the calibration to the real 76.00 s, the DP baseline, the batch runner and the official FIS race data.
- **Experiments:** the final 10-seed runs (220 jobs), the sensitivity studies and ablations 1–4.
- **Stage 3, the race simulation:** the broadcast-style race video (`skiopt/analysis/race_video.py`, `docs/media/gran_risa_race.mp4`), with the gates, dye lines, start house and finish area of the real Gran Risa.
- **Write-up:** the report (`docs/REPORT.md`), the README and the walkthrough notebook.

## Individual contributions

**There is no split to report.** We did not divide the project into halves or give parts to one person. Each of us did every part of it together with the other, so each of us contributed to, and can explain, all of the work listed above. Each of us will answer for the whole project in a separate interview.

### Decisions we made together

**TODO:** for example, approving plan v2 (Oct 6), choosing the Gran Risa and the 21 Dec 2025 race, making the 3D Gran Risa the project and 2D the validation stage, and cutting the neural controller.

## GenAI disclosure

**TODO (both):** list every tool and what it was used for, per the course's GenAI disclosure document. Known so far:

| Tool | Used by | What for |
| --- | --- | --- |
| Claude Code (Anthropic) | Kyle's laptop, Nov 7 | Reviewing the repo against the rubric; drafting the README reframing and the Stage 3 plan; writing `skiopt/analysis/race_video.py` and its tests; regenerating the best Gran Risa run of each method to render the video; drafting this file. |
| **TODO** | | |

**How we made sure GenAI did not replace our understanding (TODO):** for example, walking through each module before the interview; the parts each of you can explain line by line.
