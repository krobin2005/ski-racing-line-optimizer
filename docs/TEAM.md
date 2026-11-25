# Team roles and GenAI disclosure

The rubric for this two-person project asks for documentation of each member's roles and responsibilities, plus full disclosure of GenAI use. This file covers both.

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

We did not divide the project into halves: we worked on every part together. Within that, each of us led where our strengths were:

- **Kyle Robin:** the project idea and the ski-racing knowledge behind it: the question, the course and race data, and checking the results and the race video against real racing.
- **Liam Baron:** the physics of the simulator, from the 2D dynamics to the 3D surface model.

Each of us will answer for the whole project in a separate interview.

### Decisions we made together

- Approving implementation plan v2 (Oct 6).
- Choosing the Gran Risa and the 21 Dec 2025 World Cup race as the real-world target.
- Making the 3D Gran Risa the project and the 2D model its validation stage.
- Cutting the neural controller to keep the scope on line optimization.

## GenAI disclosure

We used GenAI heavily for code, and disclosed it per the course's GenAI disclosure document:

| Tool | Used by | What for |
| --- | --- | --- |
| Claude Code (Anthropic) | Kyle's laptop, Nov 7 | Reviewing the repo against the rubric; drafting the README reframing and the Stage 3 plan; writing `skiopt/analysis/race_video.py` and its tests; regenerating the best Gran Risa run of each method to render the video; drafting this file. |
| Claude (Anthropic) | Both | Identifying commands and packages; turning our plain-English design into Python; drafting, refining, commenting and debugging the code and notebooks; help building the terrain data; rendering the visualizations; some README text. We did not use GenAI to brainstorm the project or its research questions. |

**How we made sure GenAI did not replace our understanding:** we designed the project, the experiments and the questions ourselves, checked every result against real racing, and walked through each module together before the interviews.
