# Team roles and GenAI disclosure

> **Draft (Nov 7).** The rubric for this two-person project asks for "extensive documentation on each member's roles/responsibilities" so each member can be graded separately, plus full disclosure of GenAI use. Below, the *From the git history* parts are facts taken from the commit log. Everything marked **TODO** has to be written by Kyle and Liam: only you know who designed, decided or checked what outside git.

## Who did what

### Kyle Robin

**From the git history:**
- Project setup: pinned requirements, `.gitignore` (Oct 6).
- README and the approved v2 implementation plan, `docs/PLAN.md` (Oct 6).
- Physics bug fix: the heading limiter was weakening every turn. Fixed in `skiopt/dynamics.py`, with regression tests in `tests/test_physics.py` and `tests/test_runner.py` (Oct 6).
- Stage 3, the race simulation (Nov 7): `skiopt/analysis/race_video.py`, `tests/test_race_video.py` and `docs/media/gran_risa_race.mp4`. It is a broadcast-style 3D replay of every method's best line on the Gran Risa, raced against Schwarz's pace and the real top 5.
- Reframed the README around the 3D Gran Risa, and added the Stage 3 plan to `docs/ROADMAP_3D.md` (Nov 7).

**TODO (Kyle):** project idea and ski-racing expertise; course and equipment choices (e.g. the Völkl Racetiger GS 193 / Lange ZB setup in §2.2 of the report); reviewing results; presentations; anything else not in git.

### Liam Baron

**From the git history:**
- Plan v2.1 revisions (Oct 6).
- Days 1–8 of the 2D build: the core simulator, evaluator, runner and logger; FIS course rules, validator and generator; the switchable DNF model and risk map; the spline and apex-and-arc lines; the optimizers; the trap course.
- The 3D stage: the Gran Risa terrain pipeline, the 3D simulator, the course set, calibration, the DP baseline, the batch runner and the official FIS race data.
- The final 10-seed runs (220 jobs), the sensitivity studies and ablations 1–4.
- The report draft (`docs/REPORT.md`), the report outline and the 3D roadmap.

**TODO (Liam):** confirm, and add anything not in git.

### Together

**TODO:** the decisions you made jointly, e.g. the plan v2 approval (Oct 6), the choice of the Gran Risa and the Dec 2025 race, and cutting the neural controller.

## GenAI disclosure

**TODO (both):** list every tool and what it was used for, per the course's GenAI disclosure document. Known so far:

| Tool | Used by | What for |
| --- | --- | --- |
| Claude Code (Anthropic) | Kyle, Nov 7 | Reviewing the repo against the rubric; drafting the README reframing and the Stage 3 plan; writing `skiopt/analysis/race_video.py` and its tests; regenerating the best Gran Risa run of each method to render the video; drafting this file. |
| **TODO** | | |

**How we made sure GenAI did not replace our understanding (TODO):** for example, walking through each module before the interview; the parts each of you can explain line by line.
