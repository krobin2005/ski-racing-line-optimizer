"""Method name -> optimizer function, as used in experiment files."""

from skiopt.optim.baselines import fall_line, random_search
from skiopt.optim.evo import cmaes, cmaes_smooth, ga
from skiopt.optim.gradient import adam
from skiopt.optim.hybrid import hybrid

METHODS = {
    "fall_line": fall_line,
    "random_search": random_search,
    "adam": adam,
    "cmaes": cmaes,
    "cmaes_smooth": cmaes_smooth,
    "ga": ga,
    "hybrid": hybrid,
}
