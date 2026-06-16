import numpy as np
from src.gwo import GWO
from src.woa import WOA
from src.aoa import AOA
from src.rime import RIME

# ─────────────────────────────────────────
# STANDALONE OPTIMIZER RUNNERS
# Paper Section 4.2:
# "For comparison purposes, GWO, WOA, AOA and
# RIME algorithms were run independently"
#
# Unlike the hybrid, these start from RANDOM
# positions rather than inheriting from a
# previous stage.
# ─────────────────────────────────────────

def _random_position(lower_bounds, upper_bounds):
    """Generate one random hyperparameter vector."""
    lb = np.array(lower_bounds)
    ub = np.array(upper_bounds)
    return lb + np.random.random(len(lb)) * (ub - lb)


def run_standalone_gwo(population_size, max_iterations,
                       lower_bounds, upper_bounds,
                       verbose=True):
    """Run GWO independently (GWO-CNN in Table 7)."""
    gwo = GWO(population_size, max_iterations,
              lower_bounds, upper_bounds)
    return gwo.optimize(verbose=verbose)


def run_standalone_woa(population_size, max_iterations,
                       lower_bounds, upper_bounds,
                       verbose=True):
    """Run WOA independently (WOA-CNN in Table 7)."""
    woa = WOA(population_size, max_iterations,
              lower_bounds, upper_bounds)
    start_pos = _random_position(lower_bounds, upper_bounds)
    # gwo_best_fitness=None forces evaluation of
    # the random starting position
    return woa.optimize(gwo_best_pos=start_pos,
                        gwo_best_fitness=None,
                        verbose=verbose)


def run_standalone_aoa(population_size, max_iterations,
                       lower_bounds, upper_bounds,
                       verbose=True):
    """Run AOA independently (AOA-CNN in Table 7)."""
    aoa = AOA(population_size, max_iterations,
              lower_bounds, upper_bounds,
              alpha=0.1, mu=0.499)
    start_pos = _random_position(lower_bounds, upper_bounds)
    return aoa.optimize(woa_best_pos=start_pos,
                        woa_best_fitness=None,
                        verbose=verbose)


def run_standalone_rime(population_size, max_iterations,
                        lower_bounds, upper_bounds,
                        verbose=True):
    """Run RIME independently (RIME-CNN in Table 7)."""
    rime = RIME(population_size, max_iterations,
                 lower_bounds, upper_bounds)
    return rime.optimize(verbose=verbose)