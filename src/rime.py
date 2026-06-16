import numpy as np
from src.fitness import fitness_function

# ─────────────────────────────────────────
# RIME ALGORITHM
# Su et al. 2023 — "RIME: A physics-based optimization"
# Paper Table 3 lists RIME as a comparison baseline
#
# Two mechanisms:
# 1. Soft-rime search   — exploration, agents drift
#    toward best solution with decreasing randomness
# 2. Hard-rime puncture — exploitation, agents that
#    are performing poorly snap directly to best
#
# Positive greedy selection: new position only kept
# if it improves fitness
#
# Role: standalone comparison baseline (Table 7)
# ─────────────────────────────────────────

class RIME:
    def __init__(self,
                 population_size=20,
                 max_iterations=10,
                 lower_bounds=None,
                 upper_bounds=None,
                 w=5):
        """
        Parameters
        ----------
        population_size : int
            Number of rime agents. Paper: ps = 20
        max_iterations : int
            Number of iterations.
        lower_bounds, upper_bounds : array-like
            Search space bounds.
        w : int
            Soft-rime control constant (standard = 5).
        """
        self.population_size = population_size
        self.max_iterations  = max_iterations
        self.lower_bounds    = np.array(lower_bounds)
        self.upper_bounds    = np.array(upper_bounds)
        self.dimension       = len(lower_bounds)
        self.w = w

        self.best_pos   = None
        self.best_score = float('inf')
        self.convergence_curve = []

    def _initialize_population(self):
        population = np.zeros(
            (self.population_size, self.dimension)
        )
        for i in range(self.population_size):
            population[i] = (
                self.lower_bounds +
                np.random.random(self.dimension) *
                (self.upper_bounds - self.lower_bounds)
            )
        return population

    def optimize(self, verbose=True):
        """
        Run RIME optimization.

        Returns
        -------
        best_position, best_fitness, convergence_curve
        """
        if verbose:
            print("\nRIME — Comparison Baseline")
            print("=" * 40)

        population = self._initialize_population()

        # Initial fitness evaluation
        fitness_values = np.array([
            fitness_function(population[i])
            for i in range(self.population_size)
        ])

        best_idx = np.argmin(fitness_values)
        self.best_pos   = population[best_idx].copy()
        self.best_score = fitness_values[best_idx]

        for t in range(1, self.max_iterations + 1):

            # Rime factor — increases convergence pressure
            E = np.sqrt(t / self.max_iterations)

            # Normalize fitness for hard-rime probability
            # (lower fitness = better = closer to 1)
            fmax, fmin = fitness_values.max(), fitness_values.min()
            if fmax - fmin < 1e-12:
                norm_fitness = np.ones(self.population_size)
            else:
                norm_fitness = (fmax - fitness_values) / (fmax - fmin)

            new_population = population.copy()

            for i in range(self.population_size):
                for j in range(self.dimension):

                    # ── Soft-rime search ──
                    r1 = np.random.uniform(-1, 1)
                    r2 = np.random.random()

                    if r2 < E:
                        theta = np.pi * t / (10 * self.max_iterations)
                        beta = (1 - np.round(t * self.w /
                                self.max_iterations) / self.w)
                        h = np.random.random()
                        new_population[i, j] = (
                            self.best_pos[j] +
                            r1 * np.cos(theta) * beta *
                            (h * (self.upper_bounds[j] -
                                  self.lower_bounds[j]) +
                             self.lower_bounds[j])
                        )

                    # ── Hard-rime puncture ──
                    r3 = np.random.random()
                    if r3 < norm_fitness[i]:
                        new_population[i, j] = self.best_pos[j]

                new_population[i] = np.clip(
                    new_population[i],
                    self.lower_bounds,
                    self.upper_bounds
                )

            # ── Positive greedy selection ──
            for i in range(self.population_size):
                new_score = fitness_function(new_population[i])
                if new_score < fitness_values[i]:
                    population[i]      = new_population[i]
                    fitness_values[i]  = new_score
                    if new_score < self.best_score:
                        self.best_score = new_score
                        self.best_pos   = new_population[i].copy()

            self.convergence_curve.append(self.best_score)

            if verbose:
                print(f"  Iteration {t:2d}/{self.max_iterations}"
                      f" | E={E:.3f}"
                      f" | Best fitness={self.best_score:.4f}"
                      f" | Best accuracy="
                      f"{(1-self.best_score)*100:.2f}%")

        if verbose:
            print(f"\nRIME Complete.")
            print(f"  Best fitness:  {self.best_score:.4f}")
            print(f"  Best accuracy: "
                  f"{(1-self.best_score)*100:.2f}%")

        return (self.best_pos,
                self.best_score,
                self.convergence_curve)


if __name__ == "__main__":
    import numpy as np
    from src.fitness import set_data
    from src.cnn_model import LOWER_BOUNDS, UPPER_BOUNDS

    print("Testing RIME with small population...")
    X_train = np.load('data/processed/X_train.npy')
    y_train = np.load('data/processed/y_train.npy')
    X_val   = np.load('data/processed/X_val.npy')
    y_val   = np.load('data/processed/y_val.npy')
    set_data(X_train, y_train, X_val, y_val)

    rime = RIME(population_size=3, max_iterations=2,
                 lower_bounds=LOWER_BOUNDS,
                 upper_bounds=UPPER_BOUNDS)
    best_pos, best_fitness, curve = rime.optimize(verbose=True)
    print(f"\nBest fitness: {best_fitness:.4f}")
    print("RIME test PASSED")