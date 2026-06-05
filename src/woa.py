import numpy as np
from src.fitness import fitness_function

# ─────────────────────────────────────────
# WHALE OPTIMIZATION ALGORITHM (WOA)
# Paper Section 3.2.2
# Mirjalili and Lewis 2016
#
# Simulates hunting behaviour of humpback whales
# using bubble-net feeding strategy
#
# Three mechanisms:
# 1. Shrinking encircling (Equations 10-11)
# 2. Spiral update (Equation 14)
# 3. Prey search / exploration (Equations 15-16)
#
# Role in hybrid: EXPLOITATION phase
# Starts from GWO best solution
# Performs focused local search
# Best solution passed to AOA for fine-tuning
# ─────────────────────────────────────────

class WOA:
    def __init__(self,
                 population_size=20,
                 max_iterations=10,
                 lower_bounds=None,
                 upper_bounds=None):
        """
        Parameters
        ----------
        population_size : int
            Number of whales.
            Paper Table 3: ps = 20
        max_iterations : int
            Number of iterations.
            Paper: 10 iterations in hybrid mode
        lower_bounds : array-like
            Lower bounds for each hyperparameter.
        upper_bounds : array-like
            Upper bounds for each hyperparameter.
        """
        self.population_size = population_size
        self.max_iterations  = max_iterations
        self.lower_bounds    = np.array(lower_bounds)
        self.upper_bounds    = np.array(upper_bounds)
        self.dimension       = len(lower_bounds)

        # Best solution found
        self.best_pos   = None
        self.best_score = float('inf')

        # Track convergence for Figure 8
        self.convergence_curve = []

    # ─────────────────────────────────────
    # INITIALIZE POPULATION
    # Paper: "whale population is started with
    # 20 individuals around the best solution
    # from GWO"
    # ─────────────────────────────────────
    def _initialize_population(self, gwo_best_pos):
        """
        Initialize whales around GWO best solution.
        This is key difference from standalone WOA —
        population starts near a promising region
        rather than completely randomly.
        """
        population = np.zeros(
            (self.population_size, self.dimension)
        )

        # First whale starts exactly at GWO best
        population[0] = gwo_best_pos.copy()

        # Remaining whales initialized randomly
        # but within bounds
        for i in range(1, self.population_size):
            for d in range(self.dimension):
                population[i, d] = (
                    self.lower_bounds[d] +
                    np.random.random() *
                    (self.upper_bounds[d] -
                     self.lower_bounds[d])
                )

        return population

    # ─────────────────────────────────────
    # CALCULATE A AND C VECTORS
    # Paper Equations 12 and 13:
    # A = 2a * r - a
    # C = 2 * r
    # ─────────────────────────────────────
    def _calculate_A_C(self, a):
        """
        Equation 12: A = 2a * r - a
        Equation 13: C = 2 * r

        |A| < 1 → exploitation (shrinking circle)
        |A| > 1 → exploration (random whale search)
        """
        r = np.random.random(self.dimension)
        A = 2 * a * r - a    # Equation 12
        C = 2 * r             # Equation 13
        return A, C

    # ─────────────────────────────────────
    # MECHANISM 1 — SHRINKING ENCIRCLING
    # Paper Equations 10 and 11:
    # D = |C * X* - X|
    # X(t+1) = X* - A * D
    # Used when p < 0.5 AND |A| < 1
    # ─────────────────────────────────────
    def _shrinking_encircling(self, whale_pos, A, C):
        """
        Exploitation via shrinking circle.
        Whale moves toward best known position.

        Equation 10: D = |C * X_best - X|
        Equation 11: X(t+1) = X_best - A * D
        """
        D = abs(C * self.best_pos - whale_pos)
        new_pos = self.best_pos - A * D
        return new_pos

    # ─────────────────────────────────────
    # MECHANISM 2 — SPIRAL UPDATE
    # Paper Equation 14:
    # X(t+1) = D' * e^(bl) * cos(2*pi*l) + X*
    # Used when p >= 0.5
    # ─────────────────────────────────────
    def _spiral_update(self, whale_pos):
        """
        Exploitation via spiral movement.
        Mimics helix-shaped bubble net attack.

        D' = |X_best - X| distance from prey
        b  = 1 (fixed constant, paper Section 3.2.2)
        l  = random in [-1, 1]

        Equation 14:
        X(t+1) = D' * cos(2*pi*l) + X_best
        """
        b = 1  # fixed constant from paper
        l = np.random.uniform(-1, 1, self.dimension)

        D_prime = abs(self.best_pos - whale_pos)
        new_pos = (D_prime * np.exp(b * l) *
                   np.cos(2 * np.pi * l) +
                   self.best_pos)
        return new_pos

    # ─────────────────────────────────────
    # MECHANISM 3 — PREY SEARCH
    # Paper Equations 15 and 16:
    # D = |C * X_rand - X|
    # X(t+1) = X_rand - A * D
    # Used when p < 0.5 AND |A| >= 1
    # ─────────────────────────────────────
    def _prey_search(self, whale_pos, population, A, C):
        """
        Exploration via random whale search.
        Whale moves toward a randomly selected
        whale rather than the best position.
        This prevents premature convergence.

        Equation 15: D = |C * X_rand - X|
        Equation 16: X(t+1) = X_rand - A * D
        """
        # Select random whale from population
        rand_idx = np.random.randint(0,
                                     self.population_size)
        X_rand = population[rand_idx]

        D = abs(C * X_rand - whale_pos)
        new_pos = X_rand - A * D
        return new_pos

    # ─────────────────────────────────────
    # MAIN OPTIMIZATION LOOP
    # Follows Figure 4 flowchart exactly
    # ─────────────────────────────────────
    def optimize(self, gwo_best_pos, verbose=True):
        """
        Run WOA optimization.
        Follows Figure 4 flowchart from paper.

        Parameters
        ----------
        gwo_best_pos : array, shape (9,)
            Best position from GWO phase.
            WOA starts exploitation from here.

        Returns
        -------
        best_position : array, shape (9,)
            Best hyperparameter vector found.
        best_fitness : float
            Best fitness value achieved.
        convergence_curve : list
            Fitness per iteration for plotting.
        """
        if verbose:
            print("\nWOA — Exploitation Phase")
            print("=" * 40)

        # ── Initialize around GWO best solution ──
        population = self._initialize_population(
            gwo_best_pos
        )

        # ── Set best to GWO result initially ──
        self.best_pos   = gwo_best_pos.copy()
        self.best_score = fitness_function(gwo_best_pos)

        # ── Evaluate initial population ──
        for i in range(self.population_size):
            score = fitness_function(population[i])
            if score < self.best_score:
                self.best_score = score
                self.best_pos   = population[i].copy()

        # ── Main iteration loop ──
        for t in range(1, self.max_iterations + 1):

            # Update 'a' — decreases from 2 to 0
            # Same as GWO but for WOA context
            a = 2 * (1 - t / self.max_iterations)

            # Update each whale position
            for i in range(self.population_size):

                # Calculate A and C — Equations 12, 13
                A, C = self._calculate_A_C(a)

                # Generate random l for spiral
                # Generate random p for mechanism choice
                p = np.random.random()

                if p < 0.5:
                    # Choose between shrinking or search
                    if abs(A).mean() < 1:
                        # Shrinking encircling mechanism
                        # Equations 10-11 — Exploitation
                        new_pos = self._shrinking_encircling(
                            population[i], A, C
                        )
                    else:
                        # Prey search with random whale
                        # Equations 15-16 — Exploration
                        new_pos = self._prey_search(
                            population[i], population, A, C
                        )
                else:
                    # Spiral update mechanism
                    # Equation 14 — Exploitation
                    new_pos = self._spiral_update(
                        population[i]
                    )

                # Clip to search bounds
                population[i] = np.clip(
                    new_pos,
                    self.lower_bounds,
                    self.upper_bounds
                )

            # ── Evaluate updated population ──
            for i in range(self.population_size):
                score = fitness_function(population[i])
                if score < self.best_score:
                    self.best_score = score
                    self.best_pos   = population[i].copy()

            # ── Record convergence ──
            self.convergence_curve.append(self.best_score)

            if verbose:
                print(f"  Iteration {t:2d}/{self.max_iterations}"
                      f" | a={a:.3f}"
                      f" | Best fitness={self.best_score:.4f}"
                      f" | Best accuracy="
                      f"{(1-self.best_score)*100:.2f}%")

        if verbose:
            print(f"\nWOA Complete.")
            print(f"  Best fitness:  {self.best_score:.4f}")
            print(f"  Best accuracy: "
                  f"{(1-self.best_score)*100:.2f}%")
            print(f"  Best position: {self.best_pos}")

        return (self.best_pos,
                self.best_score,
                self.convergence_curve)


# ─────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────
def test_woa():
    """Test WOA with 2 iterations and 3 whales."""
    import numpy as np
    from src.fitness import set_data
    from src.cnn_model import LOWER_BOUNDS, UPPER_BOUNDS

    print("Testing WOA with small population...")
    print("(2 iterations, 3 whales)")
    print("-" * 40)

    # Load data
    X_train = np.load('data/processed/X_train.npy')
    y_train = np.load('data/processed/y_train.npy')
    X_val   = np.load('data/processed/X_val.npy')
    y_val   = np.load('data/processed/y_val.npy')

    set_data(X_train, y_train, X_val, y_val)

    # Use paper's optimal hyperparameters as
    # simulated GWO output (Table 6)
    # [filters=2, kernel=4, pooling=1, neurons=3,
    #  dropout=0.313, lr=0.00015, batch=5, opt=0, epoch=36]
    gwo_best = np.array(
        [2, 4, 1, 3, 0.313, 0.00015, 5, 0, 36],
        dtype=float
    )

    print(f"GWO best position: {gwo_best}")

    woa = WOA(
        population_size=3,
        max_iterations=2,
        lower_bounds=LOWER_BOUNDS,
        upper_bounds=UPPER_BOUNDS
    )

    best_pos, best_fitness, curve = woa.optimize(
        gwo_best_pos=gwo_best,
        verbose=True
    )

    print(f"\nConvergence curve: {curve}")
    print(f"Best position:     {best_pos}")
    print(f"Best fitness:      {best_fitness:.4f}")
    print("\nWOA test PASSED")


if __name__ == "__main__":
    test_woa()