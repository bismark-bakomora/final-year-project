import numpy as np
from src.fitness import fitness_function

# ─────────────────────────────────────────
# GREY WOLF OPTIMIZER (GWO)
# Paper Section 3.2.1
# Mirjalili et al. 2014
#
# Simulates the hunting behaviour of grey wolves
# Social hierarchy: Alpha > Beta > Delta > Omega
#
# Three phases:
# 1. Tracking and approaching prey
# 2. Surrounding and immobilizing prey
# 3. Attacking prey
#
# Role in hybrid: EXPLORATION phase
# Performs global search across hyperparameter space
# Best solution passed to WOA for exploitation
# ─────────────────────────────────────────

class GWO:
    def __init__(self,
                 population_size=20,
                 max_iterations=10,
                 lower_bounds=None,
                 upper_bounds=None):
        """
        Parameters
        ----------
        population_size : int
            Number of wolves in pack.
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

        # Best solutions found — paper tracks top 3
        self.alpha_pos   = None  # best solution
        self.alpha_score = float('inf')

        self.beta_pos    = None  # second best
        self.beta_score  = float('inf')

        self.delta_pos   = None  # third best
        self.delta_score = float('inf')

        # Track convergence for Figure 8
        self.convergence_curve = []

    # ─────────────────────────────────────
    # STEP 1 — Initialize wolf population
    # Paper: "20-individual wolf population
    # is initiated randomly"
    # ─────────────────────────────────────
    def _initialize_population(self):
        """
        Randomly initialize wolves within bounds.
        Each wolf is a candidate hyperparameter vector.
        """
        population = np.zeros(
            (self.population_size, self.dimension)
        )
        for i in range(self.population_size):
            for d in range(self.dimension):
                population[i, d] = (
                    self.lower_bounds[d] +
                    np.random.random() *
                    (self.upper_bounds[d] - self.lower_bounds[d])
                )
        return population

    # ─────────────────────────────────────
    # STEP 2 — Update leadership hierarchy
    # Alpha = best, Beta = 2nd, Delta = 3rd
    # ─────────────────────────────────────
    def _update_leadership(self, population, fitness_values):
        """
        Sort wolves by fitness and update
        alpha, beta, delta positions.
        Paper: solutions ranked by performance.
        """
        for i in range(self.population_size):
            fitness = fitness_values[i]

            # Update alpha — best solution
            if fitness < self.alpha_score:
                self.alpha_score = fitness
                self.alpha_pos   = population[i].copy()

            # Update beta — second best
            elif fitness < self.beta_score:
                self.beta_score = fitness
                self.beta_pos   = population[i].copy()

            # Update delta — third best
            elif fitness < self.delta_score:
                self.delta_score = fitness
                self.delta_pos   = population[i].copy()

    # ─────────────────────────────────────
    # STEP 3 — Calculate A and C vectors
    # Paper Equations 4 and 5:
    # A = 2a * r1 - a
    # C = 2 * r2
    # ─────────────────────────────────────
    def _calculate_A_C(self, a):
        """
        Calculate coefficient vectors A and C.

        Equation 4: A = 2a * r1 - a
        Equation 5: C = 2 * r2

        A controls exploration vs exploitation:
        |A| > 1 → exploration (search globally)
        |A| < 1 → exploitation (converge to prey)

        C provides random weights to avoid
        local optima entrapment.
        """
        r1 = np.random.random(self.dimension)
        r2 = np.random.random(self.dimension)

        A = 2 * a * r1 - a   # Equation 4
        C = 2 * r2            # Equation 5

        return A, C

    # ─────────────────────────────────────
    # STEP 4 — Calculate D vectors
    # Paper Equation 6:
    # D_alpha = |C1 * X_alpha - X|
    # D_beta  = |C2 * X_beta  - X|
    # D_delta = |C3 * X_delta - X|
    # ─────────────────────────────────────
    def _calculate_D(self, wolf_pos,
                     A1, C1, A2, C2, A3, C3):
        """
        Calculate distance vectors from each
        leader to the current wolf position.
        Equation 6 from the paper.
        """
        D_alpha = abs(C1 * self.alpha_pos - wolf_pos)
        D_beta  = abs(C2 * self.beta_pos  - wolf_pos)
        D_delta = abs(C3 * self.delta_pos - wolf_pos)

        return D_alpha, D_beta, D_delta

    # ─────────────────────────────────────
    # STEP 5 — Calculate X vectors
    # Paper Equation 7:
    # X1 = X_alpha - A1 * D_alpha
    # X2 = X_beta  - A2 * D_beta
    # X3 = X_delta - A3 * D_delta
    # ─────────────────────────────────────
    def _calculate_X(self, A1, D_alpha,
                           A2, D_beta,
                           A3, D_delta):
        """
        Calculate step vectors toward each leader.
        Equation 7 from the paper.
        """
        X1 = self.alpha_pos - A1 * D_alpha
        X2 = self.beta_pos  - A2 * D_beta
        X3 = self.delta_pos - A3 * D_delta

        return X1, X2, X3

    # ─────────────────────────────────────
    # STEP 6 — Update wolf position
    # Paper Equation 8:
    # X(t+1) = (X1 + X2 + X3) / 3
    # ─────────────────────────────────────
    def _update_position(self, X1, X2, X3):
        """
        New position is average of steps
        toward all three leaders.
        Equation 8 from the paper.
        Clipped to stay within bounds.
        """
        new_pos = (X1 + X2 + X3) / 3.0   # Equation 8

        # Clip to search bounds
        new_pos = np.clip(
            new_pos,
            self.lower_bounds,
            self.upper_bounds
        )
        return new_pos

    # ─────────────────────────────────────
    # MAIN OPTIMIZATION LOOP
    # Follows Figure 3 flowchart exactly
    # ─────────────────────────────────────
    def optimize(self, verbose=True):
        """
        Run GWO optimization.
        Follows Figure 3 flowchart from paper.

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
            print("\nGWO — Exploration Phase")
            print("=" * 40)

        # ── Initialize wolf population ──
        population = self._initialize_population()

        # ── Set iteration counter t = 1 ──
        for t in range(1, self.max_iterations + 1):

            # ── Calculate fitness for each wolf ──
            fitness_values = np.zeros(self.population_size)
            for i in range(self.population_size):
                fitness_values[i] = fitness_function(
                    population[i]
                )

            # ── Update alpha, beta, delta leaders ──
            self._update_leadership(population,
                                    fitness_values)

            # ── Update 'a' — Equation 9 ──
            # a decreases linearly from 2 to 0
            a = 2 - t * (2 / self.max_iterations)

            # ── Update each wolf position ──
            for i in range(self.population_size):

                # Calculate 3 sets of A and C vectors
                # (one per leader: alpha, beta, delta)
                A1, C1 = self._calculate_A_C(a)
                A2, C2 = self._calculate_A_C(a)
                A3, C3 = self._calculate_A_C(a)

                # Calculate D vectors — Equation 6
                D_alpha, D_beta, D_delta = self._calculate_D(
                    population[i], A1, C1, A2, C2, A3, C3
                )

                # Calculate X vectors — Equation 7
                X1, X2, X3 = self._calculate_X(
                    A1, D_alpha,
                    A2, D_beta,
                    A3, D_delta
                )

                # Update wolf position — Equation 8
                population[i] = self._update_position(
                    X1, X2, X3
                )

            # ── Record best fitness this iteration ──
            self.convergence_curve.append(self.alpha_score)

            if verbose:
                print(f"  Iteration {t:2d}/{self.max_iterations}"
                      f" | a={a:.3f}"
                      f" | Best fitness={self.alpha_score:.4f}"
                      f" | Best accuracy="
                      f"{(1-self.alpha_score)*100:.2f}%")

        if verbose:
            print(f"\nGWO Complete.")
            print(f"  Best fitness:  {self.alpha_score:.4f}")
            print(f"  Best accuracy: "
                  f"{(1-self.alpha_score)*100:.2f}%")
            print(f"  Best position: {self.alpha_pos}")

        return (self.alpha_pos,
                self.alpha_score,
                self.convergence_curve)


# ─────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────
def test_gwo():
    """Test GWO with 2 iterations and 3 wolves."""
    import numpy as np
    from src.fitness import set_data
    from src.cnn_model import LOWER_BOUNDS, UPPER_BOUNDS

    print("Testing GWO with small population...")
    print("(2 iterations, 3 wolves)")
    print("-" * 40)

    # Load data
    X_train = np.load('data/processed/X_train.npy')
    y_train = np.load('data/processed/y_train.npy')
    X_val   = np.load('data/processed/X_val.npy')
    y_val   = np.load('data/processed/y_val.npy')

    set_data(X_train, y_train, X_val, y_val)

    # Small test — 2 iterations, 3 wolves
    gwo = GWO(
        population_size=3,
        max_iterations=2,
        lower_bounds=LOWER_BOUNDS,
        upper_bounds=UPPER_BOUNDS
    )

    best_pos, best_fitness, curve = gwo.optimize(verbose=True)

    print(f"\nConvergence curve: {curve}")
    print(f"Best position:     {best_pos}")
    print(f"Best fitness:      {best_fitness:.4f}")
    print("\nGWO test PASSED")


if __name__ == "__main__":
    test_gwo()