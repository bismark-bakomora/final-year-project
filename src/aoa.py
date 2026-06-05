import numpy as np
from src.fitness import fitness_function

# ─────────────────────────────────────────
# ARITHMETIC OPTIMIZATION ALGORITHM (AOA)
# Paper Section 3.2.3
# Abualigah et al. 2021
#
# Inspired by arithmetic mathematical operators:
# Division, Multiplication, Addition, Subtraction
#
# Two phases controlled by MOP value:
# 1. Local Search Phase  (r1 > MOP)  — Exploration
#    - Division operator   (r3 < 0.5) Equation 19
#    - Multiplication operator (r3 >= 0.5) Equation 20
# 2. Fine-Tuning Phase   (r1 <= MOP) — Exploitation
#    - Subtraction operator (r2 < 0.5) Equation 21
#    - Addition operator    (r2 >= 0.5) Equation 22
#
# Role in hybrid: FINE-TUNING phase
# Starts from WOA best solution
# Performs precise final optimization
# Returns ultimate optimal hyperparameter set
# ─────────────────────────────────────────

class AOA:
    def __init__(self,
                 population_size=20,
                 max_iterations=10,
                 lower_bounds=None,
                 upper_bounds=None,
                 alpha=0.1,
                 mu=0.499):
        """
        Parameters
        ----------
        population_size : int
            Number of agents.
            Paper Table 3: ps = 20
        max_iterations : int
            Number of iterations.
            Paper: 10 iterations in hybrid mode
        lower_bounds : array-like
            Lower bounds for each hyperparameter.
        upper_bounds : array-like
            Upper bounds for each hyperparameter.
        alpha : float
            Control parameter for MOP calculation.
            Paper Table 3: alpha = 0.1
        mu : float
            Control parameter for position update.
            Paper Table 3: mu = 0.499
        """
        self.population_size = population_size
        self.max_iterations  = max_iterations
        self.lower_bounds    = np.array(lower_bounds)
        self.upper_bounds    = np.array(upper_bounds)
        self.dimension       = len(lower_bounds)
        self.alpha           = alpha   # paper: 0.1
        self.mu              = mu      # paper: 0.499

        # Best solution found
        self.best_pos   = None
        self.best_score = float('inf')

        # Small epsilon to prevent division by zero
        self.epsilon = 1e-10

        # Track convergence for Figure 8
        self.convergence_curve = []

    # ─────────────────────────────────────
    # INITIALIZE POPULATION
    # Paper Equation 17:
    # Xi = LB + rand(0,1) * (UB - LB)
    # Paper: "agent population is started with
    # 20 individuals around the best solution
    # from WOA"
    # ─────────────────────────────────────
    def _initialize_population(self, woa_best_pos):
        """
        Initialize agents around WOA best solution.
        Equation 17 from paper.
        First agent starts at WOA best position.
        """
        population = np.zeros(
            (self.population_size, self.dimension)
        )

        # First agent starts at WOA best solution
        population[0] = woa_best_pos.copy()

        # Remaining agents initialized randomly
        # using Equation 17
        for i in range(1, self.population_size):
            population[i] = (
                self.lower_bounds +
                np.random.random(self.dimension) *
                (self.upper_bounds - self.lower_bounds)
            )

        return population

    # ─────────────────────────────────────
    # CALCULATE MOP
    # Paper Equation 18:
    # MOP(t) = 1 - (t/Tmax)^alpha
    #
    # MOP controls phase transition:
    # High MOP → more fine-tuning (exploitation)
    # Low MOP  → more local search (exploration)
    # ─────────────────────────────────────
    def _calculate_MOP(self, t):
        """
        Mathematical Operator Proportion.
        Equation 18 from paper.

        MOP(t) = 1 - (t / Tmax)^alpha

        Controls balance between exploration
        and exploitation phases.
        alpha = 0.1 as per Table 3.
        """
        MOP = 1 - (t / self.max_iterations) ** self.alpha
        return MOP

    # ─────────────────────────────────────
    # LOCAL SEARCH PHASE (r1 > MOP)
    # Exploration using Division/Multiplication
    #
    # Division operator (r3 < 0.5):
    # Equation 19: X = Xbest / (MOP + eps)
    #                  * ((UB-LB)*mu + LB)
    #
    # Multiplication operator (r3 >= 0.5):
    # Equation 20: X = Xbest * MOP
    #                  * ((UB-LB)*mu + LB)
    # ─────────────────────────────────────
    def _local_search_phase(self, MOP):
        """
        Exploration phase — large scale search.
        Uses division and multiplication operators.

        Equation 19 (division):
        X = Xbest / (MOP + eps) * ((UB-LB)*mu + LB)

        Equation 20 (multiplication):
        X = Xbest * MOP * ((UB-LB)*mu + LB)
        """
        r3 = np.random.random()

        # Scaling factor used in both equations
        scale = ((self.upper_bounds - self.lower_bounds) *
                 self.mu + self.lower_bounds)

        if r3 < 0.5:
            # Division operator — Equation 19
            new_pos = (self.best_pos /
                      (MOP + self.epsilon) * scale)
        else:
            # Multiplication operator — Equation 20
            new_pos = self.best_pos * MOP * scale

        return new_pos

    # ─────────────────────────────────────
    # FINE-TUNING PHASE (r1 <= MOP)
    # Exploitation using Subtraction/Addition
    #
    # Subtraction operator (r2 < 0.5):
    # Equation 21: X = Xbest - MOP
    #                  * ((UB-LB)*mu + LB)
    #
    # Addition operator (r2 >= 0.5):
    # Equation 22: X = Xbest + MOP
    #                  * ((UB-LB)*mu + LB)
    # ─────────────────────────────────────
    def _fine_tuning_phase(self, MOP):
        """
        Exploitation phase — precise refinement.
        Uses subtraction and addition operators.

        Equation 21 (subtraction):
        X = Xbest - MOP * ((UB-LB)*mu + LB)

        Equation 22 (addition):
        X = Xbest + MOP * ((UB-LB)*mu + LB)
        """
        r2 = np.random.random()

        # Scaling factor used in both equations
        scale = ((self.upper_bounds - self.lower_bounds) *
                 self.mu + self.lower_bounds)

        if r2 < 0.5:
            # Subtraction operator — Equation 21
            new_pos = self.best_pos - MOP * scale
        else:
            # Addition operator — Equation 22
            new_pos = self.best_pos + MOP * scale

        return new_pos

    # ─────────────────────────────────────
    # MAIN OPTIMIZATION LOOP
    # Follows Figure 5 flowchart exactly
    # ─────────────────────────────────────
    def optimize(self, woa_best_pos, verbose=True):
        """
        Run AOA optimization.
        Follows Figure 5 flowchart from paper.

        Parameters
        ----------
        woa_best_pos : array, shape (9,)
            Best position from WOA phase.
            AOA starts fine-tuning from here.

        Returns
        -------
        best_position : array, shape (9,)
            Final optimal hyperparameter vector.
        best_fitness : float
            Best fitness value achieved.
        convergence_curve : list
            Fitness per iteration for plotting.
        """
        if verbose:
            print("\nAOA — Fine-Tuning Phase")
            print("=" * 40)

        # ── Initialize around WOA best solution ──
        population = self._initialize_population(
            woa_best_pos
        )

        # ── Set best to WOA result initially ──
        self.best_pos   = woa_best_pos.copy()
        self.best_score = fitness_function(woa_best_pos)

        # ── Calculate fitness for initial population ──
        # and determine best solution
        for i in range(self.population_size):
            score = fitness_function(population[i])
            if score < self.best_score:
                self.best_score = score
                self.best_pos   = population[i].copy()

        # ── Main iteration loop (t = 1 to Tmax) ──
        for t in range(1, self.max_iterations + 1):

            # ── Calculate MOP — Equation 18 ──
            MOP = self._calculate_MOP(t)

            # ── Update each agent position ──
            for i in range(self.population_size):

                # Generate r1 to decide phase
                r1 = np.random.random()

                if r1 > MOP:
                    # Local Search Phase — Exploration
                    # Equations 19 or 20
                    new_pos = self._local_search_phase(MOP)
                else:
                    # Fine-Tuning Phase — Exploitation
                    # Equations 21 or 22
                    new_pos = self._fine_tuning_phase(MOP)

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
                      f" | MOP={MOP:.3f}"
                      f" | Best fitness={self.best_score:.4f}"
                      f" | Best accuracy="
                      f"{(1-self.best_score)*100:.2f}%")

        if verbose:
            print(f"\nAOA Complete.")
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
def test_aoa():
    """Test AOA with 2 iterations and 3 agents."""
    import numpy as np
    from src.fitness import set_data
    from src.cnn_model import LOWER_BOUNDS, UPPER_BOUNDS

    print("Testing AOA with small population...")
    print("(2 iterations, 3 agents)")
    print("-" * 40)

    # Load data
    X_train = np.load('data/processed/X_train.npy')
    y_train = np.load('data/processed/y_train.npy')
    X_val   = np.load('data/processed/X_val.npy')
    y_val   = np.load('data/processed/y_val.npy')

    set_data(X_train, y_train, X_val, y_val)

    # Simulate WOA best position (Table 6)
    # [filters=2, kernel=4, pooling=1, neurons=3,
    #  dropout=0.313, lr=0.00015, batch=5, opt=0, epoch=36]
    woa_best = np.array(
        [2, 4, 1, 3, 0.313, 0.00015, 5, 0, 36],
        dtype=float
    )

    print(f"WOA best position: {woa_best}")
    print(f"alpha = 0.1, mu = 0.499")

    aoa = AOA(
        population_size=3,
        max_iterations=2,
        lower_bounds=LOWER_BOUNDS,
        upper_bounds=UPPER_BOUNDS,
        alpha=0.1,
        mu=0.499
    )

    best_pos, best_fitness, curve = aoa.optimize(
        woa_best_pos=woa_best,
        verbose=True
    )

    print(f"\nConvergence curve: {curve}")
    print(f"Best position:     {best_pos}")
    print(f"Best fitness:      {best_fitness:.4f}")
    print(f"Best accuracy:     {(1-best_fitness)*100:.2f}%")
    print("\nAOA test PASSED")


if __name__ == "__main__":
    test_aoa()