import numpy as np
import time
from src.gwo import GWO
from src.woa import WOA
from src.aoa import AOA
from src.fitness import set_data, reset_history
from src.cnn_model import (
    LOWER_BOUNDS, UPPER_BOUNDS,
    decode_hyperparameters, build_cnn, train_cnn
)

# ─────────────────────────────────────────
# HYBRID GWO-WOA-AOA OPTIMIZER
# Paper Section 3.2.4 and Figure 6
#
# Sequential three-stage hybrid:
# Stage 1 — GWO: Exploration (global search)
# Stage 2 — WOA: Exploitation (local refinement)
# Stage 3 — AOA: Fine-tuning (precision enhancement)
#
# Paper Table 3 parameters:
# All sub-algorithms: ps=20, mi=10 (hybrid mode)
# GWO: a=2→0, A=[-a,a], C=[0,2]
# WOA: a=2→0, b=1, A=[-a,a], C=[0,2], p=[0,1]
# AOA: alpha=0.1, mu=0.499, r1,r2,r3=[0,1]
# ─────────────────────────────────────────

class HybridOptimizer:
    def __init__(self,
                 population_size=20,
                 gwo_iterations=10,
                 woa_iterations=10,
                 aoa_iterations=10,
                 lower_bounds=None,
                 upper_bounds=None):
        """
        Parameters
        ----------
        population_size : int
            Number of agents per algorithm.
            Paper Table 3: ps = 20
        gwo_iterations : int
            GWO iterations. Paper: 10 in hybrid
        woa_iterations : int
            WOA iterations. Paper: 10 in hybrid
        aoa_iterations : int
            AOA iterations. Paper: 10 in hybrid
        lower_bounds : array-like
            Lower bounds for hyperparameters.
        upper_bounds : array-like
            Upper bounds for hyperparameters.
        """
        self.population_size  = population_size
        self.gwo_iterations   = gwo_iterations
        self.woa_iterations   = woa_iterations
        self.aoa_iterations   = aoa_iterations
        self.lower_bounds     = lower_bounds
        self.upper_bounds     = upper_bounds

        # Results from each stage
        self.gwo_best_pos     = None
        self.gwo_best_fitness = None
        self.woa_best_pos     = None
        self.woa_best_fitness = None
        self.aoa_best_pos     = None
        self.aoa_best_fitness = None

        # Combined convergence curve for Figure 8
        # Concatenates all three stages
        self.convergence_curve = []

        # Decoded final hyperparameters
        self.best_hyperparams = None

        # Timing
        self.start_time = None
        self.end_time   = None

    # ─────────────────────────────────────
    # STAGE 1 — GWO EXPLORATION
    # Paper: "GWO algorithm performs a global
    # search in the large solution space during
    # the exploration phase to identify the best
    # candidate solutions"
    # ─────────────────────────────────────
    def _run_gwo(self, verbose=True):
        """Run GWO exploration phase."""

        if verbose:
            print("\n" + "=" * 50)
            print("STAGE 1 — GWO EXPLORATION PHASE")
            print("=" * 50)
            print(f"Population: {self.population_size} wolves")
            print(f"Iterations: {self.gwo_iterations}")

        gwo = GWO(
            population_size=self.population_size,
            max_iterations=self.gwo_iterations,
            lower_bounds=self.lower_bounds,
            upper_bounds=self.upper_bounds
        )

        best_pos, best_fitness, curve = gwo.optimize(
            verbose=verbose
        )

        self.gwo_best_pos     = best_pos
        self.gwo_best_fitness = best_fitness
        self.convergence_curve.extend(curve)

        if verbose:
            print(f"\nGWO Best Fitness:  {best_fitness:.4f}")
            print(f"GWO Best Accuracy: "
                  f"{(1-best_fitness)*100:.2f}%")
            hp = decode_hyperparameters(best_pos)
            print(f"GWO Best Hyperparameters:")
            for k, v in hp.items():
                print(f"  {k}: {v}")

        return best_pos, best_fitness

    # ─────────────────────────────────────
    # STAGE 2 — WOA EXPLOITATION
    # Paper: "WOA algorithm performs local
    # improvements starting from the best
    # solution found by GWO during the
    # exploitation phase"
    # ─────────────────────────────────────
    def _run_woa(self, gwo_best_pos, verbose=True):
        """Run WOA exploitation phase."""

        if verbose:
            print("\n" + "=" * 50)
            print("STAGE 2 — WOA EXPLOITATION PHASE")
            print("=" * 50)
            print(f"Population: {self.population_size} whales")
            print(f"Iterations: {self.woa_iterations}")
            print(f"Starting from GWO best solution")

        woa = WOA(
            population_size=self.population_size,
            max_iterations=self.woa_iterations,
            lower_bounds=self.lower_bounds,
            upper_bounds=self.upper_bounds
        )

        best_pos, best_fitness, curve = woa.optimize(
            gwo_best_pos=gwo_best_pos,
            verbose=verbose
        )

        self.woa_best_pos     = best_pos
        self.woa_best_fitness = best_fitness
        self.convergence_curve.extend(curve)

        if verbose:
            print(f"\nWOA Best Fitness:  {best_fitness:.4f}")
            print(f"WOA Best Accuracy: "
                  f"{(1-best_fitness)*100:.2f}%")
            hp = decode_hyperparameters(best_pos)
            print(f"WOA Best Hyperparameters:")
            for k, v in hp.items():
                print(f"  {k}: {v}")

        return best_pos, best_fitness

    # ─────────────────────────────────────
    # STAGE 3 — AOA FINE-TUNING
    # Paper: "AOA algorithm performs precise
    # optimization using the WOA output during
    # the fine-tuning phase to produce the
    # final optimal hyperparameter set"
    # ─────────────────────────────────────
    def _run_aoa(self, woa_best_pos, verbose=True):
        """Run AOA fine-tuning phase."""

        if verbose:
            print("\n" + "=" * 50)
            print("STAGE 3 — AOA FINE-TUNING PHASE")
            print("=" * 50)
            print(f"Population: {self.population_size} agents")
            print(f"Iterations: {self.aoa_iterations}")
            print(f"alpha=0.1, mu=0.499")
            print(f"Starting from WOA best solution")

        aoa = AOA(
            population_size=self.population_size,
            max_iterations=self.aoa_iterations,
            lower_bounds=self.lower_bounds,
            upper_bounds=self.upper_bounds,
            alpha=0.1,    # Paper Table 3
            mu=0.499      # Paper Table 3
        )

        best_pos, best_fitness, curve = aoa.optimize(
            woa_best_pos=woa_best_pos,
            verbose=verbose
        )

        self.aoa_best_pos     = best_pos
        self.aoa_best_fitness = best_fitness
        self.convergence_curve.extend(curve)

        if verbose:
            print(f"\nAOA Best Fitness:  {best_fitness:.4f}")
            print(f"AOA Best Accuracy: "
                  f"{(1-best_fitness)*100:.2f}%")
            hp = decode_hyperparameters(best_pos)
            print(f"AOA Best Hyperparameters:")
            for k, v in hp.items():
                print(f"  {k}: {v}")

        return best_pos, best_fitness

    # ─────────────────────────────────────
    # MAIN HYBRID OPTIMIZATION
    # Chains GWO → WOA → AOA sequentially
    # Figure 6 flowchart from paper
    # ─────────────────────────────────────
    def optimize(self,
                 X_train, y_train,
                 X_val, y_val,
                 verbose=True):
        """
        Run full GWO-WOA-AOA hybrid optimization.

        Parameters
        ----------
        X_train, y_train : training data
        X_val, y_val     : validation data
        verbose          : print progress

        Returns
        -------
        best_hyperparams : dict
            Final decoded hyperparameters.
        best_fitness : float
            Final best fitness value.
        convergence_curve : list
            Combined curve across all 3 stages.
        """
        self.start_time = time.time()

        if verbose:
            print("\n" + "=" * 50)
            print("GWO-WOA-AOA HYBRID OPTIMIZATION")
            print("Paper Section 3.2.4")
            print("=" * 50)
            total_iter = (self.gwo_iterations +
              self.woa_iterations +
              self.aoa_iterations)
            print(f"Total iterations: {total_iter}")
            print(f"Population size: {self.population_size}")

        # Register data with fitness function
        # Test data is never used here
        set_data(X_train, y_train, X_val, y_val)
        reset_history()

        # ── Stage 1: GWO Exploration ──
        gwo_pos, gwo_fitness = self._run_gwo(verbose)

        # ── Stage 2: WOA Exploitation ──
        # Starts from GWO best solution
        woa_pos, woa_fitness = self._run_woa(
            gwo_pos, verbose
        )

        # ── Stage 3: AOA Fine-Tuning ──
        # Starts from WOA best solution
        aoa_pos, aoa_fitness = self._run_aoa(
            woa_pos, verbose
        )

        # ── Decode final hyperparameters ──
        self.best_hyperparams = decode_hyperparameters(
            aoa_pos
        )

        self.end_time = time.time()
        elapsed = (self.end_time - self.start_time) / 60

        if verbose:
            print("\n" + "=" * 50)
            print("HYBRID OPTIMIZATION COMPLETE")
            print("=" * 50)
            print(f"Computation time: {elapsed:.1f} minutes")
            print(f"\nStage comparison:")
            print(f"  GWO accuracy: "
                  f"{(1-gwo_fitness)*100:.2f}%")
            print(f"  WOA accuracy: "
                  f"{(1-woa_fitness)*100:.2f}%")
            print(f"  AOA accuracy: "
                  f"{(1-aoa_fitness)*100:.2f}%")
            print(f"\nFinal hyperparameters (Table 6):")
            for k, v in self.best_hyperparams.items():
                print(f"  {k:15s}: {v}")

        return (self.best_hyperparams,
                aoa_fitness,
                self.convergence_curve)

    # ─────────────────────────────────────
    # TRAIN FINAL MODEL ON BEST HYPERPARAMS
    # Uses full train+val data for final model
    # ─────────────────────────────────────
    def train_final_model(self,
                          X_train, y_train,
                          X_val, y_val,
                          verbose=True):
        """
        Train final CNN with optimal hyperparameters.
        Uses training + validation data combined
        for the final model as per paper.
        """
        if self.best_hyperparams is None:
            raise RuntimeError(
                "Run optimize() first."
            )

        if verbose:
            print("\nTraining final CNN model...")
            print(f"Hyperparameters: {self.best_hyperparams}")

        model = build_cnn(self.best_hyperparams)

        train_cnn(
            model=model,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            batch_size=self.best_hyperparams['batch_size'],
            max_epoch=self.best_hyperparams['max_epoch'],
            save_path='models/final_model.weights.h5'
        )

        if verbose:
            print("Final model training complete.")

        return model


# ─────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────
def test_hybrid():
    """
    Test hybrid pipeline with small populations.
    2 iterations, 3 agents per algorithm.
    Full run would use 20 agents, 10 iterations.
    """
    import numpy as np

    print("Testing Hybrid GWO-WOA-AOA...")
    print("(2 iterations, 3 agents per algorithm)")
    print("-" * 50)

    # Load preprocessed data
    X_train = np.load('data/processed/X_train.npy')
    y_train = np.load('data/processed/y_train.npy')
    X_val   = np.load('data/processed/X_val.npy')
    y_val   = np.load('data/processed/y_val.npy')

    hybrid = HybridOptimizer(
        population_size=3,    # full: 20
        gwo_iterations=2,     # full: 10
        woa_iterations=2,     # full: 10
        aoa_iterations=2,     # full: 10
        lower_bounds=LOWER_BOUNDS,
        upper_bounds=UPPER_BOUNDS
    )

    best_hp, best_fitness, curve = hybrid.optimize(
        X_train, y_train,
        X_val, y_val,
        verbose=True
    )

    print(f"\nConvergence curve ({len(curve)} points):")
    print(f"  {[round(x, 4) for x in curve]}")
    print(f"\nFinal best fitness: {best_fitness:.4f}")
    print(f"Final best accuracy: "
          f"{(1-best_fitness)*100:.2f}%")
    print("\nHybrid test PASSED")


if __name__ == "__main__":
    test_hybrid()