"""
sgpo_v2.py
==========
SGPO v2 — Simultaneous GA-PSO Optimizer version 2
Co-evolutionary loop that runs SFOA + DOA + Fungal Growth simultaneously.

Critical innovation: feature selection and hyperparameter tuning happen
in the SAME generation, not sequentially. Each (mask_i, particle_j) pair
is evaluated via a shared fitness function.
"""

import time
from datetime import datetime

import numpy as np
import pandas as pd

from ..optimizers.sfoa import SFOA
from ..optimizers.doa import DOA
from ..optimizers.fungal_growth import FungalGrowthOptimizer
from ..evaluation.fitness import evaluate_solution


class SGPOv2:
    """
    SGPO v2 Co-Evolutionary Optimizer.

    Generation G execution flow:
    1. SFOA maintains N binary feature masks
    2. DOA maintains M continuous HP vectors
    3. Evaluate a subset of (mask_i, particle_j) pairs via shared fitness
    4. SFOA fitness[i] = best fitness across DOA particles paired with mask i
    5. DOA fitness[j] = best fitness across SFOA masks paired with particle j
    6. SFOA evolves (arm exploration + regeneration)
    7. DOA evolves (dream/wake velocity updates)
    8. Fungal Growth injects noise if stagnating
    9. Generation G+1 begins
    """

    def __init__(
        self,
        n_features,
        sfoa_pop_size=10,
        doa_pop_size=10,
        n_generations=30,
        n_inner_folds=3,
        use_smote=True,
        use_mice=False,
        sample_strategy="full",
        random_state=42,
        verbose=True,
    ):
        """
        Parameters
        ----------
        n_features : int — total number of features
        sfoa_pop_size : int — SFOA population size
        doa_pop_size : int — DOA population size
        n_generations : int — number of co-evolutionary generations
        n_inner_folds : int — inner CV folds for fitness evaluation
        use_smote : bool — apply SMOTE in fitness evaluation
        use_mice : bool — use MICE imputation in fitness evaluation
        sample_strategy : str — "full" evaluates all NxM pairs, "diagonal" samples
        random_state : int
        verbose : bool
        """
        self.n_features = n_features
        self.n_generations = n_generations
        self.n_inner_folds = n_inner_folds
        self.use_smote = use_smote
        self.use_mice = use_mice
        self.sample_strategy = sample_strategy
        self.random_state = random_state
        self.verbose = verbose

        # Initialize optimizers
        self.sfoa = SFOA(
            n_features=n_features,
            pop_size=sfoa_pop_size,
            random_state=random_state,
        )
        self.doa = DOA(
            pop_size=doa_pop_size,
            random_state=random_state,
        )
        self.fgo = FungalGrowthOptimizer(
            stagnation_threshold=3,
            random_state=random_state,
        )

        # History tracking
        self.history = []
        self.global_best_fitness = -np.inf
        self.global_best_mask = None
        self.global_best_hp = None
        self.global_best_auc = 0.0
        self.global_best_sens = 0.0
        self.global_best_n_features = 0

    def _log(self, msg):
        if self.verbose:
            ts = datetime.now().strftime("%H:%M:%S")
            print(f"[{ts}] {msg}")

    def _get_evaluation_pairs(self):
        """
        Determine which (mask_i, particle_j) pairs to evaluate.

        full: evaluate all N*M pairs (most thorough but expensive)
        diagonal: each mask paired with one random particle + best particle
        """
        sfoa_pop = self.sfoa.get_population()
        doa_hps = self.doa.get_hp_dicts()
        n_sfoa = len(sfoa_pop)
        n_doa = len(doa_hps)

        if self.sample_strategy == "full":
            # All pairs
            pairs = [(i, j) for i in range(n_sfoa) for j in range(n_doa)]
        else:
            # Diagonal: each SFOA mask with one random DOA + global best DOA
            rng = np.random.RandomState(self.random_state)
            pairs = []
            for i in range(n_sfoa):
                j_rand = rng.randint(0, n_doa)
                pairs.append((i, j_rand))
                # Also pair with the current best DOA particle
                if self.doa.g_best_fit > -np.inf:
                    best_j = np.argmax(self.doa.fitness)
                    if best_j != j_rand:
                        pairs.append((i, best_j))

        return pairs

    def run(self, X, y):
        """
        Run the full SGPO v2 co-evolutionary optimization.

        Parameters
        ----------
        X : pd.DataFrame — feature matrix (no label, no subject_id)
        y : pd.Series — binary labels

        Returns
        -------
        results : dict — best solution and history
        """
        self._log("=" * 60)
        self._log("  SGPO v2 — Co-Evolutionary Optimization")
        self._log("=" * 60)
        self._log(f"  Features: {self.n_features}")
        self._log(f"  SFOA pop: {self.sfoa.pop_size}")
        self._log(f"  DOA pop:  {self.doa.pop_size}")
        self._log(f"  Generations: {self.n_generations}")
        self._log(f"  Strategy: {self.sample_strategy}")
        self._log(f"  Inner CV: {self.n_inner_folds}-fold")
        self._log(f"  SMOTE: {self.use_smote}")
        self._log("")

        start_time = time.time()

        for gen in range(self.n_generations):
            gen_start = time.time()

            # Get populations
            sfoa_masks = self.sfoa.get_population()
            doa_hps = self.doa.get_hp_dicts()

            # Get evaluation pairs
            pairs = self._get_evaluation_pairs()

            # Fitness matrix: sfoa_fitness[i] and doa_fitness[j]
            sfoa_best = np.full(len(sfoa_masks), -np.inf)
            doa_best = np.full(len(doa_hps), -np.inf)

            # Track best for this generation
            gen_best_fitness = -np.inf
            gen_best_auc = 0.0
            gen_best_sens = 0.0
            gen_best_nf = 0
            n_evals = 0

            for (i, j) in pairs:
                mask = sfoa_masks[i]
                hp = doa_hps[j]

                fitness, auc, sens, n_feat = evaluate_solution(
                    X, y, mask, hp,
                    n_inner_folds=self.n_inner_folds,
                    use_smote=self.use_smote,
                    use_mice=self.use_mice,
                    random_state=self.random_state + gen,
                )
                n_evals += 1

                # Update SFOA: mask i gets best fitness across all paired particles
                if fitness > sfoa_best[i]:
                    sfoa_best[i] = fitness

                # Update DOA: particle j gets best fitness across all paired masks
                if fitness > doa_best[j]:
                    doa_best[j] = fitness

                # Track generation best
                if fitness > gen_best_fitness:
                    gen_best_fitness = fitness
                    gen_best_auc = auc
                    gen_best_sens = sens
                    gen_best_nf = n_feat

                # Track global best
                if fitness > self.global_best_fitness:
                    self.global_best_fitness = fitness
                    self.global_best_mask = mask.copy()
                    self.global_best_hp = hp.copy()
                    self.global_best_auc = auc
                    self.global_best_sens = sens
                    self.global_best_n_features = n_feat

            # Update SFOA fitness values
            for i in range(len(sfoa_masks)):
                if sfoa_best[i] > -np.inf:
                    self.sfoa.update_fitness(i, sfoa_best[i])

            # Update DOA fitness values
            for j in range(len(doa_hps)):
                if doa_best[j] > -np.inf:
                    self.doa.update_fitness(j, doa_best[j])

            # Apply Fungal Growth perturbation
            fgo_action = self.fgo.apply(
                self.sfoa, self.doa, self.global_best_fitness
            )

            # Evolve both populations
            self.sfoa.evolve()
            self.doa.evolve(gen, self.n_generations)

            gen_time = time.time() - gen_start

            # Record history
            self.history.append({
                "generation": gen,
                "best_fitness": self.global_best_fitness,
                "gen_fitness": gen_best_fitness,
                "gen_auc": gen_best_auc,
                "gen_sensitivity": gen_best_sens,
                "gen_n_features": gen_best_nf,
                "n_evaluations": n_evals,
                "fgo_action": fgo_action,
                "time_seconds": round(gen_time, 1),
            })

            self._log(
                f"Gen {gen:3d} | "
                f"fitness={gen_best_fitness:.5f} | "
                f"AUC={gen_best_auc:.4f} | "
                f"sens={gen_best_sens:.4f} | "
                f"feat={gen_best_nf:2d} | "
                f"evals={n_evals} | "
                f"{gen_time:.1f}s | "
                f"{fgo_action}"
            )

        total_time = time.time() - start_time

        # Get selected feature names
        feature_names = list(X.columns)
        selected_features = [
            feature_names[i]
            for i in range(self.n_features)
            if self.global_best_mask[i] == 1
        ]

        self._log("")
        self._log("=" * 60)
        self._log("  OPTIMIZATION COMPLETE")
        self._log("=" * 60)
        self._log(f"  Best fitness:     {self.global_best_fitness:.5f}")
        self._log(f"  Best AUC:         {self.global_best_auc:.4f}")
        self._log(f"  Best Sensitivity: {self.global_best_sens:.4f}")
        self._log(f"  Features:         {self.global_best_n_features} / {self.n_features}")
        self._log(f"  Selected:         {selected_features}")
        self._log(f"  Best HP:          {self.global_best_hp}")
        self._log(f"  Total time:       {total_time:.0f}s")

        return {
            "best_fitness": self.global_best_fitness,
            "best_auc": self.global_best_auc,
            "best_sensitivity": self.global_best_sens,
            "best_n_features": self.global_best_n_features,
            "best_mask": self.global_best_mask.tolist(),
            "best_hp": self.global_best_hp,
            "selected_features": selected_features,
            "history": self.history,
            "total_time_seconds": round(total_time, 1),
        }
