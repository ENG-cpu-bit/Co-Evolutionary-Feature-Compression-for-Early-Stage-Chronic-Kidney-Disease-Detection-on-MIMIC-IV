"""
fungal_growth.py
================
Fungal Growth Optimizer (FGO) — 2025
Used for perturbation injection in SGPO v2.

Inspired by fungal mycelium network growth:
  - Spore dispersal: random injection of new solutions (exploration)
  - Hyphal growth: local perturbation of existing solutions (exploitation)
  - Nutrient response: solutions in rich areas grow more

Designed for stochastic/noisy environments. Handles missing values
and prevents premature convergence.

Reference: Fungal Growth Optimizer (2025)
"""

import numpy as np


class FungalGrowthOptimizer:
    """
    Fungal Growth Optimizer for perturbation injection.

    Applied to BOTH SFOA masks and DOA particles to:
      1. Prevent premature convergence
      2. Inject diversity when population stagnates
      3. Handle noise in fitness evaluations
    """

    def __init__(self, stagnation_threshold=3, spore_rate=0.2,
                 hyphal_rate=0.15, random_state=42):
        """
        Parameters
        ----------
        stagnation_threshold : int — generations without improvement before spore dispersal
        spore_rate : float — fraction of population to replace with spores
        hyphal_rate : float — mutation rate for hyphal growth
        random_state : int
        """
        self.stagnation_threshold = stagnation_threshold
        self.spore_rate = spore_rate
        self.hyphal_rate = hyphal_rate
        self.rng = np.random.RandomState(random_state)

        self.best_fitness_history = []
        self.stagnation_count = 0

    def check_stagnation(self, current_best_fitness):
        """
        Track fitness history and detect stagnation.

        Returns
        -------
        is_stagnating : bool
        """
        if len(self.best_fitness_history) > 0:
            prev_best = max(self.best_fitness_history)
            improvement = current_best_fitness - prev_best
            if improvement < 1e-5:
                self.stagnation_count += 1
            else:
                self.stagnation_count = 0
        self.best_fitness_history.append(current_best_fitness)

        return self.stagnation_count >= self.stagnation_threshold

    def spore_dispersal_binary(self, population, n_features):
        """
        Spore dispersal for binary masks (SFOA).
        Replaces worst individuals with random new masks.

        Parameters
        ----------
        population : list of np.array — binary masks
        n_features : int

        Returns
        -------
        population : list — updated population
        n_replaced : int — number of individuals replaced
        """
        n_replace = max(1, int(len(population) * self.spore_rate))

        for i in range(n_replace):
            # Replace from the end (assumed sorted worst-last or random)
            idx = len(population) - 1 - i
            if idx < 1:  # don't replace the best (index 0)
                break

            new_mask = self.rng.randint(0, 2, size=n_features)
            while new_mask.sum() < 2:
                flip = self.rng.randint(0, n_features)
                new_mask[flip] = 1
            population[idx] = new_mask

        return population, n_replace

    def spore_dispersal_continuous(self, positions):
        """
        Spore dispersal for continuous vectors (DOA).
        Replaces worst particles with random positions.

        Parameters
        ----------
        positions : np.array shape (pop_size, n_dims)

        Returns
        -------
        positions : np.array — updated
        n_replaced : int
        """
        pop_size, n_dims = positions.shape
        n_replace = max(1, int(pop_size * self.spore_rate))

        for i in range(n_replace):
            idx = pop_size - 1 - i
            if idx < 1:
                break
            positions[idx] = self.rng.uniform(0, 1, size=n_dims)

        return positions, n_replace

    def hyphal_growth_binary(self, population):
        """
        Hyphal growth: small perturbations on binary masks.
        Flips random bits with probability hyphal_rate.

        Parameters
        ----------
        population : list of np.array

        Returns
        -------
        population : list — perturbed population
        """
        for i in range(1, len(population)):  # skip best (index 0)
            mask = population[i]
            for j in range(len(mask)):
                if self.rng.random() < self.hyphal_rate:
                    mask[j] = 1 - mask[j]
            # Ensure minimum features
            while mask.sum() < 2:
                mask[self.rng.randint(0, len(mask))] = 1
            population[i] = mask

        return population

    def hyphal_growth_continuous(self, positions):
        """
        Hyphal growth: small perturbations on continuous vectors.

        Parameters
        ----------
        positions : np.array shape (pop_size, n_dims)

        Returns
        -------
        positions : np.array — perturbed
        """
        pop_size, n_dims = positions.shape
        for i in range(1, pop_size):  # skip best
            noise = self.rng.normal(0, 0.05, size=n_dims)
            positions[i] += noise
            positions[i] = np.clip(positions[i], 0, 1)
        return positions

    def apply(self, sfoa, doa, current_best_fitness):
        """
        Apply fungal growth perturbation to both SFOA and DOA populations.

        Parameters
        ----------
        sfoa : SFOA instance
        doa : DOA instance
        current_best_fitness : float

        Returns
        -------
        action : str — what action was taken
        """
        is_stagnating = self.check_stagnation(current_best_fitness)

        if is_stagnating:
            # SPORE DISPERSAL — inject completely new solutions
            sfoa.population, n_sfoa = self.spore_dispersal_binary(
                sfoa.population, sfoa.n_features
            )
            doa.positions, n_doa = self.spore_dispersal_continuous(doa.positions)
            self.stagnation_count = 0  # reset
            return f"spore_dispersal (replaced {n_sfoa} masks, {n_doa} particles)"
        else:
            # HYPHAL GROWTH — small perturbations
            sfoa.population = self.hyphal_growth_binary(sfoa.population)
            doa.positions = self.hyphal_growth_continuous(doa.positions)
            return "hyphal_growth (small perturbations)"
