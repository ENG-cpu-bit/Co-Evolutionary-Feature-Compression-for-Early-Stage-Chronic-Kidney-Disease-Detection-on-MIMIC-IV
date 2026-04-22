"""
sfoa.py
=======
Starfish Optimization Algorithm (SFOA) — 2025
Used for binary feature selection in SGPO v2.

Inspired by starfish regeneration and multi-arm search behavior.
Each starfish = a binary feature mask.
Arms explore different directions in the binary search space.

Reference: Starfish Optimization Algorithm (2025)
"""

import numpy as np


class SFOA:
    """
    Starfish Optimization Algorithm for binary feature selection.

    Each individual is a binary mask of length n_features.
    The population evolves via:
      1. Arm-based exploration (multi-directional search)
      2. Regeneration (crossover from best solutions)
      3. Tournament selection
    """

    def __init__(self, n_features, pop_size=10, n_arms=5, random_state=42):
        """
        Parameters
        ----------
        n_features : int — total number of features
        pop_size : int — population size
        n_arms : int — number of starfish arms (search directions)
        random_state : int
        """
        self.n_features = n_features
        self.pop_size = pop_size
        self.n_arms = n_arms
        self.rng = np.random.RandomState(random_state)

        # Initialize population: random binary masks
        # Ensure each mask has at least 2 features selected
        self.population = []
        for _ in range(pop_size):
            mask = self._random_mask()
            self.population.append(mask)

        self.fitness = np.full(pop_size, -np.inf)
        self.best_mask = None
        self.best_fitness = -np.inf

    def _random_mask(self, min_features=2):
        """Generate a random binary mask with at least min_features selected."""
        mask = self.rng.randint(0, 2, size=self.n_features)
        while mask.sum() < min_features:
            idx = self.rng.randint(0, self.n_features)
            mask[idx] = 1
        return mask

    def _ensure_min_features(self, mask, min_features=2):
        """Ensure mask has at least min_features selected."""
        while mask.sum() < min_features:
            idx = self.rng.randint(0, self.n_features)
            mask[idx] = 1
        return mask

    def update_fitness(self, idx, fitness_value):
        """Update fitness for individual at index idx."""
        self.fitness[idx] = fitness_value
        if fitness_value > self.best_fitness:
            self.best_fitness = fitness_value
            self.best_mask = self.population[idx].copy()

    def _tournament_select(self, k=3):
        """Select an individual via tournament selection."""
        candidates = self.rng.choice(self.pop_size, size=min(k, self.pop_size), replace=False)
        best_idx = candidates[np.argmax(self.fitness[candidates])]
        return self.population[best_idx].copy()

    def _arm_exploration(self, mask):
        """
        Multi-arm search: each arm flips a different subset of bits.
        Returns the best arm result.
        """
        arm_results = []
        bits_per_arm = max(1, self.n_features // self.n_arms)

        for arm in range(self.n_arms):
            new_mask = mask.copy()
            # Each arm explores a different region of the feature space
            start = (arm * bits_per_arm) % self.n_features
            n_flip = self.rng.randint(1, max(2, bits_per_arm // 2))

            for _ in range(n_flip):
                flip_idx = (start + self.rng.randint(0, bits_per_arm)) % self.n_features
                new_mask[flip_idx] = 1 - new_mask[flip_idx]

            new_mask = self._ensure_min_features(new_mask)
            arm_results.append(new_mask)

        return arm_results

    def _regeneration_crossover(self, parent1, parent2):
        """
        Starfish regeneration: create offspring by combining two parents.
        Mimics how a starfish can regenerate from a single arm.
        """
        child = parent1.copy()

        # Pick a "regeneration point" — one arm's worth of features
        arm_size = max(1, self.n_features // self.n_arms)
        start = self.rng.randint(0, self.n_features)

        for i in range(arm_size):
            idx = (start + i) % self.n_features
            child[idx] = parent2[idx]

        return self._ensure_min_features(child)

    def evolve(self):
        """
        Evolve the population for one generation.
        Uses arm exploration + regeneration crossover + elitism.
        """
        new_population = []

        # Elitism: keep the best individual
        if self.best_mask is not None:
            new_population.append(self.best_mask.copy())

        while len(new_population) < self.pop_size:
            # Select parent via tournament
            parent = self._tournament_select()

            if self.rng.random() < 0.7:
                # Arm exploration (70% of the time)
                arm_masks = self._arm_exploration(parent)
                # Pick a random arm result
                chosen = arm_masks[self.rng.randint(0, len(arm_masks))]
                new_population.append(chosen)
            else:
                # Regeneration crossover (30% of the time)
                parent2 = self._tournament_select()
                child = self._regeneration_crossover(parent, parent2)
                new_population.append(child)

        self.population = new_population[:self.pop_size]
        self.fitness = np.full(self.pop_size, -np.inf)

    def get_population(self):
        """Return current population of binary masks."""
        return self.population
