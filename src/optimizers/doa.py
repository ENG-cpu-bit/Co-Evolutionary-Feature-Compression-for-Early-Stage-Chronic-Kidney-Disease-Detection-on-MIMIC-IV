"""
doa.py
======
Dream Optimization Algorithm (DOA) — 2025
Used for continuous hyperparameter tuning in SGPO v2.

Two-phase search inspired by dream/wake cycles:
  - Dream phase (exploration): large random steps, wide search
  - Wake phase (exploitation): small refinements near best solutions

Reference: Dream Optimization Algorithm (2025)
"""

import numpy as np


# Hyperparameter search space definition
HP_SPACE = {
    "n_estimators":     {"low": 50,   "high": 300,  "dtype": "int"},
    "max_depth":        {"low": 3,    "high": 30,   "dtype": "int"},
    "min_samples_split":{"low": 2,    "high": 20,   "dtype": "int"},
    "min_samples_leaf": {"low": 1,    "high": 10,   "dtype": "int"},
}

HP_NAMES = list(HP_SPACE.keys())
N_DIMS = len(HP_NAMES)


def _encode_hp(hp_dict):
    """Convert HP dict to normalized [0,1] vector."""
    vec = np.zeros(N_DIMS)
    for i, name in enumerate(HP_NAMES):
        space = HP_SPACE[name]
        vec[i] = (hp_dict[name] - space["low"]) / max(1, space["high"] - space["low"])
    return vec


def _decode_hp(vec):
    """Convert normalized [0,1] vector to HP dict."""
    hp = {}
    for i, name in enumerate(HP_NAMES):
        space = HP_SPACE[name]
        val = vec[i] * (space["high"] - space["low"]) + space["low"]
        val = np.clip(val, space["low"], space["high"])
        if space["dtype"] == "int":
            val = int(round(val))
        hp[name] = val
    return hp


class DOA:
    """
    Dream Optimization Algorithm for hyperparameter tuning.

    Each particle is a continuous vector in normalized HP space [0,1]^d.
    Uses dream/wake phase cycling:
      - Dream: large velocity, random direction (exploration)
      - Wake: small velocity toward personal/global best (exploitation)
    """

    def __init__(self, pop_size=10, dream_ratio=0.4, random_state=42):
        """
        Parameters
        ----------
        pop_size : int — number of particles
        dream_ratio : float — fraction of generation in dream phase (0-1)
        random_state : int
        """
        self.pop_size = pop_size
        self.dream_ratio = dream_ratio
        self.rng = np.random.RandomState(random_state)

        # Initialize positions randomly in [0,1]^d
        self.positions = self.rng.uniform(0, 1, size=(pop_size, N_DIMS))
        self.velocities = self.rng.uniform(-0.1, 0.1, size=(pop_size, N_DIMS))

        # Personal best
        self.p_best_pos = self.positions.copy()
        self.p_best_fit = np.full(pop_size, -np.inf)

        # Global best
        self.g_best_pos = self.positions[0].copy()
        self.g_best_fit = -np.inf

        self.fitness = np.full(pop_size, -np.inf)

    def get_hp_dicts(self):
        """Return list of HP dicts for all particles."""
        return [_decode_hp(self.positions[i]) for i in range(self.pop_size)]

    def update_fitness(self, idx, fitness_value):
        """Update fitness for particle at index idx."""
        self.fitness[idx] = fitness_value

        # Update personal best
        if fitness_value > self.p_best_fit[idx]:
            self.p_best_fit[idx] = fitness_value
            self.p_best_pos[idx] = self.positions[idx].copy()

        # Update global best
        if fitness_value > self.g_best_fit:
            self.g_best_fit = fitness_value
            self.g_best_pos = self.positions[idx].copy()

    def evolve(self, generation, max_generations):
        """
        Evolve particles for one generation using dream/wake phases.

        Parameters
        ----------
        generation : int — current generation number
        max_generations : int — total generations
        """
        progress = generation / max(1, max_generations - 1)

        for i in range(self.pop_size):
            # Determine phase: dream early, wake later
            # More dreaming at start, more waking at end
            is_dreaming = self.rng.random() < self.dream_ratio * (1 - progress)

            if is_dreaming:
                # DREAM PHASE — exploration
                # Large random perturbation
                dream_intensity = 0.3 * (1 - progress)  # decreases over time
                self.velocities[i] = self.rng.uniform(
                    -dream_intensity, dream_intensity, size=N_DIMS
                )
                # Random walk with dream influence
                self.positions[i] += self.velocities[i]

            else:
                # WAKE PHASE — exploitation
                # PSO-like update toward personal and global best
                r1 = self.rng.random(N_DIMS)
                r2 = self.rng.random(N_DIMS)

                # Inertia decreases over time
                w = 0.9 - 0.5 * progress

                # Cognitive (personal best) and social (global best)
                c1 = 2.0 * (1 - progress)  # decreases: less personal memory
                c2 = 2.0 * progress         # increases: more social learning

                cognitive = c1 * r1 * (self.p_best_pos[i] - self.positions[i])
                social = c2 * r2 * (self.g_best_pos - self.positions[i])

                self.velocities[i] = w * self.velocities[i] + cognitive + social

                # Velocity clamping
                v_max = 0.2
                self.velocities[i] = np.clip(self.velocities[i], -v_max, v_max)

                self.positions[i] += self.velocities[i]

            # Boundary handling: clip to [0,1]
            self.positions[i] = np.clip(self.positions[i], 0, 1)

        self.fitness = np.full(self.pop_size, -np.inf)

    def get_best_hp(self):
        """Return the global best hyperparameter dict."""
        return _decode_hp(self.g_best_pos)

    def get_best_fitness(self):
        return self.g_best_fit
