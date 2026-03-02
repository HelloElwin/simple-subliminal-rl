"""Grid environment with random layouts for subliminal RL experiments."""

from collections import deque
from enum import IntEnum

import numpy as np


class CellType(IntEnum):
    EMPTY = 0
    WALL = 1
    AGENT = 2
    RED = 3
    BLUE = 4
    GREEN = 5
    ALPHA = 6
    BETA = 7
    GAMMA = 8
    FILLER_A0 = 9
    FILLER_A1 = 10
    FILLER_A2 = 11
    FILLER_A3 = 12
    FILLER_B0 = 13
    FILLER_B1 = 14
    FILLER_B2 = 15
    FILLER_B3 = 16


NUM_CELL_TYPES = len(CellType)

FILLERS_A = [CellType.FILLER_A0, CellType.FILLER_A1, CellType.FILLER_A2, CellType.FILLER_A3]
FILLERS_B = [CellType.FILLER_B0, CellType.FILLER_B1, CellType.FILLER_B2, CellType.FILLER_B3]
NUM_ACTIONS = 4
# Actions: 0=up, 1=down, 2=left, 3=right (row, col deltas)
ACTION_DR = [-1, 1, 0, 0]
ACTION_DC = [0, 0, -1, 1]
ACTION_DR_ARR = np.array(ACTION_DR, dtype=np.int64)
ACTION_DC_ARR = np.array(ACTION_DC, dtype=np.int64)


class GridEnv:
    """Grid environment with random wall/goal placement each episode.

    Observation: integer grid of shape (grid_size, grid_size) with CellType values.
    The model's embedding layer maps these to learned vectors.
    Actions: 0=up, 1=down, 2=left, 3=right.
    """

    def __init__(
        self,
        grid_size: int = 7,
        wall_density: float = 0.1,
        goal_types: list[CellType] | None = None,
        goal_rewards: dict[CellType, float] | None = None,
        max_steps: int = 100,
        rng: np.random.Generator | None = None,
        filler_types: list[CellType] | None = None,
        filler_density: float = 0.5,
    ):
        self.grid_size = grid_size
        self.wall_density = wall_density
        self.goal_types = goal_types or [CellType.RED, CellType.BLUE, CellType.GREEN]
        self.goal_rewards = goal_rewards or {}
        self.max_steps = max_steps
        self.rng = rng or np.random.default_rng()
        self.filler_types = filler_types or []
        self.filler_density = filler_density

        self.grid = np.zeros((grid_size, grid_size), dtype=np.int64)
        self.agent_row = 0
        self.agent_col = 0
        self.steps = 0

    def reset(self) -> tuple[np.ndarray, dict]:
        """Generate a new random layout and return (obs, info)."""
        for _ in range(50):
            if self._generate_layout():
                self.steps = 0
                info = {
                    "agent_pos": (self.agent_row, self.agent_col),
                    "goal_positions": dict(self._goal_positions),
                }
                return self._obs(), info
        # Fallback: generate with no walls
        self._generate_layout(force_no_walls=True)
        self.steps = 0
        info = {
            "agent_pos": (self.agent_row, self.agent_col),
            "goal_positions": dict(self._goal_positions),
        }
        return self._obs(), info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Take one step. Returns (obs, reward, terminated, truncated, info)."""
        dr, dc = ACTION_DR[action], ACTION_DC[action]
        new_r = self.agent_row + dr
        new_c = self.agent_col + dc

        if (
            0 <= new_r < self.grid_size
            and 0 <= new_c < self.grid_size
            and self.grid[new_r, new_c] != CellType.WALL
        ):
            self.agent_row, self.agent_col = new_r, new_c

        self.steps += 1

        info = {}
        terminated = False
        truncated = False
        reward = 0.0

        cell = int(self.grid[self.agent_row, self.agent_col])
        if cell in self._goal_set:
            info["goal_reached"] = CellType(cell).name
            reward = self.goal_rewards.get(CellType(cell), 0.0)
            terminated = True

        if not terminated and self.steps >= self.max_steps:
            truncated = True

        return self._obs(), reward, terminated, truncated, info

    def _generate_layout(self, force_no_walls: bool = False) -> bool:
        """Generate random wall + goal placement. Returns True if valid."""
        self.grid[:] = CellType.EMPTY

        # Place walls
        if not force_no_walls:
            n_cells = self.grid_size * self.grid_size
            n_walls = int(n_cells * self.wall_density)
            all_positions = [
                (r, c) for r in range(self.grid_size) for c in range(self.grid_size)
            ]
            self.rng.shuffle(all_positions)
            for i in range(n_walls):
                r, c = all_positions[i]
                self.grid[r, c] = CellType.WALL

        # Collect empty cells
        empty = [(r, c) for r in range(self.grid_size) for c in range(self.grid_size)
                 if self.grid[r, c] == CellType.EMPTY]
        self.rng.shuffle(empty)

        # Need: 1 agent + len(goal_types) goals
        needed = 1 + len(self.goal_types)
        if len(empty) < needed:
            return False

        # Place agent
        self.agent_row, self.agent_col = empty[0]

        # Place goals
        self._goal_positions = {}  # (row, col) -> CellType
        self._goal_set = set()
        for i, gtype in enumerate(self.goal_types):
            r, c = empty[1 + i]
            self.grid[r, c] = int(gtype)
            self._goal_positions[(r, c)] = gtype
            self._goal_set.add(int(gtype))

        # Place filler cells on remaining empty cells
        if self.filler_types:
            remaining_empty = [
                (r, c) for r in range(self.grid_size) for c in range(self.grid_size)
                if self.grid[r, c] == CellType.EMPTY and (r, c) != (self.agent_row, self.agent_col)
            ]
            n_fill = int(len(remaining_empty) * self.filler_density)
            if n_fill > 0:
                self.rng.shuffle(remaining_empty)
                filler_vals = self.rng.choice([int(f) for f in self.filler_types], size=n_fill)
                for i in range(n_fill):
                    r, c = remaining_empty[i]
                    self.grid[r, c] = filler_vals[i]

        # BFS reachability check
        goal_coords = set(self._goal_positions.keys())
        return self._bfs_reachable((self.agent_row, self.agent_col), goal_coords)

    def _bfs_reachable(self, start: tuple[int, int], targets: set[tuple[int, int]]) -> bool:
        """BFS from start. Returns True if all targets are reachable."""
        if not targets:
            return True
        visited = set()
        visited.add(start)
        queue = deque([start])
        remaining = set(targets)

        while queue and remaining:
            r, c = queue.popleft()
            if (r, c) in remaining:
                remaining.discard((r, c))
                if not remaining:
                    return True
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if (
                    0 <= nr < self.grid_size
                    and 0 <= nc < self.grid_size
                    and (nr, nc) not in visited
                    and self.grid[nr, nc] != CellType.WALL
                ):
                    visited.add((nr, nc))
                    queue.append((nr, nc))

        return len(remaining) == 0

    def _obs(self) -> np.ndarray:
        """Build integer observation of shape (grid_size, grid_size).

        Each cell contains its CellType integer value.
        Agent position is overlaid on the grid.
        """
        obs = self.grid.copy()
        obs[self.agent_row, self.agent_col] = CellType.AGENT
        return obs


# ============================================================
# Batched (numpy-vectorized) Environment
# ============================================================


def _generate_grid(
    grid_size: int,
    wall_density: float,
    goal_types: list[int],
    filler_types: list[int],
    filler_density: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, int, int, set[tuple[int, int]]] | None:
    """Generate a single random grid layout.

    Returns (grid, agent_row, agent_col, goal_coords) or None if invalid.
    """
    grid = np.zeros((grid_size, grid_size), dtype=np.int64)

    # Place walls
    n_cells = grid_size * grid_size
    n_walls = int(n_cells * wall_density)
    all_positions = np.arange(n_cells)
    rng.shuffle(all_positions)
    wall_flat = all_positions[:n_walls]
    grid.ravel()[wall_flat] = CellType.WALL

    # Collect empty cells
    empty_flat = np.where(grid.ravel() == CellType.EMPTY)[0]
    rng.shuffle(empty_flat)

    needed = 1 + len(goal_types)
    if len(empty_flat) < needed:
        return None

    # Place agent
    agent_flat = empty_flat[0]
    agent_row, agent_col = divmod(int(agent_flat), grid_size)

    # Place goals
    goal_coords = set()
    for i, gtype in enumerate(goal_types):
        gf = empty_flat[1 + i]
        grid.ravel()[gf] = gtype
        gr, gc = divmod(int(gf), grid_size)
        goal_coords.add((gr, gc))

    # Place fillers
    if len(filler_types) > 0:
        remaining = empty_flat[needed:]
        # Exclude agent position
        remaining = remaining[remaining != agent_flat]
        n_fill = int(len(remaining) * filler_density)
        if n_fill > 0:
            fill_positions = remaining[:n_fill]
            filler_vals = rng.choice(filler_types, size=n_fill)
            grid.ravel()[fill_positions] = filler_vals

    # BFS reachability check
    if not _bfs_reachable(grid, grid_size, agent_row, agent_col, goal_coords):
        return None

    return grid, agent_row, agent_col, goal_coords


def _bfs_reachable(
    grid: np.ndarray, grid_size: int, start_r: int, start_c: int, targets: set[tuple[int, int]]
) -> bool:
    """BFS from start. Returns True if all targets are reachable."""
    if not targets:
        return True
    visited = set()
    visited.add((start_r, start_c))
    queue = deque([(start_r, start_c)])
    remaining = set(targets)

    while queue and remaining:
        r, c = queue.popleft()
        if (r, c) in remaining:
            remaining.discard((r, c))
            if not remaining:
                return True
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if (
                0 <= nr < grid_size
                and 0 <= nc < grid_size
                and (nr, nc) not in visited
                and grid[nr, nc] != CellType.WALL
            ):
                visited.add((nr, nc))
                queue.append((nr, nc))

    return len(remaining) == 0


class BatchGridEnv:
    """Numpy-vectorized batched grid environment.

    All N environments are stored as batched arrays and stepped simultaneously.
    Auto-resets envs on termination.
    """

    def __init__(
        self,
        num_envs: int,
        grid_size: int = 7,
        wall_density: float = 0.1,
        goal_types: list[CellType] | None = None,
        goal_rewards: dict[CellType, float] | None = None,
        max_steps: int = 100,
        base_seed: int = 0,
        filler_types: list[CellType] | None = None,
        filler_density: float = 0.5,
    ):
        self.num_envs = num_envs
        self.grid_size = grid_size
        self.wall_density = wall_density
        self.goal_types = goal_types or [CellType.RED, CellType.BLUE, CellType.GREEN]
        self.goal_rewards = goal_rewards or {}
        self.max_steps = max_steps
        self.filler_types = filler_types or []
        self.filler_density = filler_density

        # Per-env RNGs (needed for grid generation on reset)
        self.rngs = [np.random.default_rng(base_seed + i) for i in range(num_envs)]

        # Precompute for vectorized goal/reward checks
        self._goal_type_vals = np.array([int(g) for g in self.goal_types], dtype=np.int64)
        self._filler_type_vals = np.array([int(f) for f in self.filler_types], dtype=np.int64)
        # Reward lookup: indexed by cell type value
        self._reward_lookup = np.zeros(NUM_CELL_TYPES, dtype=np.float32)
        for gtype, rew in self.goal_rewards.items():
            self._reward_lookup[int(gtype)] = rew

        # Batched state arrays
        self.grids = np.zeros((num_envs, grid_size, grid_size), dtype=np.int64)
        self.agent_rows = np.zeros(num_envs, dtype=np.int64)
        self.agent_cols = np.zeros(num_envs, dtype=np.int64)
        self.step_counts = np.zeros(num_envs, dtype=np.int64)

        # Precompute env index array
        self._env_idx = np.arange(num_envs)

    def reset(self) -> tuple[np.ndarray, list[dict]]:
        """Reset all envs. Returns (obs, infos) with obs shape (N, H, W) int64."""
        infos = []
        for i in range(self.num_envs):
            info = self._reset_single(i)
            infos.append(info)
        return self._obs(), infos

    def _reset_single(self, i: int) -> dict:
        """Generate a new grid for env i. Returns info dict."""
        rng = self.rngs[i]
        goal_type_ints = [int(g) for g in self.goal_types]

        for _ in range(50):
            result = _generate_grid(
                self.grid_size, self.wall_density, goal_type_ints,
                self._filler_type_vals, self.filler_density, rng,
            )
            if result is not None:
                grid, agent_row, agent_col, goal_coords = result
                self.grids[i] = grid
                self.agent_rows[i] = agent_row
                self.agent_cols[i] = agent_col
                self.step_counts[i] = 0
                return {}

        # Fallback: no walls
        grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int64)
        empty_flat = np.arange(self.grid_size * self.grid_size)
        rng.shuffle(empty_flat)
        agent_flat = empty_flat[0]
        agent_row, agent_col = divmod(int(agent_flat), self.grid_size)
        for j, gtype in enumerate(goal_type_ints):
            gf = empty_flat[1 + j]
            grid.ravel()[gf] = gtype
        if self._filler_type_vals.size > 0:
            needed = 1 + len(goal_type_ints)
            remaining = empty_flat[needed:]
            remaining = remaining[remaining != agent_flat]
            n_fill = int(len(remaining) * self.filler_density)
            if n_fill > 0:
                grid.ravel()[remaining[:n_fill]] = rng.choice(self._filler_type_vals, size=n_fill)
        self.grids[i] = grid
        self.agent_rows[i] = agent_row
        self.agent_cols[i] = agent_col
        self.step_counts[i] = 0
        return {}

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict]]:
        """Step all envs with vectorized numpy ops. Auto-resets on done."""
        H = self.grid_size
        N = self.num_envs
        idx = self._env_idx

        # 1. Compute new positions
        new_r = self.agent_rows + ACTION_DR_ARR[actions]
        new_c = self.agent_cols + ACTION_DC_ARR[actions]

        # 2. Bounds check
        in_bounds = (new_r >= 0) & (new_r < H) & (new_c >= 0) & (new_c < H)

        # 3. Wall check (clip for safe indexing, masked by in_bounds)
        safe_r = np.clip(new_r, 0, H - 1)
        safe_c = np.clip(new_c, 0, H - 1)
        not_wall = self.grids[idx, safe_r, safe_c] != CellType.WALL

        # 4. Update positions where valid
        valid = in_bounds & not_wall
        self.agent_rows = np.where(valid, new_r, self.agent_rows)
        self.agent_cols = np.where(valid, new_c, self.agent_cols)
        self.step_counts += 1

        # 5. Check goals
        cells = self.grids[idx, self.agent_rows, self.agent_cols]
        terminated = np.isin(cells, self._goal_type_vals)
        truncated = (~terminated) & (self.step_counts >= self.max_steps)
        dones = terminated | truncated

        # 6. Rewards via lookup table
        rewards = self._reward_lookup[cells]

        # 7. Build infos for terminated envs
        infos: list[dict] = [{} for _ in range(N)]
        term_indices = np.where(terminated)[0]
        for i in term_indices:
            infos[i] = {"goal_reached": CellType(int(cells[i])).name}

        # 8. Auto-reset done envs
        done_indices = np.where(dones)[0]
        if len(done_indices) > 0:
            obs = self._obs()
            for i in done_indices:
                infos[int(i)]["terminal_observation"] = obs[i]
                self._reset_single(int(i))

        return self._obs(), rewards, terminated, truncated, infos

    def _obs(self) -> np.ndarray:
        """Build batched observation (N, H, W) int64 with agent positions overlaid."""
        obs = self.grids.copy()
        obs[self._env_idx, self.agent_rows, self.agent_cols] = CellType.AGENT
        return obs


# ============================================================
# Factory functions
# ============================================================


def make_env_a(
    grid_size: int = 7,
    wall_density: float = 0.1,
    max_steps: int = 100,
    goal_rewards: dict[CellType, float] | None = None,
    rng: np.random.Generator | None = None,
    filler_density: float = 0.5,
) -> GridEnv:
    """Create Env A: random walls + RED, BLUE, GREEN goals."""
    if goal_rewards is None:
        goal_rewards = {CellType.RED: 1.0, CellType.BLUE: 0.0, CellType.GREEN: 0.0}
    return GridEnv(
        grid_size=grid_size,
        wall_density=wall_density,
        goal_types=[CellType.RED, CellType.BLUE, CellType.GREEN],
        goal_rewards=goal_rewards,
        max_steps=max_steps,
        rng=rng,
        filler_types=FILLERS_A,
        filler_density=filler_density,
    )


def make_env_b(
    grid_size: int = 7,
    wall_density: float = 0.1,
    max_steps: int = 100,
    goal_rewards: dict[CellType, float] | None = None,
    rng: np.random.Generator | None = None,
    filler_density: float = 0.5,
) -> GridEnv:
    """Create Env B: random walls + ALPHA, BETA, GAMMA goals. No colored goals."""
    if goal_rewards is None:
        goal_rewards = {CellType.ALPHA: 0.0, CellType.BETA: 0.0, CellType.GAMMA: 0.0}
    return GridEnv(
        grid_size=grid_size,
        wall_density=wall_density,
        goal_types=[CellType.ALPHA, CellType.BETA, CellType.GAMMA],
        goal_rewards=goal_rewards,
        max_steps=max_steps,
        rng=rng,
        filler_types=FILLERS_B,
        filler_density=filler_density,
    )


def make_vec_env_a(
    num_envs: int,
    grid_size: int = 7,
    wall_density: float = 0.1,
    max_steps: int = 100,
    goal_rewards: dict[CellType, float] | None = None,
    base_seed: int = 0,
    filler_density: float = 0.5,
) -> BatchGridEnv:
    """Create batched Env A with num_envs instances."""
    if goal_rewards is None:
        goal_rewards = {CellType.RED: 1.0, CellType.BLUE: 0.0, CellType.GREEN: 0.0}
    return BatchGridEnv(
        num_envs=num_envs, grid_size=grid_size, wall_density=wall_density,
        goal_types=[CellType.RED, CellType.BLUE, CellType.GREEN],
        goal_rewards=goal_rewards, max_steps=max_steps, base_seed=base_seed,
        filler_types=FILLERS_A, filler_density=filler_density,
    )


def make_vec_env_b(
    num_envs: int,
    grid_size: int = 7,
    wall_density: float = 0.1,
    max_steps: int = 100,
    goal_rewards: dict[CellType, float] | None = None,
    base_seed: int = 0,
    filler_density: float = 0.5,
) -> BatchGridEnv:
    """Create batched Env B with num_envs instances."""
    if goal_rewards is None:
        goal_rewards = {CellType.ALPHA: 0.0, CellType.BETA: 0.0, CellType.GAMMA: 0.0}
    return BatchGridEnv(
        num_envs=num_envs, grid_size=grid_size, wall_density=wall_density,
        goal_types=[CellType.ALPHA, CellType.BETA, CellType.GAMMA],
        goal_rewards=goal_rewards, max_steps=max_steps, base_seed=base_seed,
        filler_types=FILLERS_B, filler_density=filler_density,
    )
