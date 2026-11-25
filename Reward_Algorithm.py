import gymnasium as gym
import numpy as np
from collections import defaultdict, deque
from Grid_Maze_360_vision import *

class EmpiricalInstrumentalDivergenceWrapper(gym.Wrapper):
    """
    Intrinsic reward wrapper for the control agent.

    Control points are the cells with full directional freedom (1.0); if a map has
    none, its highest-freedom cells are used. Every other cell's control value falls
    off with walking distance (walls respected) to the nearest control point:

    control_value(cell) = control_decay ** steps_to_nearest_control_point

    so from any connected cell, some neighbouring cell always has a higher value
    until a control point is reached. Cells cut off from all control points get 0.

    The reward is paid only the first time a cell is entered in an episode:

    intrinsic_reward = control_value * scale    (new cell)
    intrinsic_reward = 0                        (revisit, turning, blocked move)

    The starting cell counts as already visited.
    The total reward passed to DQN is: env_reward (red ball) + intrinsic_reward.
    """
    def __init__(self, env, scale=0.1, control_decay=0.8):
        super().__init__(env)
        if not 0.0 < control_decay < 1.0:
            raise ValueError("control_decay must be between 0 and 1")
        self.scale = scale
        self.control_decay = control_decay

        self._control_values = {}
        self.control_points = []

        self.total_intrinsic_reward = 0.0
        self.step_count = 0
        self.reward_components = defaultdict(list)

    def _get_base_env(self):
        """Traverse the wrapper stack to reach the base CleanTaskEnv."""
        base_env = self.env
        while hasattr(base_env, 'env'):
            base_env = base_env.env
        return base_env

    def _get_position(self):
        """Return the agent's current position as a tuple."""
        return tuple(self._get_base_env().agent_pos)

    def _open_neighbours(self, position):
        """
        Return the neighbouring cells that can be entered from position, checking
        both exit walls on the current cell and entry walls on the neighbour.
        Uses the same rules as CleanTaskEnv._compute_directional_freedom.
        """
        base_env = self._get_base_env()

        direction_map = {
            (1,  0): ('right', 'left'),
            (-1, 0): ('left',  'right'),
            (0,  1): ('bottom', 'top'),
            (0, -1): ('top',   'bottom'),
        }

        current_cell = base_env.grid.get(*position)
        neighbours = []
        for (dx, dy), (exit_dir, entry_dir) in direction_map.items():
            target = (position[0] + dx, position[1] + dy)

            # Boundary Check
            if not (0 < target[0] < base_env.width - 1 and
                    0 < target[1] < base_env.height - 1):
                continue

            # Exit Wall Check
            if isinstance(current_cell, MultiWallCell) and current_cell.has_wall(exit_dir):
                continue

            # Entry Wall Check
            target_cell = base_env.grid.get(*target)
            if isinstance(target_cell, MultiWallCell):
                if target_cell.has_wall(entry_dir):
                    continue
            elif target_cell is not None and target_cell.type == 'wall':
                continue

            neighbours.append(target)

        return neighbours

    def _compute_directional_freedom(self, position):
        """Fraction of the 4 directions that can be moved in from position."""
        return len(self._open_neighbours(position)) / 4.0

    def _build_control_map(self):
        """
        Find the control points, then breadth-first search outwards from all of
        them at once to get every cell's distance to its nearest control point.
        """
        base_env = self._get_base_env()
        neighbours = {
            (x, y): self._open_neighbours((x, y))
            for x in range(1, base_env.width - 1)
            for y in range(1, base_env.height - 1)
        }
        max_open = max(len(n) for n in neighbours.values())
        self.control_points = [
            c for c, n in neighbours.items() if max_open > 0 and len(n) == max_open
        ]

        distance = {c: 0 for c in self.control_points}
        queue = deque(self.control_points)
        while queue:
            cell = queue.popleft()
            for nxt in neighbours[cell]:
                if nxt not in distance:
                    distance[nxt] = distance[cell] + 1
                    queue.append(nxt)

        self._control_values = {
            c: self.control_decay ** distance[c] if c in distance else 0.0
            for c in neighbours
        }

    def step(self, action):
        """
        Execute one step and return total_reward = env_reward + intrinsic_reward.
        Intrinsic reward is paid only for entering a cell that is new this episode.
        """
        if isinstance(action, np.ndarray):
            action = action.item()

        obs, env_reward, terminated, truncated, info = self.env.step(action)
        new_pos = self._get_position()

        control_value = self._control_values[new_pos]
        # The environment tracks visits; its visited observation channel uses the same record
        is_new_cell = info["new_cell"]

        intrinsic_reward = control_value * self.scale if is_new_cell else 0.0
        total_reward = env_reward + intrinsic_reward
        components = {'control': intrinsic_reward}

        # Add reward details in info dict for logging and debugging
        info.update({
            "id_reward": intrinsic_reward,
            "env_reward": env_reward,
            "freedom": self._compute_directional_freedom(new_pos),
            "control_value": control_value,
            **{f"id_{k}": v for k, v in components.items()}
        })

        self.total_intrinsic_reward += intrinsic_reward
        self.step_count += 1

        for k, v in components.items():
            self.reward_components[k].append(v)

        if self.step_count % 500 == 0:
            self._print_debug(env_reward, intrinsic_reward, total_reward, components)

        return obs, total_reward, terminated, truncated, info

    def _print_debug(self, env_rew, id_rew, total_rew, components):
        """
        Print a periodic summary of reward details.
        Triggered every 500 steps. Shows average intrinsic reward
        and recent average control reward over the last 500 steps.
        """
        avg_ir = self.total_intrinsic_reward / self.step_count
        print(f"Step {self.step_count} | Env: {env_rew:.3f} | ID: {id_rew:.3f} | "
              f"Total: {total_rew:.3f} | Avg ID: {avg_ir:.3f}")
        if len(self.reward_components.get('control', [])) > 100:
            avg_control = np.mean(self.reward_components['control'][-500:])
            print(f"  Recent avg control reward: {avg_control:.3f}")

    def reset(self, **kwargs):
        """Reset the environment, and rebuild the control map for the current grid."""
        obs, info = self.env.reset(**kwargs)
        self._build_control_map()
        return obs, info

    def get_stats(self):
        """
        Return a summary of intrinsic reward statistics accumulated
        since the last wrapper instantiation. Used by EmpiricalIDCallback.
        """
        stats = {
            "avg_intrinsic_reward": self.total_intrinsic_reward / (self.step_count + 1e-8),
            "total_steps": self.step_count
        }
        if 'control' in self.reward_components:
            stats["avg_control"] = np.mean(self.reward_components['control'])
            stats["std_control"] = np.std(self.reward_components['control'])
        return stats
