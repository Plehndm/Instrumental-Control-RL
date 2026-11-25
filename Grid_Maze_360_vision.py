import gymnasium as gym
import minigrid
import numpy as np
from collections import defaultdict
from gymnasium import spaces
from minigrid.core.world_object import Ball, WorldObj
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.constants import COLORS, OBJECT_TO_IDX, COLOR_TO_IDX
from minigrid.minigrid_env import MiniGridEnv

class ImgObsWrapper(gym.ObservationWrapper):
    """
    Converts the default dict observation {"image": [...], ...} into a
    plain numpy array with shape (C, H, W), compatible with CNN-based policies.
    Supports the customized 8-channel 360-degree observation format.
    """
    def __init__(self, env):
        super().__init__(env)
        obs_shape = env.observation_space.spaces["image"].shape
        shape = (obs_shape[2], obs_shape[0], obs_shape[1])
        # The visited channel (last) only goes up to 100. Declaring that also stops
        # Stable-Baselines3 from treating the observation as a 0-255 photo: with 8
        # channels on a 7x7 view it would wrongly transpose the channel axis.
        high = np.full(shape, 255, dtype=np.uint8)
        high[-1] = 100
        self.observation_space = gym.spaces.Box(
            low=np.zeros(shape, dtype=np.uint8), high=high, dtype=np.uint8
        )

    def observation(self, obs):
        return np.transpose(obs["image"], (2, 0, 1))

class MultiWallCell(WorldObj):
    """
    A grid cell supporting directional walls on any of its four edges.
    Multiple walls on the same cell create pocket structures that restrict
    agent movement and reduce directional freedom.
    """
    def __init__(self, color='grey'):
        super().__init__('wall', color)
        self.walls = {
            'top': False,
            'bottom': False,
            'left': False,
            'right': False
        }

    def add_wall(self, edge):
        """Activate a wall on the specified side (top/bottom/left/right)."""
        if edge in self.walls:
            self.walls[edge] = True
        return self

    def has_wall(self, edge):
        """Return True if this cell has a wall on the given side."""
        return self.walls.get(edge, False)

    def can_overlap(self):
        """Allow the agent to occupy this cell despite wall is present."""
        return True

    def see_behind(self):
        """Block visibility through this cell."""
        return False

    def encode(self):
        """
        Encode wall configuration as a bitmask in the third channel.
        Bits: top=1, bottom=2, left=4, right=8.
        """
        state = (
            (1 if self.walls['top'] else 0) |
            (2 if self.walls['bottom'] else 0) |
            (4 if self.walls['left'] else 0) |
            (8 if self.walls['right'] else 0)
        )
        return (
            OBJECT_TO_IDX['wall'],
            COLOR_TO_IDX.get(self.color, 0),
            state
        )

    def render(self, img):
        """Present walls as colored strips on the corresponding sides."""
        tile_size = img.shape[0]
        wall_thickness = 8
        wall_color = np.array(COLORS[self.color])
        if self.walls['top']:
            img[0:wall_thickness, :, :] = wall_color
        if self.walls['bottom']:
            img[tile_size - wall_thickness:tile_size, :, :] = wall_color
        if self.walls['left']:
            img[:, 0:wall_thickness, :] = wall_color
        if self.walls['right']:
            img[:, tile_size - wall_thickness:tile_size, :] = wall_color

    def __repr__(self):
        walls = ''.join([
            'T' if self.walls['top'] else '-',
            'B' if self.walls['bottom'] else '-',
            'L' if self.walls['left'] else '-',
            'R' if self.walls['right'] else '-'
        ])
        return f"MultiWall({self.color}, [{walls}])"

class CleanTaskEnv(MiniGridEnv):
    """
    Custom MiniGrid environment with:
    - 4-action space: turn left, turn right, move forward, move backward
    - 8-channel 360-degree local observation (7x7 square vision)
    - MultiWallCell obstacles that create pockets with low directional freedom
    - Optional red ball goal with sparse reward (+255 on reach)
    """
    def __init__(self, room_size=15, num_walls=20, num_pockets=4,
                 use_distance_reward=False, has_goal=False,**kwargs):
        self.num_walls = num_walls
        self.num_pockets = num_pockets
        self.use_distance_reward = use_distance_reward
        self.has_goal = has_goal
        mission_space = MissionSpace(mission_func=lambda: "go to the red ball")
        self.position_history = []
        self.episode_visit_counts = defaultdict(int)

        super().__init__(
            mission_space=mission_space,
            grid_size=room_size,
            max_steps=2000,
            see_through_walls=False,    
            **kwargs
        )

        # 4-action space: turn left, turn right, forward, backward
        self.action_space = spaces.Discrete(4)
        self.actions = type('obj', (object,), {
            'left': 0,
            'right': 1,
            'forward': 2,
            'backward': 3,
        })()

        # Redefine observation space as 8-channel 360-degree view
        vs = self.agent_view_size
        self.observation_space = spaces.Dict({
            **self.observation_space.spaces,
            "image": spaces.Box(
                low=0, high=255,
                shape=(vs, vs, 8),
                dtype=np.uint8
            )
        })

    def _compute_directional_freedom(self, position):
        """
        Compute the directional freedom of a cell as a value in [0, 1].
        Counts how many directions are movable (considering both exit walls on the current cell and entry walls on
        neighboring cells). Returns accessible_count / 4.0.
        """
        direction_map = {
        (1,  0): ('right', 'left'),
        (-1, 0): ('left',  'right'),
        (0,  1): ('bottom', 'top'),
        (0, -1): ('top',   'bottom'),
        }

        accessible = 0
        for (dx, dy), (exit_dir, entry_dir) in direction_map.items():
            target = (position[0] + dx, position[1] + dy)

            if not (0 < target[0] < self.width - 1 and
                0 < target[1] < self.height - 1):
                continue

            current_cell = self.grid.get(*position)
            if isinstance(current_cell, MultiWallCell):
                if current_cell.has_wall(exit_dir):
                    continue

            target_cell = self.grid.get(*target)
            if target_cell is not None:
                if target_cell.type == 'wall' and not isinstance(target_cell, MultiWallCell):
                    continue
                if isinstance(target_cell, MultiWallCell):
                    if target_cell.has_wall(entry_dir):
                        continue

            accessible += 1

        return accessible / 4.0

    # Absolute directions, indexed like agent_dir: 0=right, 1=down, 2=left, 3=up
    _DIR_VECS = ((1, 0), (0, 1), (-1, 0), (0, -1))
    _DIR_EDGES = ('right', 'bottom', 'left', 'top')

    def _edge_blocked(self, position, direction):
        """
        True if the edge of position facing an absolute direction cannot be crossed:
        a wall on this cell's side, a wall on the neighbour's side, or the outer wall.
        These are the same rules as movement and directional freedom.
        """
        dx, dy = self._DIR_VECS[direction]
        tx, ty = position[0] + dx, position[1] + dy
        if not (0 < tx < self.width - 1 and 0 < ty < self.height - 1):
            return True
        cell = self.grid.get(*position)
        if isinstance(cell, MultiWallCell) and cell.has_wall(self._DIR_EDGES[direction]):
            return True
        target = self.grid.get(tx, ty)
        if isinstance(target, MultiWallCell):
            return target.has_wall(self._DIR_EDGES[(direction + 2) % 4])
        return target is not None and target.type == 'wall'

    def _visible_cells(self, r):
        """
        Offsets (dx, dy) of the cells the agent can see within r cells, in all
        directions. A cell is visible if it can be reached by stepping only away
        from the agent without crossing a walled edge. Sight depends on edges, not on
        cell type, so standing in a cell that has walls never blinds the agent. The
        outer wall next to a visible cell is visible but blocks sight beyond it.
        """
        ax, ay = int(self.agent_pos[0]), int(self.agent_pos[1])
        visible = {(0, 0)}
        offsets = [(dx, dy) for dx in range(-r, r + 1) for dy in range(-r, r + 1)]
        # Nearest first, so a cell's predecessors are always decided before it
        offsets.sort(key=lambda o: abs(o[0]) + abs(o[1]))
        for dx, dy in offsets[1:]:
            x, y = ax + dx, ay + dy
            if not (0 <= x < self.width and 0 <= y < self.height):
                continue
            is_border = not (0 < x < self.width - 1 and 0 < y < self.height - 1)
            # (predecessor offset, direction of travel from it into this cell)
            steps = []
            if dx:
                steps.append(((dx - (1 if dx > 0 else -1), dy), 0 if dx > 0 else 2))
            if dy:
                steps.append(((dx, dy - (1 if dy > 0 else -1)), 1 if dy > 0 else 3))
            for prev, direction in steps:
                if prev not in visible:
                    continue
                px, py = ax + prev[0], ay + prev[1]
                if not (0 < px < self.width - 1 and 0 < py < self.height - 1):
                    continue  # cannot see past the outer wall
                if is_border:
                    cell = self.grid.get(px, py)
                    blocked = (isinstance(cell, MultiWallCell)
                               and cell.has_wall(self._DIR_EDGES[direction]))
                else:
                    blocked = self._edge_blocked((px, py), direction)
                if not blocked:
                    visible.add((dx, dy))
                    break
        return visible

    def gen_obs(self):
        """
        Generate an 8-channel 360-degree observation centered on the agent and
        rotated so that the agent always faces the top row.
        Channels:
        0    object type
        1    color
        2-5  blocked edge of the cell: ahead, behind, left, right RELATIVE TO THE AGENT
             (100 if it cannot be crossed, else 0)
        6    directional freedom * 100; the red ball cell is 255, the biggest signal
        7    visited: 100 for cells entered this episode (including the agent's own
             cell). A memory, so it is filled in even for cells that are now hidden.
        The agent's own cell is included, so it can see the walls around itself.
        """
        vs = self.agent_view_size
        r = vs // 2
        ax, ay = int(self.agent_pos[0]), int(self.agent_pos[1])
        facing = int(self.agent_dir)
        fx, fy = self._DIR_VECS[facing]
        rx, ry = self._DIR_VECS[(facing + 1) % 4]
        # Absolute directions of the agent's ahead, behind, left and right
        relative_dirs = [facing, (facing + 2) % 4, (facing + 3) % 4, (facing + 1) % 4]

        obs_image = np.zeros((vs, vs, 8), dtype=np.uint8)
        visible = self._visible_cells(r)

        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                x, y = ax + dx, ay + dy
                row = r - (dx * fx + dy * fy)
                col = r + (dx * rx + dy * ry)

                if (dx, dy) == (0, 0) or (x, y) in self.episode_visit_counts:
                    obs_image[row, col, 7] = 100

                if (dx, dy) not in visible:
                    continue

                cell = self.grid.get(x, y)
                if cell is None:
                    obs_image[row, col, 0] = OBJECT_TO_IDX['empty']
                else:
                    encoded = cell.encode()
                    obs_image[row, col, 0] = encoded[0]
                    obs_image[row, col, 1] = encoded[1]

                if not (0 < x < self.width - 1 and 0 < y < self.height - 1):
                    obs_image[row, col, 2:6] = 100  # outer wall: blocked on every side
                    continue

                for i, direction in enumerate(relative_dirs):
                    if self._edge_blocked((x, y), direction):
                        obs_image[row, col, 2 + i] = 100

                if cell is not None and cell.type == 'ball':
                    obs_image[row, col, 6] = 255
                else:
                    obs_image[row, col, 6] = int(self._compute_directional_freedom((x, y)) * 100)

        return {
            "image": obs_image,
            "direction": self.agent_dir,
            "mission": self.mission,
        }

    def _manhattan_distance(self, pos1, pos2):
        """Compute Manhattan distance between two grid positions."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def start_visit_tracking(self):
        """Forget all visited cells; only the agent's current cell counts as visited."""
        current_pos = tuple(int(v) for v in self.agent_pos)
        self.episode_visit_counts = defaultdict(int)
        self.episode_visit_counts[current_pos] = 1
        self.position_history = [current_pos]

    def reset(self, **kwargs):
        # Clear first: super().reset() already generates an observation
        self.episode_visit_counts = defaultdict(int)
        obs, info = super().reset(**kwargs)
        self.start_visit_tracking()

        total_cells = (self.grid.width - 2) * (self.grid.height - 2)
        info["coverage"] = (1 / total_cells) * 100
        info["unique_visited"] = 1
        info["agent_pos"] = tuple(self.agent_pos)

        return obs, info

    def _gen_grid(self, width, height):
        """
        Generate the grid layout:
        1. Place maze boundary walls.
        2. Randomly create num_walls single-side MultiWallCells.
        3. Randomly create num_pockets multi-sides MultiWallCells.
        4. Place red ball (has_goal=True) and agent at random open positions.
        """
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        edges = ['top', 'bottom', 'left', 'right']
        walls_placed = 0
        while walls_placed < self.num_walls:
            pos = (self._rand_int(1, width - 1), self._rand_int(1, height - 1))
            cell = self.grid.get(*pos)
            if cell is None:
                cell = MultiWallCell()
                self.grid.set(pos[0], pos[1], cell)
            if isinstance(cell, MultiWallCell):
                edge_type = self._rand_elem(edges)
                if not cell.has_wall(edge_type):
                    cell.add_wall(edge_type)
                    walls_placed += 1

        pockets_created = 0
        while pockets_created < self.num_pockets:
            pos = (self._rand_int(1, width - 1), self._rand_int(1, height - 1))
            cell = self.grid.get(*pos)
            if cell is None:
                pocket = MultiWallCell()
                num_walls_in_pocket = self._rand_int(2, 4)
                selected_edges = self._rand_subset(edges, num_walls_in_pocket)
                for edge in selected_edges:
                    pocket.add_wall(edge)
                self.grid.set(pos[0], pos[1], pocket)
                pockets_created += 1

        if self.has_goal:
            self.red_ball = Ball('red')
            self.red_ball.can_overlap = lambda: True
            self.place_obj(self.red_ball)
        else:
            self.red_ball = None
        self.place_agent()

    def step(self, action):
        """
        Execute one step. Handles the customized backward action separately.
        Updates visit counts and coverage stats each step.
        If has_goal=True, checks whether agents reach the red ball and assign reward.
        """

        # Handle backward action independently
        if action == self.actions.backward:
            obs, reward, terminated, truncated, info = self._step_backward()

        # Forward blocked by wall: increase step count but stay in place
        elif action == self.actions.forward and not self._can_move_forward():
            self.step_count += 1
            obs = self.gen_obs()
            current_pos = tuple(self.agent_pos)
            if current_pos not in self.episode_visit_counts:
                self.episode_visit_counts[current_pos] = 0
            unique_visited = len(self.episode_visit_counts)
            total_cells = (self.grid.width - 2) * (self.grid.height - 2)
            info = {
                "coverage": (unique_visited / total_cells) * 100,
                "unique_visited": unique_visited,
                "agent_pos": current_pos,
                "new_cell": False
            }
            return obs, 0.0, False, self.step_count >= self.max_steps, info

        else:
            _, _, terminated, truncated, info = super().step(action)
            obs = self.gen_obs()

        current_pos = tuple(self.agent_pos)
        self.position_history.append(current_pos)
        info["new_cell"] = current_pos not in self.episode_visit_counts
        self.episode_visit_counts[current_pos] += 1

        unique_visited = len(self.episode_visit_counts)
        total_cells = (self.grid.width - 2) * (self.grid.height - 2)
        info["coverage"] = (unique_visited / total_cells) * 100
        info["unique_visited"] = unique_visited
        info["agent_pos"] = current_pos

        reward = 0.0
        is_success = False

        if self.has_goal:
            new_distance = self._manhattan_distance(
                self.agent_pos,
                self.red_ball.cur_pos
            )
            if np.array_equal(self.agent_pos, self.red_ball.cur_pos):
                reward = 255.0
                terminated = True
                is_success = True

            info["distance"] = new_distance
            info["distance_reward"] = reward

        info["is_success"] = is_success

        return obs, reward, terminated, truncated, info

    def _step_backward(self):
        """
        Move a step in the direction opposite to the agent's facing.
        """
        # agent_dir: 0=right, 1=down, 2=left, 3=up
        reverse_dir = {0: (-1, 0), 1: (0, -1), 2: (1, 0), 3: (0, 1)}
        dx, dy = reverse_dir[self.agent_dir]
        back_pos = (self.agent_pos[0] + dx, self.agent_pos[1] + dy)

        self.step_count += 1

        if self._can_move_to(back_pos, moving_backward=True):
            self.agent_pos = np.array(back_pos)

        obs = self.gen_obs()
        truncated = self.step_count >= self.max_steps
        info = {"agent_pos": tuple(self.agent_pos)}

        return obs, 0.0, False, truncated, info

    def _can_move_to(self, target_pos, moving_backward=False):
        """
        Check whether the agent can move to target_pos.
        """
        tx, ty = int(target_pos[0]), int(target_pos[1])
        if not (0 < tx < self.width - 1 and 0 < ty < self.height - 1):
            return False

        current_cell = self.grid.get(*self.agent_pos)
        target_cell = self.grid.get(tx, ty)

        direction_map = {0: 'right', 1: 'bottom', 2: 'left', 3: 'top'}
        opposite_map = {'right': 'left', 'left': 'right', 'top': 'bottom', 'bottom': 'top'}

        if moving_backward:
            moving_direction = opposite_map[direction_map[self.agent_dir]]
        else:
            moving_direction = direction_map[self.agent_dir]

        exit_direction = moving_direction
        entry_direction = opposite_map[moving_direction]

        if isinstance(current_cell, MultiWallCell):
            if current_cell.has_wall(exit_direction):
                return False

        if isinstance(target_cell, MultiWallCell):
            if target_cell.has_wall(entry_direction):
                return False

        if target_cell is not None and not isinstance(target_cell, MultiWallCell):
            if not target_cell.can_overlap():
                return False

        return True

    def _can_move_forward(self):
        """Check whether the agent can move forward."""
        return self._can_move_to(tuple(self.front_pos), moving_backward=False)
