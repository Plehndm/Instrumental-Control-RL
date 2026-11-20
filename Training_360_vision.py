import argparse
import gymnasium as gym
import minigrid
import numpy as np
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from Reward_Algorithm import *
from Grid_Maze_360_vision import *
from DQN_Agent import make_dqn, resolve_device, snapshot_weights, weights_equal

# Register the customized environment once
if "BabyAI-CleanTask-v0" not in gym.envs.registration.registry:
    gym.register(
        id='BabyAI-CleanTask-v0',
        entry_point=CleanTaskEnv,
    )

class CoveringMultiMapWrapper(gym.Wrapper):
    """
    Training wrapper that provides exploration across maps.
    Each 'set' contains a fixed map, a fixed number of valid cells and steps. Once all valid cells on the current map are iterated, a new map
    is generated from the next seed.
    """
    def __init__(self, env, actions_per_set=30, base_seed=1000):
        super().__init__(env)
        self.actions_per_set = actions_per_set
        self.base_seed = base_seed

        self.steps_in_current_set = 0
        self.current_spawn_idx = 0
        self.global_map_counter = 0          

        self.current_map_seed = None
        self.current_spawn_sequence = []

        self.set_counter = 0

        self._load_new_map()

    def _load_new_map(self):
        """
        Load a new map using the next seed, enumerate all valid cells
        (position + direction), shuffle them, and reset the spawn index.
        """
        self.current_map_seed = self.base_seed + self.global_map_counter

        self.env.reset(seed=self.current_map_seed)

        base_env = self.env
        while hasattr(base_env, 'env'):
            base_env = base_env.env

        spawn_points = []
        for x in range(1, base_env.width - 1):
            for y in range(1, base_env.height - 1):
                cell = base_env.grid.get(x, y)
                if cell is None or (hasattr(cell, 'can_overlap') and cell.can_overlap()):
                    for direction in range(4):
                        spawn_points.append((x, y, direction))

        rng = np.random.RandomState(self.current_map_seed)
        rng.shuffle(spawn_points)

        self.current_spawn_sequence = spawn_points
        self.current_spawn_idx = 0

        print(f"\nLoading new map #{self.global_map_counter} "
              f"(seed={self.current_map_seed}): {len(spawn_points)} spawn points")

        self.global_map_counter += 1

    def reset(self, **kwargs):
        """
        Start a new set: pick the next cell on the current map,
        transfer the agent position, and regenerate the observation.
        Loads a new map when all cells are exhausted.
        """

        if self.current_spawn_idx >= len(self.current_spawn_sequence):
            self._load_new_map()

        pos_x, pos_y, direction = self.current_spawn_sequence[self.current_spawn_idx]

        obs, info = self.env.reset(seed=self.current_map_seed)

        base_env = self.env
        while hasattr(base_env, 'env'):
            base_env = base_env.env

        base_env.agent_pos = np.array([pos_x, pos_y])
        base_env.agent_dir = direction

        raw_obs = base_env.gen_obs()
        obs = self.env.observation(raw_obs)

        self.steps_in_current_set = 0
        self.set_counter += 1

        info['map_seed'] = self.current_map_seed
        info['map_number'] = self.global_map_counter - 1
        info['spawn_idx'] = self.current_spawn_idx
        info['set_number'] = self.set_counter

        if self.set_counter % 100 == 0:
            total_spawns = len(self.current_spawn_sequence)
            print(f"  Completed {self.set_counter} sets | "
                  f"Map #{self.global_map_counter - 1} | "
                  f"Spawn {self.current_spawn_idx}/{total_spawns}")

        return obs, info

    def step(self, action):
        """
        Execute one step. Truncate the episode when reaching the maximum steps and move on to the next cell.
        """
        obs, reward, terminated, truncated, info = self.env.step(action)

        self.steps_in_current_set += 1

        if self.steps_in_current_set >= self.actions_per_set:
            self.current_spawn_idx += 1
            truncated = True
            info['set_complete'] = True

        return obs, reward, terminated, truncated, info

class EmpiricalIDCallback(BaseCallback):
    """
    Callback for the control agent training run.
    Periodically prints episode reward, episode length, exploration rate and
    intrinsic reward statistics from the EmpiricalInstrumentalDivergenceWrapper.
    """
    def __init__(self, verbose=0, print_freq=5000):
        super().__init__(verbose)
        self.print_freq = print_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.print_freq == 0 and len(self.model.ep_info_buffer) > 0:
            ep_rew = np.mean([ep['r'] for ep in self.model.ep_info_buffer])
            ep_len = np.mean([ep['l'] for ep in self.model.ep_info_buffer])

            print(f"\n{'='*60}")
            print(f"Step {self.n_calls}")
            print(f"  Episode reward:   {ep_rew:.2f}")
            print(f"  Episode length:   {ep_len:.1f}")
            print(f"  Exploration rate: {self.model.exploration_rate:.3f}")

            env = self.training_env.envs[0]
            while hasattr(env, 'env'):
                if hasattr(env, 'get_stats'):
                    stats = env.get_stats()
                    print(f"  ID stats:")
                    print(f"    Avg ID reward: {stats.get('avg_intrinsic_reward', 0):.3f}")
                    if 'avg_freedom' in stats:
                        print(f"    avg_freedom: {stats['avg_freedom']:.3f}")
                    break
                env = env.env

            print(f"{'='*60}")

        return True

def make_env(room_size, num_walls, num_pockets, use_control_reward=False,
             control_scale=0.1, actions_per_set=20, base_seed=1000):

    def _init():
        env = gym.make(
            "BabyAI-CleanTask-v0",
            room_size=room_size,
            num_walls=num_walls,
            num_pockets=num_pockets,
            use_distance_reward=False,
            has_goal=False,
            render_mode="rgb_array"
        )

        env = ImgObsWrapper(env)

        env = CoveringMultiMapWrapper(
            env,
            actions_per_set=actions_per_set,
            base_seed=base_seed
        )

        if use_control_reward:
            env = EmpiricalInstrumentalDivergenceWrapper(
                env,
                scale=control_scale,
            )

        env = Monitor(env)

        return env

    return _init

def train_with_curriculum(mode, seed=0, control_scale=100.0, total_steps=100000, device="auto"):
    """
    Phase 1: exploration-only training (has_goal=False) across many maps with DQN.
    - 'baseline': no reward at all, so its Q-network must stay exactly at its
      initial weights. This is verified at the end of training.
    - 'control':  freedom-based intrinsic reward; the only source of learning.
    Use the same seed for both modes so they start from identical weights.
    """
    print(f"Initializing {mode.upper()} mode (seed={seed})...")

    curriculum = [{
        "room_size": 11,
        "num_walls": 22,
        "num_pockets": 10,
        "steps": total_steps,
        "actions_per_set": 20,
        "base_seed": 1000
    }]

    use_control = (mode == "control")
    model = None
    initial_weights = None

    for i, stage in enumerate(curriculum):
        # No VecNormalize: adaptive reward statistics would differ between agents
        env = DummyVecEnv([
            make_env(
                stage['room_size'],
                stage['num_walls'],
                stage['num_pockets'],
                use_control_reward=use_control,
                control_scale=control_scale,
                actions_per_set=stage['actions_per_set'],
                base_seed=stage['base_seed']
            )
        ])

        if model is None:
            model = make_dqn(env, seed=seed, device=device,
                             tensorboard_log=f"./logs_multimap_{mode}/")
            initial_weights = snapshot_weights(model)
        else:
            model.set_env(env)

        callbacks = [
            CheckpointCallback(
                save_freq=50000,
                save_path=f'./checkpoints_{mode}_multimap/',
                name_prefix='model'
            )
        ]

        if use_control:
            callbacks.append(EmpiricalIDCallback(print_freq=10000))

        print(f"Start training...\n")
        model.learn(
            total_timesteps=stage['steps'],
            callback=callbacks,
            reset_num_timesteps=False,
            progress_bar=False
        )

    model.save(f"model_{mode}_multimap_final")
    print(f"\nTraining complete.")
    print(f"Model saved: model_{mode}_multimap_final.zip\n")

    unchanged = weights_equal(initial_weights, snapshot_weights(model))
    if mode == "baseline":
        if not unchanged:
            raise RuntimeError(
                "Baseline Q-network weights changed during Phase 1 even though it "
                "received no reward. Something other than reward is training the agent."
            )
        print("Check passed: baseline weights are bit-for-bit identical to their initial values.\n")
    else:
        print(f"Control weights changed during Phase 1: {not unchanged}\n")

    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 1: exploration pre-training (no goal).")
    parser.add_argument("--modes", nargs="+", choices=["baseline", "control"],
                        default=["baseline", "control"], help="Agents to train.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Model seed. Use the same value in Phase 2.")
    parser.add_argument("--steps", type=int, default=100000, help="Training timesteps per agent.")
    parser.add_argument("--control-scale", type=float, default=100.0,
                        help="Multiplier on the control agent's freedom reward.")
    parser.add_argument("--device", default="auto",
                        help="'auto' (NVIDIA GPU if available, else CPU), 'cuda' or 'cpu'.")
    args = parser.parse_args()
    args.device = resolve_device(args.device)

    for mode in args.modes:
        train_with_curriculum(mode, seed=args.seed, control_scale=args.control_scale,
                              total_steps=args.steps, device=args.device)
