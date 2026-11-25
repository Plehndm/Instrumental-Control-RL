import argparse
import gymnasium as gym
import minigrid
import os
import json
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from Reward_Algorithm import *
from Grid_Maze_360_vision import *
from DQN_Agent import (EPSILON, NoWarmupDQN, make_dqn, reset_optimizer, resolve_device,
                       set_epsilon, snapshot_weights, weights_equal)
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

if "BabyAI-CleanTask-v0" not in gym.envs.registration.registry:
    gym.register(
        id='BabyAI-CleanTask-v0',
        entry_point=CleanTaskEnv,
    )

# Phase 2 agents: (Phase 1 model to load, rate of random actions, learns in Phase 2)
AGENTS = {
    "random":         (None,       EPSILON, True),
    "baseline":       ("baseline", EPSILON, True),
    "control":        ("control",  EPSILON, True),
    "control_greedy": ("control",  0.0,     True),
    "control_frozen": ("control",  0.0,     False),
}

class FixedSeedWrapper(gym.Wrapper):
    """
    Forces every env.reset() call to use the same fixed max seed. 
    Also resets step_count to 0 to prevent premature episode truncation.
    """
    def __init__(self, env, seed):
        super().__init__(env)
        self.fixed_seed = seed

    def reset(self, **kwargs):
        kwargs['seed'] = self.fixed_seed
        obs, info = self.env.reset(**kwargs)

        base_env = self.env
        while hasattr(base_env, 'env'):
            base_env = base_env.env
        base_env.step_count = 0

        return obs, info

class SuccessTrackingCallback(BaseCallback):
    """
    Records each successful episode along with its relative timestep and episode length.
    Prints a rolling success summary every log_freq steps.
    """
    def __init__(self, log_freq=2000, verbose=0):
        super().__init__(verbose)
        self.log_freq         = log_freq
        self.success_episodes = []
        self.start_timestep   = None 

    def _on_training_start(self) -> None:
        self.start_timestep = self.num_timesteps  

    def _on_step(self) -> bool:
        relative_t = self.num_timesteps - self.start_timestep  

        for info in self.locals.get('infos', []):
            ep_info = info.get('episode')
            if ep_info is not None and info.get("is_success", False):
                self.success_episodes.append({
                    'timestep': relative_t, 
                    'steps':    ep_info['l'],
                    'reward':   ep_info['r'],
                })

        if relative_t % self.log_freq == 0 and relative_t > 0:
            recent = [
                e for e in self.success_episodes
                if e['timestep'] > relative_t - self.log_freq
            ]
            total = len(self.success_episodes)
            if recent:
                avg_steps = np.mean([e['steps'] for e in recent])
                print(f"  [Step {relative_t}] "
                      f"Last {self.log_freq} steps: {len(recent)} successes | "
                      f"Avg steps: {avg_steps:.1f} | "
                      f"Total successes: {total}")
            else:
                print(f"  [Step {relative_t}] "
                      f"Last {self.log_freq} steps: 0 successes | "
                      f"Total successes: {total}")
        return True

def make_env_phase2(room_size, num_walls, num_pockets, map_seed=2000):
    """
    Phase 2 environment. Every agent gets the same reward: +255 for reaching the
    red ball and nothing else. The control agent's Phase 1 reward is deliberately
    left out so that all agents are compared on the same goal-only reward.
    """
    def _init():
        env = gym.make(
            "BabyAI-CleanTask-v0",
            room_size=room_size,
            num_walls=num_walls,
            num_pockets=num_pockets,
            use_distance_reward=False,
            has_goal=True,
            render_mode="rgb_array"
        )
        env = ImgObsWrapper(env)
        env = FixedSeedWrapper(env, seed=map_seed)
        env = Monitor(env)
        return env
    return _init

def train_phase2(mode, seed=0, map_seed=2000, total_steps=100000, device="auto"):
    """
    Learning in Testing: fine-tunes agents on a fixed map with has_goal=True.
    - 'random'   mode: a fresh DQN built with the same seed as Phase 1, so its weights
                       are identical to the Baseline's Phase 1 starting weights.
    - 'baseline' mode: loads the Phase 1 baseline checkpoint and continues training.
    - 'control'  mode: loads the Phase 1 control checkpoint and continues training.
    - 'control_greedy': like 'control', but takes no random actions.
    - 'control_frozen': like 'control_greedy', but never learns in Phase 2, so it
                        follows its Phase 1 training only.
    Every agent starts Phase 2 the same way: fresh optimizer, empty replay buffer,
    the same random seed, and the same goal-only reward. There is no random warm-up:
    agents pick their own actions from step 1, and learning starts after
    learning_starts steps (1,000).
    """
    source, epsilon, learns = AGENTS[mode]
    print(f"\n{'='*60}")
    print(f"Learning in Testing: {mode.upper()} | has_goal=True | map_seed={map_seed} | seed={seed}")
    if mode != "random":
        print(f"Loading pre-trained model: model_{source}_multimap_final.zip")
    else:
        print(f"Random model: initializing from scratch (sanity check)")
    print(f"{'='*60}\n")

    print(f"Random actions: {epsilon:.0%} | learns in Phase 2: {learns} | random warm-up: none")

    # No VecNormalize: adaptive reward statistics would differ between agents
    env = DummyVecEnv([
        make_env_phase2(
            room_size=11,
            num_walls=22,
            num_pockets=10,
            map_seed=map_seed,
        )
    ])

    if mode == "random":
        model = make_dqn(env, seed=seed, device=device,
                         tensorboard_log=f"./logs_phase2_{mode}/", algo=NoWarmupDQN)
        # Sanity check: an untrained baseline should equal a fresh model with the same seed
        if os.path.exists("model_baseline_multimap_final.zip"):
            baseline = DQN.load("model_baseline_multimap_final", device=device)
            same = weights_equal(snapshot_weights(model), snapshot_weights(baseline))
            print(f"Random weights identical to Phase 1 baseline checkpoint: {same}"
                  + ("" if same else "  (expected True; check that both phases used the same --seed)"))
            del baseline
    else:
        # Load pretrained weights; hyperparameters are preserved. The replay buffer is not
        # saved in the checkpoint, so it starts empty, just like the random agent's.
        model = NoWarmupDQN.load(
            f"model_{source}_multimap_final",
            env=env,
            device=device,
            tensorboard_log=f"./logs_phase2_{mode}/",
        )

    # Identical starting conditions for every agent
    reset_optimizer(model)          # no Adam state carried over from Phase 1
    model._n_calls = 0              # align target-network update timing
    model._n_updates = 0            # restart the update counter shown in the logs
    model.set_random_seed(seed)     # same exploration / tie-break random stream

    set_epsilon(model, epsilon)
    if not learns:
        model.learning_starts = total_steps  # never reached, so no gradient updates
    phase1_weights = snapshot_weights(model)

    # Print map layout info before training begins
    env.reset()
    base_env = env.envs[0]
    while hasattr(base_env, 'env'):
        base_env = base_env.env
    print(f"Map info | agent_pos={tuple(base_env.agent_pos)} | "
          f"ball_pos={tuple(base_env.red_ball.cur_pos)}\n")

    success_callback = SuccessTrackingCallback(log_freq=2000)

    callbacks = [
        CheckpointCallback(
            save_freq=50000,
            save_path=f'./checkpoints_{mode}_phase2/',
            name_prefix='model'
        ),
        success_callback,
    ]

    print(f"Start learning...\n")

    # reset_num_timesteps=True makes every agent start learning at the same step
    model.learn(
        total_timesteps=total_steps,
        callback=callbacks,
        reset_num_timesteps=True,
        progress_bar=False,
    )
 
    if not learns and not weights_equal(phase1_weights, snapshot_weights(model)):
        raise RuntimeError(f"{mode} weights changed during Phase 2, but it should not learn.")

    # Save model
    model.save(f"model_{mode}_phase2_final")
    print(f"\nModel saved: model_{mode}_phase2_final.zip")
 
    # Save success log as JSON for later Excel export
    log_path = f"success_log_{mode}_phase2.json"
    with open(log_path, "w") as f:
        json.dump(success_callback.success_episodes, f, indent=2)
    print(f"Success log saved: {log_path}")
 
    # Print step trend
    successes = success_callback.success_episodes
    if successes:
        steps_list = [e['steps'] for e in successes]
        print(f"\nSuccess summary:")
        print(f"  Total successes: {len(successes)}")
        print(f"  Avg steps:       {np.mean(steps_list):.1f}")
        print(f"  Min steps:       {np.min(steps_list)}")
        print(f"  Max steps:       {np.max(steps_list)}")
 
        seg_size = total_steps // 5
        print(f"\n  Step trend (per {seg_size} steps):")
        for seg in range(5):
            seg_start = seg * seg_size
            seg_end   = (seg + 1) * seg_size
            seg_data  = [e['steps'] for e in successes
                         if seg_start <= e['timestep'] < seg_end]
            if seg_data:
                print(f"    {seg_start:>7}-{seg_end:<7}: "
                      f"{len(seg_data):>4} successes | "
                      f"Avg steps: {np.mean(seg_data):.1f}")
            else:
                print(f"    {seg_start:>7}-{seg_end:<7}: no successes")
    else:
        print("\nNo successful episodes during training.")
 
    print(f"\n{'='*60}")
    return model
 
def export_summary_excel(modes, total_steps, log_freq=2000,
                          output_path="phase2_summary.xlsx"):
    logs = {}
    for mode in modes:
        log_path = f"success_log_{mode}_phase2.json"
        if os.path.exists(log_path):
            with open(log_path) as f:
                logs[mode] = json.load(f)
        else:
            print(f"Warning: {log_path} not found, using empty data")
            logs[mode] = []
 
    windows = list(range(log_freq, total_steps + log_freq, log_freq))
 
    wb = Workbook()
    ws = wb.active
    ws.title = "Training Summary"
 
    # Styles
    header_font    = Font(name="Arial", bold=True, color="FFFFFF", size=11)
    subheader_font = Font(name="Arial", bold=True, size=10)
    data_font      = Font(name="Arial", size=10)
    total_font     = Font(name="Arial", bold=True, size=10)
 
    header_fill    = PatternFill("solid", start_color="1F4E79")
    subheader_fill = PatternFill("solid", start_color="2E75B6")
    alt_fill       = PatternFill("solid", start_color="EBF3FB")
    total_fill     = PatternFill("solid", start_color="D6E4F0")
 
    center = Alignment(horizontal="center", vertical="center")
    left   = Alignment(horizontal="left",   vertical="center")
 
    thin   = Side(style="thin", color="BFBFBF")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)
 
    mode_colors = {
        "random":   "C55A11",
        "baseline": "2E75B6",
        "control":  "375623",
        "control_greedy": "7030A0",
        "control_frozen": "595959",
    }
 
    # Title row
    n_cols = 1 + len(modes) * 2
    last_col = get_column_letter(n_cols)
    ws.merge_cells(f"A1:{last_col}1")
    ws["A1"] = "Phase 2 Training Summary — Successes & Avg Steps per 2000 Timesteps"
    ws["A1"].font      = Font(name="Arial", bold=True, size=13, color="FFFFFF")
    ws["A1"].fill      = header_fill
    ws["A1"].alignment = center
    ws.row_dimensions[1].height = 28
 
    # Config info row
    ws.merge_cells(f"A2:{last_col}2")
    ws["A2"] = (f"Total steps: {total_steps:,}    |    "
                f"Window: {log_freq:,} steps    |    "
                f"Modes: {', '.join(m.replace('_', ' ').capitalize() for m in modes)}")
    ws["A2"].font      = Font(name="Arial", italic=True, size=9, color="595959")
    ws["A2"].alignment = left
    ws.row_dimensions[2].height = 16
    ws.row_dimensions[3].height = 6  # spacer
 
    # Column headers rows 4-5
    col_map = {}
    start_col = 2
 
    # Step Window header (spans rows 4-5)
    ws.cell(row=4, column=1, value="Step Window")
    ws.cell(row=4, column=1).font      = subheader_font
    ws.cell(row=4, column=1).fill      = subheader_fill
    ws.cell(row=4, column=1).alignment = center
    ws.cell(row=4, column=1).border    = border
    ws.merge_cells(start_row=4, start_column=1, end_row=5, end_column=1)
    ws.column_dimensions["A"].width = 20
 
    for i, mode in enumerate(modes):
        sc = start_col + i * 2
        col_map[mode] = (sc, sc + 1)
 
        # Group header
        ws.merge_cells(start_row=4, start_column=sc, end_row=4, end_column=sc+1)
        cell = ws.cell(row=4, column=sc, value=mode.replace('_', ' ').capitalize())
        cell.font      = Font(name="Arial", bold=True, color="FFFFFF", size=11)
        cell.fill      = PatternFill("solid", start_color=mode_colors.get(mode, "2E75B6"))
        cell.alignment = center
        cell.border    = border
 
        # Sub-headers
        for col, label in [(sc, "Successes"), (sc+1, "Avg Steps")]:
            c = ws.cell(row=5, column=col, value=label)
            c.font      = subheader_font
            c.fill      = PatternFill("solid", start_color="D9E2F3")
            c.alignment = center
            c.border    = border
            ws.column_dimensions[get_column_letter(col)].width = 14
 
    ws.row_dimensions[4].height = 20
    ws.row_dimensions[5].height = 18
 
    # Data rows
    data_start_row = 6
    for idx, window_end in enumerate(windows):
        window_start = window_end - log_freq
        row  = data_start_row + idx
        fill = alt_fill if idx % 2 == 0 else PatternFill("solid", start_color="FFFFFF")
 
        c = ws.cell(row=row, column=1,
                    value=f"{window_start:,} – {window_end:,}")
        c.font      = data_font
        c.fill      = fill
        c.alignment = center
        c.border    = border
 
        for mode in modes:
            sc, ac = col_map[mode]
            recent    = [e for e in logs[mode]
                         if window_start < e['timestep'] <= window_end]
            count     = len(recent)
            avg_steps = round(np.mean([e['steps'] for e in recent]), 1) if recent else 0.0
 
            for col, val in [(sc, count), (ac, avg_steps)]:
                c = ws.cell(row=row, column=col, value=val)
                c.font      = data_font
                c.fill      = fill
                c.alignment = center
                c.border    = border
 
        ws.row_dimensions[row].height = 16
 
    # Total row
    total_row = data_start_row + len(windows)
    ws.row_dimensions[total_row].height = 18
 
    c = ws.cell(row=total_row, column=1, value="TOTAL")
    c.font      = total_font
    c.fill      = total_fill
    c.alignment = center
    c.border    = border
 
    for mode in modes:
        sc, ac = col_map[mode]
        total_success = len(logs[mode])
        all_steps     = [e['steps'] for e in logs[mode]]
        overall_avg   = round(np.mean(all_steps), 1) if all_steps else 0.0
 
        for col, val in [(sc, total_success), (ac, overall_avg)]:
            c = ws.cell(row=total_row, column=col, value=val)
            c.font      = total_font
            c.fill      = total_fill
            c.alignment = center
            c.border    = border
 
    ws.freeze_panes = ws.cell(row=data_start_row, column=2)
 
    wb.save(output_path)
    print(f"\nExcel summary saved: {output_path}")
    return output_path
 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 2: goal-directed fine-tuning on a fixed map.")
    parser.add_argument("--modes", nargs="+", choices=list(AGENTS),
                        default=list(AGENTS), help="Agents to train.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Model seed. Must match the seed used in Phase 1.")
    parser.add_argument("--map-seed", type=int, default=2000, help="Seed of the fixed test map.")
    parser.add_argument("--steps", type=int, default=100000, help="Training timesteps per agent.")
    parser.add_argument("--device", default="auto",
                        help="'auto' (NVIDIA GPU if available, else CPU), 'cuda' or 'cpu'.")
    parser.add_argument("--output", default="phase2_summary.xlsx", help="Excel summary path.")
    args = parser.parse_args()
    args.device = resolve_device(args.device)

    for mode in args.modes:
        train_phase2(
            mode=mode,
            seed=args.seed,
            map_seed=args.map_seed,
            total_steps=args.steps,
            device=args.device,
        )

    export_summary_excel(
        modes=args.modes,
        total_steps=args.steps,
        log_freq=2000,
        output_path=args.output,
    )
