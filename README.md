# Agentic AI Experiment: Instrumental Control as a Reward Surrogate

## Overview

This project investigates whether a reinforcement learning agent that treats
instrumental control (local environmental freedom) as a reward adapts faster 
to reward distribution shifts than an agent trained without this signal. 
The experiment is built on a custom MiniGrid-based environment and 
compares three agents — a **control agent**, a **baseline agent**, and 
a **random agent** — across a two-phase training pipeline.

### Core hypothesis

If an agent is trained to prefer high-controllability regions (cells with
more open directions, fewer walls and pockets), does this exploration bias
help or hurt its ability to adapt when a reward (a red ball) is introduced
in a new location?

## Repository structure

```
Grid_Maze_360_vision.py    Custom MiniGrid environment, observation encoding, grid generation
Reward_Algorithm.py        Freedom-based intrinsic reward wrapper 
DQN_Agent.py               Shared DQN setup: network, zero-head Q-network, random-tie ε-greedy
Training_360_vision.py     Phase 1 training: exploration-only, no goal
Learning in Testing.py     Phase 2 training: goal-directed fine-tuning 
```

## Environment design (`Grid_Maze_360_vision.py`)

### CleanTaskEnv

A custom MiniGrid environment with:
- **4-action space**: turn left, turn right, move forward, move backward
- **8-channel local observation** (7×7 window): object type, color, four
  blocked-edge flags (ahead, behind, left, right relative to the agent), a
  directional freedom value, and a visited flag marking the cells entered so
  far this episode
- **MultiWallCell**: grid cells that support directional walls on any of
  their four edges, used to construct "pockets" — low-freedom regions
  that restrict movement
- **Optional red ball goal**: when `has_goal=True`, a ball is placed on the
  grid; reaching it terminates the episode with reward `+255`

### Directional freedom

The core metric of the experiment. For any grid cell, freedom is defined as:

```
freedom(cell) = accessible_directions / 4.0
```

where a direction is accessible if neither the current cell's exit edge
nor the neighboring cell's entry edge has a wall. This value is encoded
directly into the 7th observation channel, making local controllability
visible to the policy network without requiring it to be inferred from
raw pixel data.

### 360-degree observation

`gen_obs()` builds a single 7×7 window around the agent, rotated so the agent
always faces the top row. Only walled edges block sight, so an agent standing
in a cell that has walls can still see out through its open sides. The window
includes the agent's own cell, so it can see the walls around itself.

## Reward design (`Reward_Algorithm.py`)

### EmpiricalInstrumentalDivergenceWrapper

Adds an intrinsic reward to the control agent (Phase 1 only) for entering a
cell for the first time in an episode:

```
control_value     = control_decay ** steps_to_nearest_control_point
intrinsic_reward  = control_value * scale   (new cell; otherwise 0)
total_reward      = env_reward + intrinsic_reward
```

Control points are the cells with full directional freedom (1.0). Distance is
counted in walking steps around walls, so control values rise along every
shortest path to a control point. `control_decay` is 0.8. Turning, blocked
moves and revisits earn nothing, and the starting cell counts as visited.
The environment's visit record decides what is new, and the same record fills
the visited observation channel, so the agent can see which cells will not pay.
The wrapper independently recomputes freedom from the grid state 
(rather than reading the observation tensor), so it can be attached or removed 
without changing the environment itself.

## Training pipeline

All agents are trained with **DQN** (Q-learning with a neural network), set up
so that **reward is the only thing that can change an agent**
(`DQN_Agent.py`):

- **Zero-initialised output layer.** The hidden layers (`ImprovedCNN` plus a
  64-64 MLP) use normal random initialisation, but the final layer of the
  Q-network starts at exactly zero, so Q(s, a) = 0 everywhere. With zero
  reward the TD target and TD error are exactly 0, so every gradient is 0
  and the network never changes.
- **Constant ε-greedy with random tie-breaking** (ε = 0.1). The exploration
  rate is identical for every agent and never decays. Ties between equal
  Q-values are broken uniformly at random, so an agent whose Q-values are all
  zero acts uniformly at random rather than always choosing action 0.
- **No adaptive reward normalisation** (no `VecNormalize`), and every agent
  uses the same hyperparameters and the same seed.

### Phase 1 — Exploration pretraining (`Training_360_vision.py`)

Both agents are trained from scratch with **no goal present**
(`has_goal=False`) across many generated maps:

- **Baseline agent**: no reward at all. Its Q-network therefore stays exactly
  at its initial weights; the script verifies this bit-for-bit at the end of
  training and raises an error if anything changed.
- **Control agent**: the new-cell control reward from
  `EmpiricalInstrumentalDivergenceWrapper`, which is its only learning signal.

`CoveringMultiMapWrapper` ensures the agent trains across a continuous
stream of new maps and spawn points, ensuring it's learning a generalizable exploration policy.

### Phase 2 — Goal-directed fine-tuning (`Learning in Testing.py`)

Phase 2 introduces `has_goal=True` and fine-tunes each agent on a single
fixed map (via `FixedSeedWrapper`) so that learning curves are comparable
across agents. Every agent starts Phase 2 the same way: a fresh optimizer
(no Adam state carried over), an empty replay buffer, and the same
random seed. There is no random-action warm-up: agents pick their own actions
from step 1, and learning starts after 1,000 steps.

Five conditions are tested:
- **random**: a fresh DQN built with the same seed as Phase 1. Its weights
  are identical to the Baseline's Phase 1 starting weights (the script checks
  this), so it serves as a sanity check: **Random and Baseline should produce
  identical results**.
- **baseline**: Phase 1 baseline checkpoint, continued training
- **control**: Phase 1 control checkpoint, continued training
- **control_greedy**: same as control, but with no random actions (the others use 10%)
- **control_frozen**: no random actions and no Phase 2 learning, so it follows its
  Phase 1 training only

All agents get the same Phase 2 reward: +255 for reaching the red ball
and nothing else. The control reward is not used in Phase 2, so the agents are
compared on the same goal-only reward.

`SuccessTrackingCallback` records every successful episode
along with its relative timestep and episode length, enabling step-trend
analysis across training. Results are exported to a formatted Excel
workbook (`export_summary_excel`) summarising success counts and average
steps-to-goal per time window for direct comparison across the agents.

## Running the experiment

Commands are for Windows PowerShell, run from the repository root.

### 1. One-time setup

```powershell
python -m venv liljeholm_venv
.\liljeholm_venv\Scripts\Activate.ps1
python -m pip install gymnasium minigrid stable-baselines3 torch numpy openpyxl tensorboard
```

In later sessions, only the activation line is needed. Use `python -m pip`
rather than `pip`: if the venv folder is ever moved, the `pip.exe` launcher
breaks ("Fatal error in launcher"), but `python -m pip` keeps working.

### 2. Phase 1 — pretraining (creates `model_baseline_multimap_final.zip` and `model_control_multimap_final.zip`)

```powershell
python Training_360_vision.py --seed 0
```

This trains the baseline and then the control agent. To train one at a time:

```powershell
python Training_360_vision.py --modes baseline --seed 0
python Training_360_vision.py --modes control --seed 0
```

The baseline run must end with
`Check passed: baseline weights are bit-for-bit identical to their initial values.`

### 3. Phase 2 — goal-directed fine-tuning (creates `phase2_summary.xlsx`)

```powershell
python "Learning in Testing.py" --seed 0
```

This trains the five agents in turn, then writes the Excel
summary. The random run should print
`Random weights identical to Phase 1 baseline checkpoint: True`.

**Use the same `--seed` in both phases.** To repeat the experiment over
several seeds, run steps 2 and 3 once per seed, giving each Phase 2 run its
own output file (each run overwrites the model and log files of the
previous one, so run the seeds one after another):

```powershell
foreach ($s in 0..4) {
    python Training_360_vision.py --seed $s
    python "Learning in Testing.py" --seed $s --output "phase2_summary_seed$s.xlsx"
}
```

### Options

| Option | Phase | Default | Meaning |
|---|---|---|---|
| `--modes` | both | all agents | Which agents to train |
| `--seed` | both | `0` | Model seed; must match between phases |
| `--steps` | both | `100000` | Training timesteps per agent |
| `--control-scale` | 1 | `100.0` | Multiplier on the new-cell control reward |
| `--device` | both | `auto` | `auto` uses an NVIDIA GPU if available, otherwise the CPU; `cuda` or `cpu` forces one. Deterministic GPU settings are on, so results are reproducible on either |
| `--map-seed` | 2 | `2000` | Seed of the fixed test map |
| `--output` | 2 | `phase2_summary.xlsx` | Excel summary path |

### Outputs

| File / folder | Contents |
|---|---|
| `model_{mode}_multimap_final.zip` | Phase 1 models |
| `model_{mode}_phase2_final.zip` | Phase 2 models |
| `success_log_{mode}_phase2.json` | Every successful Phase 2 episode |
| `phase2_summary.xlsx` | Successes and average steps per 2,000-step window |
| `checkpoints_*/` | Checkpoints every 50,000 steps |
| `logs_*/` | TensorBoard logs (`tensorboard --logdir .`) |

## Requirements

```
gymnasium
minigrid
stable-baselines3
torch
numpy
openpyxl
tensorboard
```
