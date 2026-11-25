"""
Shared DQN setup used by both training phases.

Design goal: the ONLY thing that can change an agent's network is reward.
  - The Q-network's output layer starts at exactly zero, so Q(s, a) = 0 everywhere.
    With zero reward the TD target is 0 + gamma * 0 = 0, the TD error is 0, and every
    gradient (including those into the hidden layers, which are multiplied by the
    zero output weights) is exactly 0. A zero-reward agent therefore never changes.
  - Exploration is constant epsilon-greedy with uniformly random tie-breaking.
    An all-zero Q-function makes every action a tie, so an untrained agent acts
    uniformly at random, and the exploration rate is identical for every agent
    regardless of the scale of its rewards.
  - No adaptive reward normalisation (VecNormalize), and every agent gets a fresh
    optimizer and the same seed at the start of Phase 2.
"""
import os

# Deterministic GPU maths, so identical agents (Random vs Baseline) stay identical on CUDA.
# Must be set before CUDA is initialised.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import torch.nn as nn
from stable_baselines3 import DQN

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=True)
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.utils import LinearSchedule
from stable_baselines3.dqn.policies import DQNPolicy, QNetwork


class ImprovedCNN(BaseFeaturesExtractor):
    """
    Dual-stream CNN feature extractor for the grid observation.
    - obj_stream:   processes the full observation (object type, color, wall bits, freedom)
    - coord_stream: processes a fixed 2-channel coordinate grid (normalized x/y positions)
    Both streams are flattened and fused via a linear layer with LayerNorm and Tanh,
    producing a features_dim-dimensional representation for the Q-value head.
    """
    def __init__(self, observation_space, features_dim=256):
        super().__init__(observation_space, features_dim)

        obs_shape = observation_space.shape
        in_channels = obs_shape[0]
        H, W = obs_shape[1], obs_shape[2]

        self.obj_stream = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.Flatten(),
        )

        self.coord_stream = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.Flatten(),
        )

        # Fixed coordinate grid in [-1, 1], centred on the agent
        cx, cy = W // 2, H // 2
        x_coords = (torch.arange(W).float() - cx) / max(cx, 1)
        y_coords = (torch.arange(H).float() - cy) / max(cy, 1)
        x_grid = x_coords.unsqueeze(0).expand(H, -1)
        y_grid = y_coords.unsqueeze(1).expand(-1, W)
        coord_grid = torch.stack([x_grid, y_grid], dim=0).unsqueeze(0)
        self.register_buffer('coord_grid', coord_grid)

        with torch.no_grad():
            dummy_obj   = torch.zeros(1, in_channels, H, W)
            dummy_coord = torch.zeros(1, 2, H, W)
            n_obj   = self.obj_stream(dummy_obj).shape[1]
            n_coord = self.coord_stream(dummy_coord).shape[1]

        self.fusion = nn.Sequential(
            nn.Linear(n_obj + n_coord, features_dim),
            nn.LayerNorm(features_dim),
            nn.Tanh()
        )

        self._apply_init()

    def _apply_init(self):
        """Orthogonal init (gain=sqrt(2)) for all Conv2d and Linear layers, zero biases."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(m.weight, gain=1.414)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, observations):
        # 255 (SB3 no longer applies this itself, see ImgObsWrapper) x 10
        obs = observations.float() / 2550.0
        B =obs.shape[0]
        coords = self.coord_grid.expand(B, -1, -1, -1)
        obj_feat   = self.obj_stream(obs)
        coord_feat = self.coord_stream(coords)
        combined = torch.cat([obj_feat, coord_feat], dim=1)
        return self.fusion(combined)


class RandomTieQNetwork(QNetwork):
    """
    Q-network whose greedy action breaks ties uniformly at random.
    SB3's default argmax always returns the first tied index, which would make an
    all-zero Q-function always pick action 0 ("turn left") instead of acting randomly.
    """
    def _predict(self, observation, deterministic=True):
        q_values = self(observation)
        is_max = q_values == q_values.max(dim=1, keepdim=True).values
        noise = torch.rand_like(q_values)  # in [0, 1)
        # Non-maximal actions get -1 so only tied maxima can win the argmax
        return torch.where(is_max, noise, torch.full_like(noise, -1.0)).argmax(dim=1).reshape(-1)


class ZeroHeadDQNPolicy(DQNPolicy):
    """
    DQN policy with random-tie greedy actions and a zero-initialised output layer.
    Hidden layers keep their normal random init (avoids the symmetry problem of an
    all-zero network); only the final Linear layer is zeroed, so Q(s, a) starts at 0.
    """
    def make_q_net(self) -> QNetwork:
        net_args = self._update_features_extractor(self.net_args, features_extractor=None)
        q_net = RandomTieQNetwork(**net_args).to(self.device)
        output_layer = [m for m in q_net.q_net if isinstance(m, nn.Linear)][-1]
        nn.init.zeros_(output_layer.weight)
        nn.init.zeros_(output_layer.bias)
        return q_net


# Shared hyperparameters: identical for every agent in both phases
EPSILON = 0.1
DQN_HYPERPARAMS = dict(
    learning_rate=1e-4,
    buffer_size=100_000,
    learning_starts=1_000,
    batch_size=64,
    gamma=0.99,
    train_freq=4,
    gradient_steps=1,
    target_update_interval=1_000,
    # Constant epsilon: initial == final, so the linear schedule never changes it
    exploration_initial_eps=EPSILON,
    exploration_final_eps=EPSILON,
    exploration_fraction=1.0,
    max_grad_norm=10,
)

POLICY_KWARGS = dict(
    features_extractor_class=ImprovedCNN,
    features_extractor_kwargs=dict(features_dim=256),
    net_arch=[64, 64],
)


def resolve_device(device="auto"):
    """'auto' -> 'cuda' if an NVIDIA GPU is usable, otherwise 'cpu'. Prints the choice."""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda"):
        print(f"Using device: {device} ({torch.cuda.get_device_name(0)})")
    else:
        print(f"Using device: {device}")
    return device


class NoWarmupDQN(DQN):
    """
    DQN without SB3's random-action warm-up: the agent picks its own actions from the
    first step. learning_starts then only delays gradient updates, not the policy.
    Used in Phase 2 so that what was learned in Phase 1 is visible immediately.
    """
    def _sample_action(self, learning_starts, action_noise=None, n_envs=1):
        return super()._sample_action(0, action_noise, n_envs)


def set_epsilon(model, epsilon):
    """Use a constant rate of random actions."""
    model.exploration_initial_eps = epsilon
    model.exploration_final_eps = epsilon
    model.exploration_rate = epsilon
    model.exploration_schedule = LinearSchedule(epsilon, epsilon, 1.0)


def make_dqn(env, seed, tensorboard_log=None, device="auto", verbose=1, algo=DQN):
    """Build a fresh DQN agent. The same seed always gives the same initial weights."""
    return algo(
        ZeroHeadDQNPolicy,
        env,
        policy_kwargs=POLICY_KWARGS,
        seed=seed,
        device=device,
        tensorboard_log=tensorboard_log,
        verbose=verbose,
        **DQN_HYPERPARAMS,
    )


def reset_optimizer(model):
    """Replace the optimizer with a fresh one so no Adam state carries over between phases."""
    policy = model.policy
    policy.optimizer = policy.optimizer_class(
        policy.q_net.parameters(),
        lr=model.lr_schedule(1),
        **policy.optimizer_kwargs,
    )


def snapshot_weights(model):
    """Detached copy of the Q-network weights, for later comparison."""
    return {k: v.detach().clone() for k, v in model.q_net.state_dict().items()}


def weights_equal(weights_a, weights_b):
    """True if two weight snapshots are bit-for-bit identical."""
    return weights_a.keys() == weights_b.keys() and all(
        torch.equal(weights_a[k].cpu(), weights_b[k].cpu()) for k in weights_a
    )
