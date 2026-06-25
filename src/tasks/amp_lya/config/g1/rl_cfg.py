"""RL configuration for Unitree G1 LYA AMP locomotion task."""

import os
from dataclasses import dataclass, field
from typing import List

from mjlab.rl import RslRlModelCfg, RslRlPpoAlgorithmCfg

from src.tasks.amp_loco.config.g1.rl_cfg import RslRlAmpRunnerCfg, _MOTION_DATA_DIR


@dataclass
class RslRlAmpLyaRunnerCfg(RslRlAmpRunnerCfg):
    """Extended runner config with Lyapunov parameters."""

    lqf_loss_cnst: float = 0.01
    tclf_coef: float = 0.2
    tclf_tau: float = 0.005
    lyapunov_warmup_iters: int = 50
    lyapunov_ramp_iters: int = 200
    lyapunov_ve_gate_threshold: float = 0.03
    lyapunov_lqf_gate_threshold: float | None = None
    tclf_hidden_dims: List[int] = field(default_factory=lambda: [256, 128, 64])
    tclf_ub: float = 20.0
    lie_derivative_upper: float = 0.2


def g1_lya_ppo_runner_cfg() -> RslRlAmpLyaRunnerCfg:
    """Create RL runner configuration for Unitree G1 LYA AMP locomotion task."""
    return RslRlAmpLyaRunnerCfg(
        actor=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=True,
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
        ),
        critic=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=True,
        ),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.005,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1.0e-3,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
            class_name="LyaPPO",
        ),
        experiment_name="g1_lya_locomotion",
        logger="tensorboard",
        save_interval=100,
        num_steps_per_env=24,
        max_iterations=100001,
        amp_reward_coef=0.1,
        amp_motion_files=os.path.normpath(_MOTION_DATA_DIR),
        amp_num_preload_transitions=200000,
        amp_task_reward_lerp=0.75,
        amp_discr_hidden_dims=[1024, 512, 256],
        min_normalized_std=[0.05] * 29,
        amp_body_names=(
            "pelvis",
            "left_hip_roll_link",
            "left_knee_link",
            "left_ankle_roll_link",
            "right_hip_roll_link",
            "right_knee_link",
            "right_ankle_roll_link",
            "left_shoulder_roll_link",
            "left_elbow_link",
            "left_wrist_yaw_link",
            "right_shoulder_roll_link",
            "right_elbow_link",
            "right_wrist_yaw_link",
        ),
        amp_anchor_name="torso_link",
        lqf_loss_cnst=0.01,
        tclf_coef=0.2,
        tclf_tau=0.005,
        lyapunov_warmup_iters=50,
        lyapunov_ramp_iters=200,
        lyapunov_ve_gate_threshold=0.03,
        lyapunov_lqf_gate_threshold=None,
        tclf_hidden_dims=[256, 128, 64],
        tclf_ub=20.0,
        lie_derivative_upper=0.2,
    )
