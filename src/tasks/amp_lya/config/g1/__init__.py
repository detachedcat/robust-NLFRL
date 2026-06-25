from mjlab.tasks.registry import register_mjlab_task
from src.tasks.amp_lya.rl import LYAOnPolicyRunner

from .env_cfgs import g1_lya_flat_env_cfg
from .rl_cfg import g1_lya_ppo_runner_cfg

register_mjlab_task(
    task_id="Unitree-G1-LYA-Flat",
    env_cfg=g1_lya_flat_env_cfg(),
    play_env_cfg=g1_lya_flat_env_cfg(play=True),
    rl_cfg=g1_lya_ppo_runner_cfg(),
    runner_cls=LYAOnPolicyRunner,
)
