from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

from src.tasks.amp_lya.mdp.reference_motion import ReferenceMotionTracker

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot", joint_names=(".*",))


def lyapunov_error(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    pos_scale: float = 1.0,
    vel_scale: float = 0.05,
) -> torch.Tensor:
    """Lyapunov error state: scaled joint pos/vel deviation from reference motion."""
    asset: Entity = env.scene[asset_cfg.name]
    joint_ids = asset_cfg.joint_ids
    if isinstance(joint_ids, list):
        joint_ids = torch.tensor(joint_ids, device=env.device)

    ref_pos, ref_vel = ReferenceMotionTracker.get().get_reference(env, asset_cfg)
    joint_pos = asset.data.joint_pos[:, joint_ids]
    joint_vel = asset.data.joint_vel[:, joint_ids]

    pos_error = (joint_pos - ref_pos) * pos_scale
    vel_error = (joint_vel - ref_vel) * vel_scale
    return torch.cat([pos_error, vel_error], dim=-1)
