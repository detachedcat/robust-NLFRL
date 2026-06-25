"""Unitree G1 LYA AMP environment configurations."""

import os

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

import src.tasks.amp_lya.mdp as lya_mdp
from src.tasks.amp_loco.config.g1.env_cfgs import g1_amp_flat_env_cfg


def g1_lya_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create Unitree G1 flat terrain LYA AMP configuration."""
    cfg = g1_amp_flat_env_cfg(play=play)

    _motion_base = os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "..", "assets", "motions", "g1", "amp"
    )
    _motion_dir = os.path.abspath(os.path.join(_motion_base, "WalkandRun"))

    cfg.observations["lyapunov"] = ObservationGroupCfg(
        terms={
            "error": ObservationTermCfg(
                func=lya_mdp.observations.lyapunov_error,
                params={
                    "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
                    "pos_scale": 1.0,
                    "vel_scale": 0.05,
                },
            ),
        },
        concatenate_terms=True,
        enable_corruption=False,
        history_length=1,
    )

    cfg.events["init_reference_motion"] = EventTermCfg(
        func=lya_mdp.reference_motion.init_reference_motion,
        mode="startup",
        params={"motion_dir": _motion_dir},
    )
    cfg.events["sync_reference_motion"] = EventTermCfg(
        func=lya_mdp.reference_motion.sync_reference_motion,
        mode="reset",
        params={"motion_dir": _motion_dir},
    )
    cfg.events["step_reference_motion"] = EventTermCfg(
        func=lya_mdp.reference_motion.step_reference_motion,
        mode="interval",
        interval_range_s=(0.0, 0.0),
        params={"motion_dir": _motion_dir},
    )

    return cfg
