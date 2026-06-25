from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

from src.tasks.amp_loco.mdp.events import MotionResetManager

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


class ReferenceMotionTracker:
    """Per-env reference motion phase for Lyapunov error computation."""

    _instance: ReferenceMotionTracker | None = None

    def __init__(self) -> None:
        self.motion_dir: str | None = None
        self.local_phase: torch.Tensor | None = None
        self.clip_id: torch.Tensor | None = None
        self.use_recovery: torch.Tensor | None = None

    @classmethod
    def get(cls) -> ReferenceMotionTracker:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def init(self, env: ManagerBasedRlEnv, motion_dir: str) -> None:
        if self.motion_dir == motion_dir and self.local_phase is not None:
            return
        self.motion_dir = motion_dir
        self._ensure_buffers(env)
        MotionResetManager.get().init(env=env, motion_dir=motion_dir)

    def _ensure_buffers(self, env: ManagerBasedRlEnv) -> None:
        if self.local_phase is None or self.local_phase.shape[0] != env.num_envs:
            self.local_phase = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
            self.clip_id = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
            self.use_recovery = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def _global_idx(self, clip_id: torch.Tensor, local_phase: torch.Tensor) -> torch.Tensor:
        assert self.motion_dir is not None and self.use_recovery is not None
        mgr = MotionResetManager.get()
        walk_starts = mgr.walk_clip_starts[self.motion_dir]
        rec_starts = mgr.recovery_clip_starts.get(self.motion_dir, walk_starts)
        starts = torch.zeros_like(local_phase)
        normal_mask = ~self.use_recovery
        if normal_mask.any():
            starts[normal_mask] = walk_starts[clip_id[normal_mask]]
        if self.use_recovery.any():
            starts[self.use_recovery] = rec_starts[clip_id[self.use_recovery]]
        return starts + local_phase

    def sync_from_reset(self, env: ManagerBasedRlEnv, env_ids: torch.Tensor) -> None:
        """Sync clip/phase from AMP reset_from_motion frame indices."""
        if self.motion_dir is None or len(env_ids) == 0:
            return
        self._ensure_buffers(env)
        mgr = MotionResetManager.get()
        if mgr.reset_frame_idx is None or mgr.reset_use_recovery is None:
            return

        idx = mgr.reset_frame_idx[env_ids]
        use_recovery = mgr.reset_use_recovery[env_ids]
        self.use_recovery[env_ids] = use_recovery

        walk_starts = mgr.walk_clip_starts[self.motion_dir]
        walk_lengths = mgr.walk_clip_lengths[self.motion_dir]
        if motion_dir_has_recovery := self.motion_dir in mgr.recovery_clip_starts:
            rec_starts = mgr.recovery_clip_starts[self.motion_dir]
            rec_lengths = mgr.recovery_clip_lengths[self.motion_dir]
        else:
            rec_starts = walk_starts
            rec_lengths = walk_lengths

        clip_id = torch.zeros_like(idx)
        local_phase = torch.zeros_like(idx)
        for use_rec in (False, True):
            mask = use_recovery == use_rec
            if not mask.any():
                continue
            sub_idx = idx[mask]
            starts = rec_starts if use_rec and motion_dir_has_recovery else walk_starts
            lengths = rec_lengths if use_rec and motion_dir_has_recovery else walk_lengths
            sub_clip = torch.searchsorted(starts, sub_idx, right=True) - 1
            sub_clip = sub_clip.clamp(min=0, max=lengths.shape[0] - 1)
            sub_phase = sub_idx - starts[sub_clip]
            clip_id[mask] = sub_clip
            local_phase[mask] = sub_phase

        self.clip_id[env_ids] = clip_id
        self.local_phase[env_ids] = local_phase

    def step(self, env: ManagerBasedRlEnv) -> None:
        if self.motion_dir is None or self.local_phase is None or self.clip_id is None:
            return
        self._ensure_buffers(env)
        mgr = MotionResetManager.get()
        walk_lengths = mgr.walk_clip_lengths[self.motion_dir]
        if self.motion_dir in mgr.recovery_clip_lengths:
            rec_lengths = mgr.recovery_clip_lengths[self.motion_dir]
        else:
            rec_lengths = walk_lengths

        assert self.use_recovery is not None
        lengths = torch.ones_like(self.local_phase)
        normal_mask = ~self.use_recovery
        if normal_mask.any():
            lengths[normal_mask] = walk_lengths[self.clip_id[normal_mask]]
        if self.use_recovery.any():
            lengths[self.use_recovery] = rec_lengths[self.clip_id[self.use_recovery]]
        self.local_phase = (self.local_phase + 1) % lengths

    def get_reference(
        self,
        env: ManagerBasedRlEnv,
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        asset: Entity = env.scene[asset_cfg.name]
        joint_ids = asset_cfg.joint_ids
        if isinstance(joint_ids, list):
            joint_ids = torch.tensor(joint_ids, device=env.device)
        elif isinstance(joint_ids, slice):
            total_joints = asset.data.joint_pos.shape[1]
            joint_ids = torch.arange(total_joints, device=env.device)[joint_ids]
        num_joints = int(joint_ids.shape[0])

        if (
            self.motion_dir is None
            or self.local_phase is None
            or self.clip_id is None
            or self.use_recovery is None
        ):
            zeros = torch.zeros(env.num_envs, num_joints, device=env.device)
            return zeros, zeros.clone()

        global_idx = self._global_idx(self.clip_id, self.local_phase)
        mgr = MotionResetManager.get()
        ref_pos = torch.zeros(env.num_envs, num_joints, device=env.device)
        ref_vel = torch.zeros(env.num_envs, num_joints, device=env.device)

        normal_mask = ~self.use_recovery
        if normal_mask.any():
            frames = mgr.walk_run_frames[self.motion_dir]
            idx = global_idx[normal_mask]
            ref_pos[normal_mask] = frames["joint_pos"][idx][:, joint_ids]
            ref_vel[normal_mask] = frames["joint_vel"][idx][:, joint_ids]

        rec_mask = self.use_recovery
        if rec_mask.any():
            frames = mgr.recovery_frames.get(self.motion_dir, mgr.walk_run_frames[self.motion_dir])
            idx = global_idx[rec_mask]
            ref_pos[rec_mask] = frames["joint_pos"][idx][:, joint_ids]
            ref_vel[rec_mask] = frames["joint_vel"][idx][:, joint_ids]

        # Keep reference joint position limits consistent with reset clamping.
        soft_joint_pos_limits = asset.data.soft_joint_pos_limits
        if soft_joint_pos_limits is not None:
            limits = soft_joint_pos_limits[:, joint_ids]
            ref_pos = torch.clamp(ref_pos, min=limits[..., 0], max=limits[..., 1])

        return ref_pos, ref_vel


def init_reference_motion(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    motion_dir: str,
) -> None:
    del env_ids
    ReferenceMotionTracker.get().init(env, motion_dir)


def sync_reference_motion(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    motion_dir: str,
) -> None:
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
    ReferenceMotionTracker.get().init(env, motion_dir)
    ReferenceMotionTracker.get().sync_from_reset(env, env_ids)


def step_reference_motion(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    motion_dir: str,
) -> None:
    del env_ids, motion_dir
    ReferenceMotionTracker.get().step(env)
