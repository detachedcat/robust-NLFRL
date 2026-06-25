from __future__ import annotations

import os
from types import MethodType

import torch

from rsl_rl.algorithms import LyaPPO
from rsl_rl.runners.amp_on_policy_runner import AmpOnPolicyRunner, _unpack_obs

_LYAP_CFG_KEYS = (
    "lqf_loss_cnst",
    "tclf_coef",
    "tclf_tau",
    "lyapunov_warmup_iters",
    "lyapunov_ramp_iters",
    "lyapunov_ve_gate_threshold",
    "lyapunov_lqf_gate_threshold",
    "tclf_hidden_dims",
    "tclf_ub",
    "lie_derivative_upper",
)


class LyaOnPolicyRunner(AmpOnPolicyRunner):
    """AMP on-policy runner extended with Lyapunov error tracking and TCLF checkpoints."""

    def __init__(self, env, train_cfg: dict, log_dir: str | None = None, device="cpu"):
        train_cfg = dict(train_cfg)
        train_cfg["algorithm"] = dict(train_cfg["algorithm"])
        for key in _LYAP_CFG_KEYS:
            if key in train_cfg:
                train_cfg["algorithm"][key] = train_cfg[key]
        super().__init__(env, train_cfg, log_dir, device)
        if isinstance(self.alg, LyaPPO):
            self.current_error = self._read_lyapunov_error()
        else:
            self.current_error = None

    def _read_lyapunov_error(self, extras: dict | None = None) -> torch.Tensor:
        if extras is None:
            _, extras = _unpack_obs(self.env.get_observations())
        lyap = extras.get("observations", {}).get("lyapunov")
        if lyap is not None:
            return lyap.to(self.device).clone()
        return torch.zeros(self.env.num_envs, self.alg.error_dim, device=self.device)

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):
        if not isinstance(self.alg, LyaPPO):
            return super().learn(num_learning_iterations, init_at_random_ep_len)

        self.current_error = self._read_lyapunov_error()
        orig_process = self.alg.process_env_step

        def process_with_lyapunov(rewards, dones, infos, amp_obs):
            lyap = infos.get("observations", {}).get("lyapunov")
            if lyap is not None:
                next_error = lyap.to(self.device).clone()
            else:
                next_error = torch.zeros_like(self.current_error)
            # Keep two versions of "next error":
            # - raw_next_error: true post-step observation (after auto-reset if done)
            # - stored_next_error: terminal-aware successor used for transition storage
            raw_next_error = next_error.clone()
            stored_next_error = next_error
            # Keep terminal transition semantics aligned with AMP handling:
            # for done envs, use pre-step error as terminal successor proxy.
            reset_env_ids = (dones > 0).nonzero(as_tuple=False).flatten()
            if len(reset_env_ids) > 0:
                stored_next_error[reset_env_ids] = self.current_error[reset_env_ids]
            orig_process(
                rewards,
                dones,
                infos,
                amp_obs,
                lyapunov_state=self.current_error,
                next_lyapunov_state=stored_next_error,
            )
            # For the next step's s_t, always use the true env observation.
            self.current_error = raw_next_error

        self.alg.process_env_step = MethodType(
            lambda alg, rewards, dones, infos, amp_obs: process_with_lyapunov(
                rewards, dones, infos, amp_obs
            ),
            self.alg,
        )
        try:
            super().learn(num_learning_iterations, init_at_random_ep_len)
        finally:
            self.alg.process_env_step = orig_process

    def save(self, path: str, infos=None):
        super().save(path, infos)
        if not isinstance(self.alg, LyaPPO):
            return
        tclf_path = path.replace("model_", "tclf_", 1)
        torch.save(
            {
                "tclf_state_dict": self.alg.tclf.state_dict(),
                "tclf_target_state_dict": self.alg.tclf_target.state_dict(),
                "tclf_optimizer_state_dict": self.alg.tclf_optimizer.state_dict(),
                "lyapunov_update_count": self.alg._lyapunov_update_count,
                "iter": self.current_learning_iteration,
            },
            tclf_path,
        )

    def load(self, path: str, load_optimizer: bool = True):
        infos = super().load(path, load_optimizer)
        if not isinstance(self.alg, LyaPPO):
            return infos
        tclf_path = path.replace("model_", "tclf_", 1)
        if os.path.isfile(tclf_path):
            loaded = torch.load(tclf_path, weights_only=False)
            self.alg.tclf.load_state_dict(loaded["tclf_state_dict"])
            self.alg.tclf_target.load_state_dict(loaded["tclf_target_state_dict"])
            if load_optimizer and "tclf_optimizer_state_dict" in loaded:
                self.alg.tclf_optimizer.load_state_dict(loaded["tclf_optimizer_state_dict"])
            if "lyapunov_update_count" in loaded:
                self.alg._lyapunov_update_count = loaded["lyapunov_update_count"]
        return infos

    def train_mode(self):
        super().train_mode()
        if isinstance(self.alg, LyaPPO):
            self.alg.tclf.train()
            self.alg.tclf_target.train()

    def eval_mode(self):
        super().eval_mode()
        if isinstance(self.alg, LyaPPO):
            self.alg.tclf.eval()
            self.alg.tclf_target.eval()
