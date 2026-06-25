from __future__ import annotations

from copy import deepcopy

import torch
import torch.nn as nn
import torch.optim as optim

from rsl_rl.algorithms.amp_ppo import AMPPPO
from rsl_rl.algorithms.neural_lyapunov.tclf import TwinControlLyapunovFunction
from rsl_rl.storage.lya_rollout_storage import LyaRolloutStorage


def _polyak_update(params, target_params, tau: float) -> None:
    for param, target_param in zip(params, target_params):
        target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)


class LyaPPO(AMPPPO):
    """AMPPPO extended with Twin Control Lyapunov Function co-training."""

    def __init__(
        self,
        policy,
        discriminator,
        amp_data,
        amp_normalizer,
        device="cpu",
        **kwargs,
    ):
        self.lqf_loss_cnst = kwargs.pop("lqf_loss_cnst", 0.01)
        self.tclf_coef = kwargs.pop("tclf_coef", 0.2)
        self.tclf_tau = kwargs.pop("tclf_tau", 0.005)
        self.lyapunov_warmup_iters = kwargs.pop("lyapunov_warmup_iters", 50)
        self.lyapunov_ramp_iters = kwargs.pop("lyapunov_ramp_iters", 200)
        self.lyapunov_ve_gate_threshold = kwargs.pop("lyapunov_ve_gate_threshold", 0.03)
        self.lyapunov_lqf_gate_threshold = kwargs.pop("lyapunov_lqf_gate_threshold", None)
        tclf_hidden_dims = kwargs.pop("tclf_hidden_dims", [256, 128, 64])
        self.tclf_ub = kwargs.pop("tclf_ub", 20.0)
        lie_derivative_upper = kwargs.pop("lie_derivative_upper", 0.2)

        super().__init__(
            policy,
            discriminator,
            amp_data,
            amp_normalizer,
            device=device,
            **kwargs,
        )

        if hasattr(policy, "std"):
            num_actions = int(policy.std.shape[0])
        elif hasattr(policy, "log_std"):
            num_actions = int(policy.log_std.shape[0])
        else:
            raise ValueError("Cannot infer num_actions from policy for LyaPPO.")

        self.error_dim = 2 * num_actions
        lf_structure = [self.error_dim] + list(tclf_hidden_dims) + [1]
        lqf_structure = [self.error_dim + num_actions] + list(tclf_hidden_dims) + [1]
        sink = [0.0] * self.error_dim

        self.tclf = TwinControlLyapunovFunction(
            lf_structure=lf_structure,
            lqf_structure=lqf_structure,
            ub=self.tclf_ub,
            sink=sink,
            lie_derivative_upper=lie_derivative_upper,
            device=self.device,
        )
        self.tclf_target = deepcopy(self.tclf)
        self.tclf_target.requires_grad_(False)
        self.tclf_optimizer = optim.Adam(self.tclf.parameters(), lr=self.learning_rate)
        self._lyapunov_update_count = 0
        self._current_ly_scale = 0.0
        self._batch_metric_sums: dict[str, float] = {}

    def _lyapunov_schedule_scale(self) -> float:
        it = self._lyapunov_update_count
        warmup = self.lyapunov_warmup_iters
        ramp = self.lyapunov_ramp_iters
        if it < warmup:
            return 0.0
        if ramp <= 0:
            return 1.0
        if it < warmup + ramp:
            return float(it - warmup) / float(max(ramp, 1))
        return 1.0

    def init_storage(
        self,
        training_type,
        num_envs,
        num_transitions_per_env,
        actor_obs_shape,
        critic_obs_shape,
        actions_shape,
    ):
        if self.rnd:
            rnd_state_shape = [self.rnd.num_states]
        else:
            rnd_state_shape = None
        self.storage = LyaRolloutStorage(
            training_type,
            num_envs,
            num_transitions_per_env,
            actor_obs_shape,
            critic_obs_shape,
            actions_shape,
            rnd_state_shape,
            error_dim=self.error_dim,
            device=self.device,
        )

    def process_env_step(
        self,
        rewards,
        dones,
        infos,
        amp_obs,
        lyapunov_state=None,
        next_lyapunov_state=None,
    ):
        step_idx = self.storage.step
        super().process_env_step(rewards, dones, infos, amp_obs)
        if self.error_dim is not None and lyapunov_state is not None and next_lyapunov_state is not None:
            self.storage.lyapunov_states[step_idx].copy_(lyapunov_state)
            self.storage.next_lyapunov_states[step_idx].copy_(next_lyapunov_state)

    def update(self):
        self._lyapunov_update_count += 1
        return super().update()

    def _split_storage_sample(self, sample):
        (
            obs_batch,
            critic_obs_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            hid_states_batch,
            masks_batch,
            rnd_state_batch,
            lyapunov_states_batch,
            next_lyapunov_states_batch,
            dones_batch,
        ) = sample
        core = (
            obs_batch,
            critic_obs_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            hid_states_batch,
            masks_batch,
            rnd_state_batch,
        )
        extra = {
            "lyapunov_states": lyapunov_states_batch,
            "next_lyapunov_states": next_lyapunov_states_batch,
            "dones": dones_batch,
        }
        return core, extra

    def _repeat_extra_for_augmentation(self, extra, num_aug: int):
        if num_aug <= 1 or not extra:
            return extra
        out = dict(extra)
        if extra.get("lyapunov_states") is not None:
            out["lyapunov_states"] = extra["lyapunov_states"].repeat(num_aug, 1)
            out["next_lyapunov_states"] = extra["next_lyapunov_states"].repeat(num_aug, 1)
            out["dones"] = extra["dones"].repeat(num_aug, 1)
        return out

    def _on_update_begin(self) -> None:
        self._current_ly_scale = self._lyapunov_schedule_scale()
        self._batch_metric_sums = {
            "tclf": 0.0,
            "lqf_actor": 0.0,
            "lyapunov_gate_ratio": 0.0,
            "lyapunov_mean_v": 0.0,
        }

    def _sync_extra_optimizer_lr(self) -> None:
        for param_group in self.tclf_optimizer.param_groups:
            param_group["lr"] = self.learning_rate

    def _zero_extra_optimizer_grads(self) -> None:
        self.tclf_optimizer.zero_grad()

    def _compute_tclf_loss(self, current_error, actions_batch, next_error, lf_ve, not_done):
        return self.tclf.loss(
            s0=current_error,
            a=actions_batch,
            s1=next_error,
            current_q=lf_ve,
            not_done=not_done,
        )

    def _augment_loss_before_backward(self, loss, ctx):
        extra = ctx["extra"]
        lyapunov_states = extra.get("lyapunov_states")
        next_lyapunov_states = extra.get("next_lyapunov_states")
        dones = extra.get("dones")
        ly_scale = self._current_ly_scale
        metrics = {
            "tclf": 0.0,
            "lqf_actor": 0.0,
            "lyapunov_gate_ratio": 0.0,
            "lyapunov_mean_v": 0.0,
        }

        if ly_scale <= 0.0 or lyapunov_states is None or next_lyapunov_states is None:
            return loss, metrics

        obs_batch = ctx["obs_batch"]
        actions_batch = ctx["actions_batch"]

        with torch.no_grad():
            lf_ve = self.tclf.forward_lf(lyapunov_states).detach().squeeze(-1)
            if dones is not None:
                not_done_mask = 1.0 - dones.view(-1).float()
            else:
                not_done_mask = torch.ones_like(lf_ve)
            valid_mask = not_done_mask > 0.5

        if valid_mask.any():
            tclf_loss_dict = self._compute_tclf_loss(
                lyapunov_states,
                actions_batch,
                next_lyapunov_states,
                lf_ve,
                not_done_mask,
            )
            raw_tclf = tclf_loss_dict["loss_sum"]
            tclf_loss = raw_tclf * ly_scale * self.tclf_coef
        else:
            raw_tclf = torch.zeros((), device=self.device)
            tclf_loss = raw_tclf

        curr_actions = self.policy.act_inference(obs_batch)
        lqf_per = self.tclf_target.forward_lqf(lyapunov_states, curr_actions).squeeze(-1)
        metrics["lyapunov_mean_v"] = lf_ve.mean().item()

        if self.lyapunov_ve_gate_threshold is not None:
            mask = (lf_ve > self.lyapunov_ve_gate_threshold).float()
        elif self.lyapunov_lqf_gate_threshold is not None:
            mask = (lqf_per > self.lyapunov_lqf_gate_threshold).float()
        else:
            mask = torch.ones_like(lf_ve)
        mask = mask * not_done_mask

        denom = mask.sum().clamp_min(1.0)
        metrics["lyapunov_gate_ratio"] = (mask.sum() / mask.numel()).item()
        lqf_actor_term = (lqf_per * mask).sum() / denom

        loss = loss + ly_scale * self.lqf_loss_cnst * lqf_actor_term
        ctx["tclf_loss"] = tclf_loss
        ctx["tclf_should_step"] = bool(valid_mask.any())
        metrics["tclf"] = raw_tclf.item()
        metrics["lqf_actor"] = lqf_actor_term.item()
        return loss, metrics

    def _extra_optimizer_steps(self, ctx) -> None:
        extra = ctx["extra"]
        if (
            self._current_ly_scale <= 0.0
            or extra.get("lyapunov_states") is None
            or extra.get("next_lyapunov_states") is None
            or not ctx.get("tclf_should_step", False)
        ):
            return
        tclf_loss = ctx.get("tclf_loss")
        if tclf_loss is None or not torch.isfinite(tclf_loss):
            return
        tclf_loss.backward()
        nn.utils.clip_grad_norm_(self.tclf.parameters(), self.max_grad_norm)
        self.tclf_optimizer.step()
        _polyak_update(self.tclf.parameters(), self.tclf_target.parameters(), self.tclf_tau)

    def _accumulate_batch_metrics(self, batch_metrics) -> None:
        for key, value in batch_metrics.items():
            self._batch_metric_sums[key] = self._batch_metric_sums.get(key, 0.0) + float(value)

    def _merge_update_metrics(self, loss_dict):
        count = max(getattr(self, "_last_effective_updates", 1), 1)
        for key, total in self._batch_metric_sums.items():
            loss_dict[key] = total / count
        loss_dict["lyapunov_scale"] = self._current_ly_scale
        return loss_dict
