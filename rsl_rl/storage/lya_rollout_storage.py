from __future__ import annotations

import torch

from rsl_rl.storage.rollout_storage import RolloutStorage


class LyaRolloutStorage(RolloutStorage):
    """Rollout storage extended with Lyapunov error states."""

    def __init__(
        self,
        training_type,
        num_envs,
        num_transitions_per_env,
        obs_shape,
        privileged_obs_shape,
        actions_shape,
        rnd_state_shape=None,
        error_dim: int | None = None,
        device="cpu",
    ):
        super().__init__(
            training_type,
            num_envs,
            num_transitions_per_env,
            obs_shape,
            privileged_obs_shape,
            actions_shape,
            rnd_state_shape,
            device,
        )
        self.error_dim = error_dim
        if self.error_dim is not None:
            self.lyapunov_states = torch.zeros(
                num_transitions_per_env, num_envs, self.error_dim, device=self.device
            )
            self.next_lyapunov_states = torch.zeros(
                num_transitions_per_env, num_envs, self.error_dim, device=self.device
            )

    def add_transitions(
        self,
        transition: RolloutStorage.Transition,
        lyapunov_state: torch.Tensor | None = None,
        next_lyapunov_state: torch.Tensor | None = None,
    ):
        step_idx = self.step
        super().add_transitions(transition)
        if (
            self.error_dim is not None
            and lyapunov_state is not None
            and next_lyapunov_state is not None
        ):
            self.lyapunov_states[step_idx].copy_(lyapunov_state)
            self.next_lyapunov_states[step_idx].copy_(next_lyapunov_state)

    def mini_batch_generator(self, num_mini_batches, num_epochs=8):
        if self.training_type != "rl":
            raise ValueError("This function is only available for reinforcement learning training.")

        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches
        indices = torch.randperm(num_mini_batches * mini_batch_size, requires_grad=False, device=self.device)

        observations = self.observations.flatten(0, 1)
        if self.privileged_observations is not None:
            privileged_observations = self.privileged_observations.flatten(0, 1)
        else:
            privileged_observations = observations

        actions = self.actions.flatten(0, 1)
        values = self.values.flatten(0, 1)
        returns = self.returns.flatten(0, 1)
        old_actions_log_prob = self.actions_log_prob.flatten(0, 1)
        advantages = self.advantages.flatten(0, 1)
        old_mu = self.mu.flatten(0, 1)
        old_sigma = self.sigma.flatten(0, 1)
        dones_flat = self.dones.flatten(0, 1).float()

        if self.rnd_state_shape is not None:
            rnd_state = self.rnd_state.flatten(0, 1)
        else:
            rnd_state = None

        lyapunov_states = None
        next_lyapunov_states = None
        if self.error_dim is not None:
            lyapunov_states = self.lyapunov_states.flatten(0, 1)
            next_lyapunov_states = self.next_lyapunov_states.flatten(0, 1)

        for _ in range(num_epochs):
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                end = (i + 1) * mini_batch_size
                batch_idx = indices[start:end]

                if rnd_state is not None:
                    rnd_state_batch = rnd_state[batch_idx]
                else:
                    rnd_state_batch = None

                lyapunov_states_batch = lyapunov_states[batch_idx] if lyapunov_states is not None else None
                next_lyapunov_states_batch = (
                    next_lyapunov_states[batch_idx] if next_lyapunov_states is not None else None
                )

                yield (
                    observations[batch_idx],
                    privileged_observations[batch_idx],
                    actions[batch_idx],
                    values[batch_idx],
                    advantages[batch_idx],
                    returns[batch_idx],
                    old_actions_log_prob[batch_idx],
                    old_mu[batch_idx],
                    old_sigma[batch_idx],
                    (None, None),
                    None,
                    rnd_state_batch,
                    lyapunov_states_batch,
                    next_lyapunov_states_batch,
                    dones_flat[batch_idx],
                )
