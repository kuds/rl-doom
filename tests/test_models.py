"""Shape and dtype tests for the CNN networks."""

from __future__ import annotations

import torch

from rl_doom.models import ActorCriticNetwork, DQNNetwork


def test_dqn_network_output_shape() -> None:
    net = DQNNetwork(in_channels=4, n_actions=3)
    x = torch.zeros(2, 4, 84, 84, dtype=torch.float32)
    out = net(x)
    assert out.shape == (2, 3)


def test_dqn_network_handles_uint8_input() -> None:
    net = DQNNetwork(in_channels=4, n_actions=3)
    x = torch.randint(0, 256, (1, 4, 84, 84), dtype=torch.uint8)
    out = net(x)
    assert out.shape == (1, 3)
    assert out.dtype == torch.float32


def test_actor_critic_returns_logits_and_value() -> None:
    net = ActorCriticNetwork(in_channels=4, n_actions=5)
    x = torch.zeros(3, 4, 84, 84, dtype=torch.float32)
    logits, value = net(x)
    assert logits.shape == (3, 5)
    assert value.shape == (3, 1)


def test_networks_are_deterministic_with_fixed_seed() -> None:
    torch.manual_seed(0)
    a = DQNNetwork(in_channels=4, n_actions=3)
    torch.manual_seed(0)
    b = DQNNetwork(in_channels=4, n_actions=3)
    for pa, pb in zip(a.parameters(), b.parameters()):
        assert torch.equal(pa, pb)
