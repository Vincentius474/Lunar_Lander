import torch
from lunar_lander.network import QNetwork

def test_network_output_shape():
    net = QNetwork(obs_dim=9, n_actions=6, hidden=32)
    x = torch.zeros(4, 9)               # batch of 4
    out = net(x)
    assert out.shape == (4, 6)

def test_network_gradients_flow():
    net = QNetwork(obs_dim=9, n_actions=6, hidden=32)
    x = torch.randn(8, 9)
    out = net(x).sum()
    out.backward()

    # Every parameter should have a gradient.
    for name, p in net.named_parameters():
        assert p.grad is not None, f"no gradient for {name}"
        assert torch.isfinite(p.grad).all(), f"non-finite grad in {name}"