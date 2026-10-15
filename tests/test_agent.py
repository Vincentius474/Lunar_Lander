import numpy as np
from lunar_lander.agent import DQNAgent

def test_select_action_in_range():
    agent = DQNAgent(obs_dim=9, n_actions=6, hidden=16, seed=0)
    obs = np.zeros(9, dtype=np.float32)
    for _ in range(50):
        a = agent.select_action(obs)
        assert 0 <= a < 6

def test_greedy_ignores_epsilon():
    agent = DQNAgent(obs_dim=9, n_actions=6, hidden=16, seed=0)
    agent.epsilon = 1.0         # force random for non-greedy
    obs = np.zeros(9, dtype=np.float32)

    # Greedy should still return the argmax of the network, deterministically.
    a1 = agent.select_action(obs, greedy=True)
    a2 = agent.select_action(obs, greedy=True)
    assert a1 == a2

def test_update_changes_weights_and_returns_finite_loss():
    agent = DQNAgent(obs_dim=9, n_actions=6, hidden=16, seed=0)
    before = [p.detach().clone() for p in agent.online.parameters()]
    batch = (
        np.random.randn(8, 9).astype(np.float32),
        np.random.randint(0, 6, size=8),
        np.random.randn(8).astype(np.float32),
        np.random.randn(8, 9).astype(np.float32),
        np.zeros(8, dtype=np.float32),
    )

    loss = agent.update(batch)
    assert np.isfinite(loss)
    after = list(agent.online.parameters())
    assert any(not (b == a).all() for b, a in zip(before, after))


def test_target_syncs_after_target_update_steps():
    agent = DQNAgent(obs_dim=9, n_actions=6, hidden=16, seed=0, target_update=5)

    # Manually nudge online weights so target mismatch is visible.
    with __import__("torch").no_grad():
        for p in agent.online.parameters():
            p.add_(0.1)

    # Before syncing, target should differ from online.
    online_before = [p.clone() for p in agent.online.parameters()]
    target_before = [p.clone() for p in agent.target.parameters()]
    assert any(not (o == t).all() for o, t in zip(online_before, target_before))

    # Run 5 updates — target should refresh at step 5.
    batch = (
        np.zeros((4, 9), dtype=np.float32),
        np.zeros(4, dtype=np.int64),
        np.zeros(4, dtype=np.float32),
        np.zeros((4, 9), dtype=np.float32),
        np.zeros(4, dtype=np.float32),
    )
    
    for _ in range(5):
        agent.update(batch)

    online_after = list(agent.online.parameters())
    target_after = list(agent.target.parameters())
    assert all((o == t).all() for o, t in zip(online_after, target_after))