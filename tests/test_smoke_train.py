from lunar_lander.trainer import TrainConfig, train

"""
Smoke test: run a very short training session and verify it terminates.
"""

def test_smoke_train_completes(tmp_path, monkeypatch):
    # Redirect output to a temp dir so the test doesn't pollute the repo.
    monkeypatch.chdir(tmp_path)
    cfg = TrainConfig(
        episodes=3,
        max_steps=50,
        batch_size=8,
        buffer_size=500,
        warmup_steps=20,
        hidden=16,
        target_update=10,
        epsilon_decay=0.9,
        log_every=1,
        save_every=100,
        seed=0,
    )
    history = train(cfg)
    assert len(history["episode"]) == 3
    assert all(isinstance(r, float) for r in history["reward"])
    assert (tmp_path / "results" / "seed_0" / "training_history.json").exists()