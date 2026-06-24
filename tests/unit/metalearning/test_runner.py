"""Integration test for the end-to-end experiment runner (mocked data)."""

import json

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from themap.metalearning import runner as runner_mod  # noqa: E402
from themap.metalearning.config import ExperimentConfig, TrainConfig  # noqa: E402
from themap.metalearning.episodes import FeatureBank, TaskFeatures  # noqa: E402
from themap.metalearning.runner import MetaLearnExperiment  # noqa: E402


def _bank(task_ids, dim=32, seed=0):
    rng = np.random.default_rng(seed)
    tasks = {}
    for i, tid in enumerate(task_ids):
        pos = rng.normal(2.0 + i, 0.5, (50, dim)).astype(np.float32)
        neg = rng.normal(-2.0 - i, 0.5, (50, dim)).astype(np.float32)
        X = np.vstack([pos, neg])
        y = np.array([1] * 50 + [0] * 50)
        tasks[tid] = TaskFeatures.from_arrays(tid, X, y)
    return FeatureBank(tasks)


@pytest.mark.integration
@pytest.mark.parametrize("algorithm", ["proto", "maml"])
def test_runner_end_to_end(tmp_path, monkeypatch, algorithm):
    source_ids = ["S1", "S2", "S3"]
    target_id = "T"

    # Stub distance selection (avoid needing a real distance file content match).
    monkeypatch.setattr(
        runner_mod,
        "select_k_nearest_sources",
        lambda path, tid, k: [(s, 0.1 * (i + 1)) for i, s in enumerate(source_ids)],
    )
    # Stub dataset loading (return dummies; featurization is also stubbed).
    monkeypatch.setattr(
        MetaLearnExperiment,
        "_load_datasets",
        lambda self, sids: ({s: object() for s in sids}, object()),
    )
    # Stub featurization with a prebuilt FeatureBank.
    bank = _bank(source_ids + [target_id])
    monkeypatch.setattr(FeatureBank, "from_datasets", classmethod(lambda cls, *a, **k: bank))

    dist_file = tmp_path / "d.json"
    dist_file.write_text(json.dumps({target_id: {s: 1.0 for s in source_ids}}))

    config = ExperimentConfig(
        data_dir=str(tmp_path),
        distance_file=str(dist_file),
        target_id=target_id,
        k=3,
        algorithm=algorithm,
        support_sizes=[16],
        seeds=2,
        output_dir=str(tmp_path / "out"),
        train=TrainConfig(
            num_epochs=1,
            episodes_per_epoch=3,
            meta_batch_size=2,
            n_support=6,
            n_query=8,
            val_episodes=2,
            patience=0,
            device="cpu",
        ),
    )

    results = MetaLearnExperiment(config).run()
    assert isinstance(results, pd.DataFrame)
    assert not results.empty
    assert (tmp_path / "out" / "results.csv").exists()
    assert (tmp_path / "out" / "summary.csv").exists()
    assert (tmp_path / "out" / "selected_sources.json").exists()


@pytest.mark.integration
def test_fixed_mode_trains_single_model(tmp_path, monkeypatch):
    source_ids = ["S1", "S2", "S3"]
    target_id = "T"
    monkeypatch.setattr(
        runner_mod,
        "select_k_nearest_sources",
        lambda path, tid, k: [(s, 0.1 * (i + 1)) for i, s in enumerate(source_ids)],
    )
    monkeypatch.setattr(
        MetaLearnExperiment,
        "_load_datasets",
        lambda self, sids: ({s: object() for s in sids}, object()),
    )
    bank = _bank(source_ids + [target_id])
    monkeypatch.setattr(FeatureBank, "from_datasets", classmethod(lambda cls, *a, **k: bank))

    # Count how many times a model is meta-trained.
    calls = {"n": 0}
    orig = MetaLearnExperiment._meta_train

    def counting_meta_train(self, *a, **k):
        calls["n"] += 1
        return orig(self, *a, **k)

    monkeypatch.setattr(MetaLearnExperiment, "_meta_train", counting_meta_train)

    dist_file = tmp_path / "d.json"
    dist_file.write_text(json.dumps({target_id: {s: 1.0 for s in source_ids}}))

    config = ExperimentConfig(
        data_dir=str(tmp_path),
        distance_file=str(dist_file),
        target_id=target_id,
        k=3,
        algorithm="proto",
        support_sizes=[16, 32],
        train_shot_mode="fixed",
        seeds=2,
        output_dir=None,
        train=TrainConfig(
            num_epochs=1,
            episodes_per_epoch=2,
            meta_batch_size=2,
            n_support=16,
            n_query=8,
            val_episodes=0,
            patience=0,
            device="cpu",
        ),
    )

    results = MetaLearnExperiment(config).run()
    # FS-Mol single-model: one training run, but every support size evaluated.
    assert calls["n"] == 1
    assert set(results["support_size"]) == {16, 32}


@pytest.mark.unit
def test_train_shot_capping():
    # 12 pos / 12 neg: feasible per class = 12 - (n_query//2=4) = 8 -> 16 total.
    rng = np.random.default_rng(0)
    X = rng.normal(size=(24, 4)).astype(np.float32)
    y = np.array([1] * 12 + [0] * 12)
    tasks = [TaskFeatures.from_arrays("s", X, y)]

    base = ExperimentConfig(
        data_dir="x", distance_file="d", target_id="t", train=TrainConfig(n_query=8, n_support=6)
    )
    exp = MetaLearnExperiment(base)
    # "match": tracks N but is capped to the feasible 16.
    assert exp._train_shot_for_n(8, tasks, base) == 8
    assert exp._train_shot_for_n(128, tasks, base) == 16
    # "fixed": always the configured n_support.
    fixed = ExperimentConfig(
        data_dir="x",
        distance_file="d",
        target_id="t",
        train_shot_mode="fixed",
        train=TrainConfig(n_query=8, n_support=6),
    )
    assert MetaLearnExperiment(fixed)._train_shot_for_n(128, tasks, fixed) == 6
