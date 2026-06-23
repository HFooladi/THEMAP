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
