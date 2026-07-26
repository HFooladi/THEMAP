"""Tests for deterministic selection of the FS-Mol benchmark task subset."""

import gzip
import json

import numpy as np
import pandas as pd
import pytest

from themap.metalearning.subset import (
    SubsetSpec,
    load_subset,
    save_subset,
    select_benchmark_subset,
    subset_representativeness,
    task_label_counts,
)

_EC_CLASSES = ["transferase"] * 30 + ["hydrolase"] * 8 + ["lyase"] * 2


def _build_fixture(tmp_path, n_tasks=40):
    """Write a miniature FS-Mol-shaped dataset: task files, protein CSV, reference table."""
    test_dir = tmp_path / "test"
    test_dir.mkdir(parents=True)
    rng = np.random.default_rng(0)

    task_ids, sizes = [], []
    for i in range(n_tasks):
        task_id = f"CHEMBL{1000 + i}"
        # Sizes span 100..800 so the size terciles are meaningful.
        n = 100 + (i * 17) % 700
        n_pos = n // 2
        with gzip.open(test_dir / f"{task_id}.jsonl.gz", "wt") as handle:
            for j in range(n):
                label = 1.0 if j < n_pos else 0.0
                handle.write(json.dumps({"SMILES": "C", "Property": f"{label}"}) + "\n")
        task_ids.append(task_id)
        sizes.append(n)

    proteins = pd.DataFrame(
        {
            "chembl_id": task_ids,
            "EC_super_class_name": [_EC_CLASSES[i % len(_EC_CLASSES)] for i in range(n_tasks)],
        }
    )
    proteins_csv = tmp_path / "proteins.csv"
    proteins.to_csv(proteins_csv, index=False)

    records = []
    for method in ("PN", "RF"):
        offset = 0.0 if method == "PN" else -0.08
        for i, task_id in enumerate(task_ids):
            for size in (16, 32, 64, 128, 256):
                base = 0.05 + 0.4 * rng.random()
                records.append(
                    {
                        "method": method,
                        "task_id": task_id,
                        "support_size": size,
                        "auprc": np.nan,
                        "auprc_std": np.nan,
                        "fraction_positive": 0.5,
                        "delta_auprc": base + offset + 0.02 * (size == 256),
                    }
                )
    reference = pd.DataFrame.from_records(records)
    return tmp_path, reference, proteins_csv, task_ids, sizes


@pytest.mark.unit
class TestTaskLabelCounts:
    def test_counts_match_the_files(self, tmp_path):
        root, _, _, task_ids, sizes = _build_fixture(tmp_path, n_tasks=5)
        counts = task_label_counts(root, "test").set_index("task_id")
        for task_id, size in zip(task_ids, sizes):
            assert counts.loc[task_id, "n"] == size
            assert counts.loc[task_id, "n_pos"] == size // 2
            assert counts.loc[task_id, "frac_pos"] == pytest.approx((size // 2) / size)

    def test_missing_fold_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            task_label_counts(tmp_path, "nope")


@pytest.mark.unit
class TestSelection:
    def test_is_deterministic(self, tmp_path):
        root, reference, proteins, _, _ = _build_fixture(tmp_path)
        spec = SubsetSpec(n_tasks=12, min_large=3)
        first = select_benchmark_subset(root, reference, proteins, spec)
        second = select_benchmark_subset(root, reference, proteins, spec)
        assert first["task_id"].tolist() == second["task_id"].tolist()

    def test_selects_the_requested_number(self, tmp_path):
        root, reference, proteins, _, _ = _build_fixture(tmp_path)
        selected = select_benchmark_subset(root, reference, proteins, SubsetSpec(n_tasks=12, min_large=3))
        assert len(selected) == 12

    def test_every_pick_is_in_the_universe(self, tmp_path):
        root, reference, proteins, task_ids, _ = _build_fixture(tmp_path)
        selected = select_benchmark_subset(root, reference, proteins, SubsetSpec(n_tasks=12, min_large=3))
        assert set(selected["task_id"]) <= set(task_ids)

    def test_dominant_ec_class_is_represented(self, tmp_path):
        root, reference, proteins, _, _ = _build_fixture(tmp_path)
        selected = select_benchmark_subset(root, reference, proteins, SubsetSpec(n_tasks=12, min_large=3))
        assert "transferase" in set(selected["ec_class"])
        # Rare classes fold into "other" rather than each claiming a slot.
        assert "lyase" not in set(selected["ec_class"])

    def test_large_task_quota_is_honoured(self, tmp_path):
        root, reference, proteins, _, _ = _build_fixture(tmp_path)
        spec = SubsetSpec(n_tasks=12, min_large=5, large_task_threshold=400)
        selected = select_benchmark_subset(root, reference, proteins, spec)
        assert int((selected["n"] > 400).sum()) >= 5

    def test_tasks_without_reference_scores_are_excluded(self, tmp_path):
        root, reference, proteins, task_ids, _ = _build_fixture(tmp_path)
        dropped = task_ids[0]
        reference = reference[~((reference.method == "PN") & (reference.task_id == dropped))]
        selected = select_benchmark_subset(root, reference, proteins, SubsetSpec(n_tasks=12, min_large=3))
        assert dropped not in set(selected["task_id"])

    def test_requesting_more_than_available_raises(self, tmp_path):
        root, reference, proteins, _, _ = _build_fixture(tmp_path, n_tasks=5)
        with pytest.raises(ValueError, match="candidate task"):
            select_benchmark_subset(root, reference, proteins, SubsetSpec(n_tasks=50))

    def test_precomputed_counts_give_the_same_answer(self, tmp_path):
        root, reference, proteins, _, _ = _build_fixture(tmp_path)
        spec = SubsetSpec(n_tasks=12, min_large=3)
        counts = task_label_counts(root, "test")
        assert (
            select_benchmark_subset(root, reference, proteins, spec, counts=counts)["task_id"].tolist()
            == select_benchmark_subset(root, reference, proteins, spec)["task_id"].tolist()
        )


@pytest.mark.unit
class TestRepresentativenessAndIO:
    def test_representativeness_table_shape(self, tmp_path):
        root, reference, proteins, _, _ = _build_fixture(tmp_path)
        selected = select_benchmark_subset(root, reference, proteins, SubsetSpec(n_tasks=12, min_large=3))
        rep = subset_representativeness(reference, selected["task_id"].tolist())
        assert {"method", "support_size", "subset_mean", "full_mean", "abs_delta"} <= set(rep.columns)
        assert len(rep) == 2 * 5  # two methods x five support sizes
        assert (rep["n_subset"] <= rep["n_full"]).all()

    def test_save_and_load_roundtrip(self, tmp_path):
        root, reference, proteins, _, _ = _build_fixture(tmp_path)
        spec = SubsetSpec(n_tasks=12, min_large=3)
        selected = select_benchmark_subset(root, reference, proteins, spec)
        path = tmp_path / "subset.json"
        save_subset(selected, path, spec, {"reference_sha256": {"PN": "abc"}})
        assert load_subset(path) == sorted(selected["task_id"].tolist())
        payload = json.loads(path.read_text())
        assert payload["spec"]["n_tasks"] == 12
        assert payload["provenance"]["reference_sha256"]["PN"] == "abc"
        assert len(payload["per_task"]) == 12

    def test_load_rejects_empty_file(self, tmp_path):
        path = tmp_path / "empty.json"
        path.write_text(json.dumps({"task_ids": []}))
        with pytest.raises(ValueError, match="No task_ids"):
            load_subset(path)
