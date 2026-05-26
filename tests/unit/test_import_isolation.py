"""Regression test for THEMAP subsystem import isolation.

Asserts that importing a public API surface does not transitively load
modules from unrelated subsystems. Each probe runs in a fresh Python
subprocess so ``sys.modules`` starts clean and previous imports in the
test session can't mask a leak.

The forbidden sets encode the contract documented in the module isolation
plan: e.g. the OTDD path may load torch/pot/geomloss but must not load
biopython; the molecule featurizer path may load molfeat but must not
load biopython.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import textwrap

import pytest

# Module names whose presence in ``sys.modules`` indicates a forbidden
# subsystem leaked into the probed import path.
FORBIDDEN = {
    "bare_themap": [
        "torch",
        "molfeat",
        "Bio",
        "esm",
        "ot",
        "geomloss",
        "pykeops",
        "munkres",
        "adjustText",
        "transformers",
        "dgl",
    ],
    "molecule_only": [
        "Bio",
        "esm",
        "sentence_transformers",
        "ot",
        "geomloss",
        "pykeops",
        "munkres",
        "adjustText",
    ],
    "molecule_featurizer": [
        "Bio",
        "esm",
        "sentence_transformers",
        "ot",
        "geomloss",
        "pykeops",
        "munkres",
        "adjustText",
    ],
    "protein_featurizer": [
        "molfeat",
        "ot",
        "geomloss",
        "pykeops",
        "munkres",
        "adjustText",
    ],
    "otdd_distance": [
        "Bio",
        "esm",
        "sentence_transformers",
        "adjustText",
    ],
}

# Python source to execute in a fresh subprocess for each probe.
PROBES = {
    "bare_themap": "import themap",
    "molecule_only": "from themap import MoleculeDataset; from themap.data.loader import DatasetLoader",
    "molecule_featurizer": "from themap.features.molecule import MoleculeFeaturizer",
    "protein_featurizer": "from themap.features.protein import ProteinFeaturizer",
    "otdd_distance": "from themap.models.otdd.src.distance import DatasetDistance",
}

# Optional deps required for each probe to be meaningful. If absent, skip.
PROBE_REQUIRES = {
    "protein_featurizer": ["Bio"],
    "otdd_distance": ["geomloss", "ot"],
}


def _have(modname: str) -> bool:
    """Return True if ``modname`` is importable in the current environment."""
    return importlib.util.find_spec(modname) is not None


@pytest.mark.unit
@pytest.mark.parametrize("name", list(PROBES))
def test_subsystem_import_isolation(name: str) -> None:
    """Each probe must not pull in any forbidden top-level module."""
    for required in PROBE_REQUIRES.get(name, []):
        if not _have(required):
            pytest.skip(f"{name}: requires optional dep '{required}' to be meaningful")

    code = textwrap.dedent(
        f"""
        {PROBES[name]}
        import sys
        print("\\n".join(sorted(sys.modules)))
        """
    ).strip()

    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        pytest.fail(f"probe '{name}' failed to import\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}")

    loaded_top_level = {m.split(".")[0] for m in result.stdout.splitlines()}
    leaks = loaded_top_level & set(FORBIDDEN[name])
    assert not leaks, (
        f"{name} leaked forbidden modules: {sorted(leaks)}.\n"
        f"Probe: {PROBES[name]!r}\n"
        f"Loaded top-level modules: {sorted(loaded_top_level)}"
    )
