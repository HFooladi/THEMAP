"""CLI smoke test and torch-free import-isolation guarantees."""

import subprocess
import sys

import pytest


@pytest.mark.unit
def test_metalearn_help_no_torch_import():
    """`themap metalearn --help` must succeed without importing torch."""
    code = (
        "import sys; from click.testing import CliRunner; from themap.cli import cli; "
        "r = CliRunner().invoke(cli, ['metalearn', '--help']); "
        "assert r.exit_code == 0, r.output; "
        "assert 'distance-file' in r.output; "
        "assert 'torch' not in sys.modules, 'torch was imported by --help'"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.unit
def test_selection_and_config_are_torch_free():
    """Importing the lightweight utilities must not pull in torch."""
    code = (
        "import sys; "
        "import themap.metalearning.selection; "
        "import themap.metalearning.config; "
        "from themap.metalearning import select_k_nearest_sources, ExperimentConfig; "
        "assert 'torch' not in sys.modules, 'torch imported by torch-free modules'"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
