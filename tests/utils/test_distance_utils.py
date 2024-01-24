import pytest
import numpy as np

from themap.data.tasks import MoleculeDatapoint
from themap.utils.distance_utils import compute_fp_similarity, compute_fp_similarities, normalize

@pytest.fixture
def first_mol():
    return MoleculeDatapoint(smiles="CCO", label=0.5, bool_label=True)

@pytest.fixture
def second_mol():
    return MoleculeDatapoint(smiles="CCO", label=0.5, bool_label=True)

def test_normalize():
    x = np.array([1, 2, 3, 4, 5])
    assert np.allclose(normalize(x), np.array([0, 0.25, 0.5, 0.75, 1]))

def test_compute_fp_similarity():
    pass



def test_compute_fp_similarities():
    pass