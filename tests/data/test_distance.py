from themap.data.distance import MoleculeDatasetDistance, ProteinDatasetDistance
import pytest

def test_molecule_distance():
    pass

@pytest.mark.slow
def test_protein_distance(manual_protein_dataset):
    _ = manual_protein_dataset.get_features('esm2_t33_650M_UR50D')
    Distance = ProteinDatasetDistance(D1=manual_protein_dataset, D2=manual_protein_dataset, method="euclidean")
    d = Distance.get_distance()
    d = Distance.to_pandas()

    assert d.shape == (2, 2)
    assert d.iloc[0, 0] == 0
    assert d.iloc[1, 1] == 0
    assert d.iloc[0, 0] <= d.iloc[0, 1]
    assert d.iloc[1, 1] <= d.iloc[1, 0]



