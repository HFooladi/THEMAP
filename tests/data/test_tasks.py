from themap.data.tasks import MoleculeDatapoint, ProteinDataset, MoleculeDataset


def test_MoleculeDatapoint(datapoint_molecule):
    # Create a MoleculeDatapoint object

    # Test the __repr__ method
    assert (
        str(datapoint_molecule)
        == "MoleculeDatapoint(task_id=task_id, smiles=c1ccccc1, bool_label=True, numeric_label=1.0)"
    )

    # Test the number_of_atoms method
    assert datapoint_molecule.number_of_atoms == 6

    # Test the number_of_bonds method
    assert datapoint_molecule.number_of_bonds == 6

    # Test the molecular_weight method
    assert round(datapoint_molecule.molecular_weight) == 78


def test_ProteinDataset():
    # Create a ProteinDatapoint object
    protein_dataset = ProteinDataset(
        task_id=["task_id"],
        protein={"task_id": "LNMHMNVQNG"},
    )

    # Test the __repr__ method
    assert (
        str(protein_dataset)
        == "ProteinDataset(task_id=['task_id'], protein={'task_id': 'LNMHMNVQNG'})"
    )


def test_MoleculeDataset_load_from_file(dataset_CHEMBL2219236):
    # Load the dataset from a file
    dataset = MoleculeDataset.load_from_file(dataset_CHEMBL2219236)

    # Test the  __len__ method
    assert len(dataset) == 157

    # Test the __getitem__ method
    assert isinstance(dataset[0], MoleculeDatapoint)

    # Test the __repr__ method
    assert str(dataset) == "MoleculeDataset(task_id=CHEMBL2219236, task_size=157)"


def ProteinDataset_load_from_file(protein_dataset_train):
    # Load the dataset from a file
    dataset = ProteinDataset.load_from_file(protein_dataset_train)

    # Test the  __len__ method
    assert len(dataset) == 10

    # Test the __getitem__ method
    assert isinstance(dataset[0], tuple)
    assert isinstance(dataset[0][0], str)
    assert isinstance(dataset[0][1], str)
