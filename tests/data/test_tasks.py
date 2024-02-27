from themap.data.tasks import MoleculeDatapoint, ProteinDatapoint


def test_MoleculeDatapoint():
    # Create a MoleculeDatapoint object
    molecule_datapoint = MoleculeDatapoint(
        task_id="task_id",
        smiles="c1ccccc1",
        numeric_label=1.0,
        bool_label=True,
    )

    # Test the __repr__ method
    assert (
        str(molecule_datapoint)
        == "MoleculeDatapoint(task_id=task_id, smiles=smiles, protein=protein, numeric_label=1.0, bool_label=True)"
    )



def test_ProteinDatapoint():
    # Create a ProteinDatapoint object
    protein_datapoint = ProteinDatapoint(
        task_id="task_id",
        protein="LNMHMNVQNG",
        numeric_label=1.0,
        bool_label=True,
    )

    # Test the __repr__ method
    assert (
        str(protein_datapoint)
        == "ProteinDatapoint(task_id=task_id, protein=protein, numeric_label=1.0, bool_label=True)"
    )

