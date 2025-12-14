import pandas as pd
import pytest
from dpu_utils.utils import RichPath

from themap.data.protein_datasets import ProteinMetadataDataset


@pytest.fixture
def manual_smiles():
    smiles = "C1=CC=CC=C1"
    return smiles


@pytest.fixture(scope="module")
def datapoint_molecule():
    """Legacy fixture - returns a dictionary with molecule data."""
    return {
        "task_id": "task_id",
        "smiles": "c1ccccc1",
        "bool_label": True,
        "numeric_label": 1.0,
    }


@pytest.fixture(scope="module")
def manual_smiles_list():
    return [
        "CCCCC",
        "C1=CC=CC=C1",
        "CCCCOC(C1=CC=CC=C1)OCCCC",
        "CC1=CC(=CC(=C1O)C)C(=O)C",
        "CCN(CC)S(=O)(=O)C1=CC=C(C=C1)C(=O)OCC",
        "C[Si](C)(C)CC1=CC=CC=C1",
        "CN1C=NC2=C1C(=O)NC(=O)N2C",
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    ]


@pytest.fixture
def dataset_CHEMBL2219236():
    return RichPath.create("datasets/test/CHEMBL2219236.jsonl.gz")


@pytest.fixture
def dataset_CHEMBL1963831():
    return RichPath.create("datasets/test/CHEMBL1963831.jsonl.gz")


@pytest.fixture
def dataset_CHEMBL1023359():
    return RichPath.create("datasets/test/CHEMBL1023359.jsonl.gz")


@pytest.fixture
def dataset_CHEMBL2219358():
    return RichPath.create("datasets/test/CHEMBL2219358.jsonl.gz")


@pytest.fixture
def dataset_CHEMBL1963831_csv():
    return pd.read_csv("tests/conftest/CHEMBL1963831.csv")


@pytest.fixture
def manual_protein_dataset():
    return ProteinMetadataDataset(
        task_id=["CHEMBL2219236", "CHEMBL2219358"],
        protein={"Q13177": "MSDNGELEDKPPAPPVRMSSTI", "P50750": "MAKQYDSVECPFCDEVSKYEK"},
    )


@pytest.fixture
def protein_dataset_train():
    path = "datasets/train/train_proteins.fasta"
    return path


@pytest.fixture
def protein_dataset_test():
    path = "datasets/test/test_proteins.fasta"
    return path
