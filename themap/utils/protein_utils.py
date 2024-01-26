import os
from io import StringIO
from pathlib import Path
from typing import List, Tuple, Union

import requests as r
import torch
from Bio import SeqIO
from chembl_webresource_client.new_client import new_client
from tqdm import tqdm


def get_protein_accession(target_chembl_id: str) -> Union[str, None]:
    """Returns the target protein accesion id for a given target chembl id.
    This id can be used to retrieve the protein sequence from the UniProt.

    Args:
        target_chembl_id: Chembl id of the target.
    """
    target = new_client.target
    target_result = target.get(target_chembl_id)
    if "target_components" in target_result:
        return target_result["target_components"][0]["accession"]
    else:
        return None


def get_target_chembl_id(assay_chembl_id: str) -> Union[str, None]:
    """Returns the target chembl id for a given assay chembl id.

    Args:
        assay_chembl_id: Chembl id of the assay.
    """
    assay = new_client.assay
    assay_result = assay.get(assay_chembl_id)
    if "target_chembl_id" in assay_result:
        target_chembl_id = assay_result["target_chembl_id"]
        return target_chembl_id
    return None


def get_protein_sequence(protein_accession: str) -> List[SeqIO.SeqRecord]:
    """Returns the protein sequence for a given protein accession id.

    Args:
        protein_accession: Accession id of the protein.
    """
    cID = protein_accession
    baseUrl = "http://www.uniprot.org/uniprot/"
    currentUrl = baseUrl + cID + ".fasta"
    response = r.post(currentUrl)
    cData = "".join(response.text)

    Seq = StringIO(cData)
    pSeq = list(SeqIO.parse(Seq, "fasta"))
    return pSeq


def read_esm_embedding(
    fs_mol_dataset_path: str, esm2_model: str = "esm2_t33_650M_UR50D", layer: int = 33
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List, List, List]:
    """Reads the ESM embedding from a given path.

    Args:
        fs_mol_dataset_path: Path to the FS_MOL Dataset.
        esm2_model: Name of the ESM2 model.
        layer: Layer of the ESM2 model to be used.
    """
    ESM_EMBEDDING_PATH = os.path.join(fs_mol_dataset_path, "targets", "esm2_output", esm2_model)

    train_esm = os.path.join(ESM_EMBEDDING_PATH, "train_proteins")
    valid_esm = os.path.join(ESM_EMBEDDING_PATH, "valid_proteins")
    test_esm = os.path.join(ESM_EMBEDDING_PATH, "test_proteins")

    train_files = Path(train_esm).glob("*")
    valid_files = Path(valid_esm).glob("*")
    test_files = Path(test_esm).glob("*")

    train_emb = []
    valid_emb = []
    test_emb = []

    train_emb_label = []
    valid_emb_label = []
    test_emb_label = []

    train_emb_tensor = torch.empty(0)
    valid_emb_tensor = torch.empty(0)
    test_emb_tensor = torch.empty(0)

    layer = layer

    for file in tqdm(train_files):
        train_emb.append(torch.load(file))
        train_emb_tensor = torch.cat(
            (train_emb_tensor, train_emb[-1]["mean_representations"][layer][None, :]), 0
        )
        train_emb_label.append(train_emb[-1]["label"])

    for file in tqdm(valid_files):
        valid_emb.append(torch.load(file))
        valid_emb_tensor = torch.cat(
            (valid_emb_tensor, valid_emb[-1]["mean_representations"][layer][None, :]), 0
        )
        valid_emb_label.append(valid_emb[-1]["label"])

    for file in tqdm(test_files):
        test_emb.append(torch.load(file))
        test_emb_tensor = torch.cat(
            (test_emb_tensor, test_emb[-1]["mean_representations"][layer][None, :]), 0
        )
        test_emb_label.append(test_emb[-1]["label"])

    return (
        train_emb_tensor,
        valid_emb_tensor,
        test_emb_tensor,
        train_emb_label,
        valid_emb_label,
        test_emb_label,
    )
