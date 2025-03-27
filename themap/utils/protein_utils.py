import os
from io import StringIO
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional, Any

import esm
import numpy as np
import pandas as pd
import requests as r
import torch
from Bio import SeqIO
from chembl_webresource_client.new_client import new_client
from tqdm import tqdm

here = Path(__file__).parent
root_dir = here.parent.parent


def get_protein_accession(target_chembl_id: str) -> Optional[str]:
    """Returns the target protein accesion id for a given target chembl id.
    This id can be used to retrieve the protein sequence from the UniProt.

    Args:
        target_chembl_id: Chembl id of the target.

    Returns:
        Optional[str]: The protein accession ID if found, None otherwise.
    """
    target = new_client.target
    target_result = target.get(target_chembl_id)
    if "target_components" in target_result:
        return target_result["target_components"][0]["accession"]
    else:
        return None


def get_target_chembl_id(assay_chembl_id: str) -> Optional[str]:
    """Returns the target chembl id for a given assay chembl id.

    Args:
        assay_chembl_id: Chembl id of the assay.

    Returns:
        Optional[str]: The target ChEMBL ID if found, None otherwise.
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

    Returns:
        List[SeqIO.SeqRecord]: List of sequence records from UniProt.
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
    fs_mol_dataset_path: str, 
    esm2_model: str = "esm2_t33_650M_UR50D", 
    layer: int = 33
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[Any], List[Any], List[Any]]:
    """Reads the ESM embedding from a given path.

    Args:
        fs_mol_dataset_path: Path to the FS_MOL Dataset.
        esm2_model: Name of the ESM2 model.
        layer: Layer of the ESM2 model to be used.

    Returns:
        Tuple containing:
        - train_emb_tensor: Training embeddings tensor
        - valid_emb_tensor: Validation embeddings tensor
        - test_emb_tensor: Test embeddings tensor
        - train_emb_label: Training labels
        - valid_emb_label: Validation labels
        - test_emb_label: Test labels
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


def convert_fasta_to_dict(fasta_file: str) -> Dict[str, str]:
    """Converts a fasta file to a dictionary.

    Args:
        fasta_file: Path to the fasta file.
    
    Returns:
        Dict[str, str]: Dictionary containing the fasta sequences.
        {'id': 'sequence'}
    """
    fasta_dict = {}
    for record in SeqIO.parse(fasta_file, "fasta"):
        fasta_dict[record.id] = str(record.seq)
    return fasta_dict


def get_task_name_from_uniprot(
    uniprot_id: List[str], 
    df_path: str = f"{root_dir}/datasets/uniprot_mapping.csv"
) -> List[str]:
    """Returns the task id from the list of uniprot_ids

    Args:
        uniprot_id: List of uniprot ids.
        df_path: Path to the uniprot mapping file.

    Returns:
        List[str]: List of task IDs.
    """
    df = pd.read_csv(df_path)
    task_id = []
    for id in uniprot_id:
        task_id.append(df[df["target_accession_id"] == id]["chembl_id"].values[0])
    return task_id


def get_protein_features(
    protein_dict: Dict[str, str], 
    featurizer: str = "esm2_t33_650M_UR50D", 
    layer: int = 33
) -> np.ndarray:
    """Returns a featurizer object based on the input string.

    Args:
        protein_dict: Dictionary containing the protein sequences.
        featurizer: String specifying the featurizer to use.
        layer: Layer of the ESM2 model to be used.

    Returns:
        np.ndarray: Array containing the protein features.
    """
    featurizer_dict = {"esm2_t33_650M_UR50D": esm.pretrained.esm2_t33_650M_UR50D()}
    # Load ESM-2 model
    model, alphabet = featurizer_dict[featurizer]
    batch_converter = alphabet.get_batch_converter()
    model.eval()  # disables dropout for deterministic results

    # Prepare data (first 2 sequences from ESMStructuralSplitDataset superfamily / 4)
    data = list(protein_dict.items())
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

    # Extract per-residue representations (on CPU)
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33], return_contacts=True)
    token_representations = results["representations"][33]

    sequence_representations = []
    for i, tokens_len in enumerate(batch_lens):
        sequence_representations.append(token_representations[i, 1 : tokens_len - 1].mean(0))

    return torch.stack(sequence_representations).numpy()
