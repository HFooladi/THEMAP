import os
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

import numpy as np
import pandas as pd
import requests as r
import torch
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
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
        accession = target_result["target_components"][0]["accession"]
        return str(accession) if accession is not None else None
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
        return str(target_chembl_id) if target_chembl_id is not None else None
    return None


def get_protein_sequence(protein_accession: str) -> List[SeqRecord]:
    """Returns the protein sequence for a given protein accession id.

    Args:
        protein_accession: Accession id of the protein.

    Returns:
        List[SeqRecord]: List of sequence records from UniProt.
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
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[str], List[str], List[str]]:
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

    train_emb: List[Dict[str, Any]] = []
    valid_emb: List[Dict[str, Any]] = []
    test_emb: List[Dict[str, Any]] = []

    train_emb_label: List[str] = []
    valid_emb_label: List[str] = []
    test_emb_label: List[str] = []

    train_emb_tensor = torch.empty(0)
    valid_emb_tensor = torch.empty(0)
    test_emb_tensor = torch.empty(0)

    layer = layer

    for file in tqdm(train_files):
        train_emb.append(cast(Dict[str, Any], torch.load(file)))
        train_emb_tensor = torch.cat(
            (train_emb_tensor, train_emb[-1]["mean_representations"][layer][None, :]), 0
        )
        train_emb_label.append(cast(str, train_emb[-1]["label"]))

    for file in tqdm(valid_files):
        valid_emb.append(cast(Dict[str, Any], torch.load(file)))
        valid_emb_tensor = torch.cat(
            (valid_emb_tensor, valid_emb[-1]["mean_representations"][layer][None, :]), 0
        )
        valid_emb_label.append(cast(str, valid_emb[-1]["label"]))

    for file in tqdm(test_files):
        test_emb.append(cast(Dict[str, Any], torch.load(file)))
        test_emb_tensor = torch.cat(
            (test_emb_tensor, test_emb[-1]["mean_representations"][layer][None, :]), 0
        )
        test_emb_label.append(cast(str, test_emb[-1]["label"]))

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
    uniprot_id: List[str], df_path: str = f"{root_dir}/datasets/uniprot_mapping.csv"
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
        task_id.append(str(df[df["target_accession_id"] == id]["chembl_id"].values[0]))
    return task_id


def get_protein_features(
    protein_dict: Dict[str, str], featurizer: str = "esm2_t33_650M_UR50D", layer: Optional[int] = None
) -> np.ndarray:
    """Computes protein sequence embeddings using a pre-trained language model.

    This function takes a dictionary of protein sequences and computes fixed-length embeddings
    using a pre-trained ESM-2 protein language model. It averages the per-residue embeddings
    to get a single vector representation for each sequence.

    Args:
        protein_dict: Dictionary mapping protein IDs to their amino acid sequences.
        featurizer: Name of the pre-trained model to use. Currently only supports ESM2.
        layer: Which transformer layer to extract embeddings from. If None, uses final embeddings.

    Returns:
        np.ndarray: Array of shape (num_proteins, embedding_dim) containing the computed
            protein embeddings.

    Note:
        - For using this function, you need to install the fair-esm package and import esm from this package.
        - If layer is None, uses final embeddings.
        - It downloads the model from the internet (to your local .cache), so it may take a while to load.
    """

    # Available ESM2 models
    available_models = ["esm2_t12_35M_UR50D", "esm2_t33_650M_UR50D"]

    if featurizer not in available_models:
        raise ValueError(f"Unsupported ESM2 featurizer: {featurizer}. Available models: {available_models}")

    # Load ESM-2 model and prepare batch converter
    model, alphabet = torch.hub.load("facebookresearch/esm", featurizer)
    batch_converter = alphabet.get_batch_converter()
    model.eval()  # disables dropout for deterministic results

    # Convert protein dictionary to list of (id, sequence) tuples
    data = list(protein_dict.items())
    batch_labels, batch_strs, batch_tokens = batch_converter(data)

    # Calculate sequence lengths excluding padding tokens
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

    if layer is None:
        layer = model.cfg.layers

    # Extract embeddings from specified layer
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[layer], return_contacts=True)
    token_representations = results["representations"][layer]

    # Average the per-residue embeddings for each sequence
    # Skip the special start/end tokens using slice 1:-1
    sequence_representations = []
    for i, tokens_len in enumerate(batch_lens):
        sequence_representations.append(token_representations[i, 1 : tokens_len - 1].mean(0))

    return torch.stack(sequence_representations).numpy()


def get_protein_features_esm3(
    protein_dict: Dict[str, str], featurizer: str = "esm3_sm_open_v1", layer: Optional[int] = None
) -> np.ndarray:
    """Computes protein sequence embeddings using ESM3 models.

    This function takes a dictionary of protein sequences and computes fixed-length embeddings
    using a pre-trained ESM3 protein language model. It averages the per-residue embeddings
    to get a single vector representation for each sequence.

    Args:
        protein_dict: Dictionary mapping protein IDs to their amino acid sequences.
        featurizer: Name of the ESM3 model to use. Supported models:
            - "esm3_sm_open_v1": ESM3 small model (open source)
            - "esm3_open_small": ESM3 small model (alternative name)
        layer: Which transformer layer to extract embeddings from. If None, uses final embeddings.

    Returns:
        np.ndarray: Array of shape (num_proteins, embedding_dim) containing the computed
            protein embeddings.

    Raises:
        ValueError: If the specified featurizer is not supported.
        ImportError: If ESM3 is not properly installed.
        RuntimeError: If protein processing fails.

    Note: For using this function, you need to install the esm package and import esm from this package.
    """
    try:
        from esm.models.esm3 import ESM3
        from esm.sdk.api import ESM3InferenceClient, ESMProtein, LogitsConfig
    except ImportError as e:
        raise ImportError(
            "ESM3 not found. Please install the esm package with ESM3 support: pip install esm"
        ) from e

    # Available ESM3 models
    available_models = ["esm3_sm_open_v1", "esm3_open_small"]

    if featurizer not in available_models:
        raise ValueError(f"Unsupported ESM3 featurizer: {featurizer}. Available models: {available_models}")

    # Initialize ESM3 inference client
    model_name = featurizer
    try:
        client: ESM3InferenceClient = ESM3.from_pretrained(model_name).to("cuda")
    except Exception as e:
        raise RuntimeError(f"Failed to initialize ESM3 model {model_name}: {e}") from e

    sequence_representations = []

    # Process each protein sequence
    for protein_id, sequence in protein_dict.items():
        if not sequence or not isinstance(sequence, str):
            raise ValueError(f"Invalid sequence for protein {protein_id}: {sequence}")

        try:
            # Create ESMProtein object from sequence
            protein = ESMProtein(sequence=sequence)

            # Encode the protein to get internal representation
            protein_tensor = client.encode(protein)

            # Configure to get sequence embeddings
            logits_config = LogitsConfig(sequence=True)

            # Get embeddings from the model
            embeddings = client.logits(protein_tensor, logits_config)

            # Extract sequence embeddings
            if hasattr(embeddings, "sequence_logits") and embeddings.sequence_logits is not None:
                # Use sequence logits if available
                sequence_logits = embeddings.sequence_logits
                # Average over sequence dimension to get a single vector per protein
                protein_embedding = sequence_logits.mean(dim=1).squeeze()

                # Convert to numpy if it's a torch tensor
                if hasattr(protein_embedding, "numpy"):
                    protein_embedding = protein_embedding.numpy()
                elif hasattr(protein_embedding, "detach"):
                    protein_embedding = protein_embedding.detach().numpy()

                sequence_representations.append(protein_embedding)
            else:
                # Fallback: try to get hidden states from the encoded tensor
                if hasattr(protein_tensor, "sequence") and protein_tensor.sequence is not None:
                    hidden_states = protein_tensor.sequence
                    # Average over sequence dimension
                    protein_embedding = hidden_states.mean(dim=1).squeeze()

                    # Convert to numpy if it's a torch tensor
                    if hasattr(protein_embedding, "numpy"):
                        protein_embedding = protein_embedding.numpy()
                    elif hasattr(protein_embedding, "detach"):
                        protein_embedding = protein_embedding.detach().numpy()

                    sequence_representations.append(protein_embedding)
                else:
                    raise ValueError(f"Could not extract embeddings for protein {protein_id}")

        except Exception as e:
            raise RuntimeError(
                f"Error processing protein {protein_id} with sequence '{sequence[:50]}...': {e}"
            ) from e

    if not sequence_representations:
        raise ValueError("No protein embeddings were successfully computed")

    # Stack all embeddings into a single array
    try:
        return np.stack(sequence_representations)
    except Exception as e:
        raise RuntimeError(f"Failed to stack protein embeddings: {e}") from e
