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

# Global model cache to avoid reloading models
_MODEL_CACHE: Dict[str, Any] = {}

here = Path(__file__).parent
root_dir = here.parent.parent


def get_protein_accession(target_chembl_id: str) -> Optional[str]:
    """Returns the target protein accesion id for a given target chembl id.
    This id can be used to retrieve the protein sequence from the UniProt.

    Args:
        target_chembl_id: Chembl id of the target.

    Returns:
        Optional[str]: The protein accession ID if found, None otherwise.

    Raises:
        Exception: If there is an error fetching the protein accession ID. For example,
        if the target ID is invalid.
    """
    target = new_client.target
    try:
        target_result: Dict[str, Any] = target.get(target_chembl_id)
        if "target_components" in target_result:
            accession = target_result["target_components"][0]["accession"]
            return str(accession) if accession is not None else None
    except Exception as e:
        raise Exception(f"Error fetching protein accession: {e}")
    return None


def get_target_chembl_id(assay_chembl_id: str) -> Optional[str]:
    """Returns the target chembl id for a given assay chembl id.

    Args:
        assay_chembl_id: Chembl id of the assay.

    Returns:
        Optional[str]: The target ChEMBL ID if found, None otherwise.

    Raises:
        Exception: If there is an error fetching the target ChEMBL ID. For example,
        if the assay ID is invalid.
    """
    assay = new_client.assay
    try:
        assay_result: Dict[str, Any] = assay.get(assay_chembl_id)
        if "target_chembl_id" in assay_result:
            target_chembl_id = assay_result["target_chembl_id"]
            return str(target_chembl_id) if target_chembl_id is not None else None
    except Exception as e:
        raise Exception(f"Error fetching target ChEMBL ID: {e}")
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
    using either ESM2 or ESM3 models. It processes all proteins in a single batch for optimal
    efficiency and caches models to avoid reloading.

    Args:
        protein_dict: Dictionary mapping protein IDs to their amino acid sequences.
        featurizer: Name of the pre-trained model to use. Supports ESM2 and ESM3 models:
            - ESM2: "esm2_t12_35M_UR50D", "esm2_t33_650M_UR50D"
            - ESM3: "esm3_sm_open_v1", "esm3_open_small"
        layer: Which transformer layer to extract embeddings from. If None, uses final embeddings.

    Returns:
        np.ndarray: Array of shape (num_proteins, embedding_dim) containing the computed
            protein embeddings.

    Note:
        - Models are cached globally to avoid reloading on subsequent calls
        - All proteins are processed in a single batch for optimal efficiency
        - For ESM2: requires fair-esm package
        - For ESM3: requires esm package with ESM3 support
    """
    # Available models
    esm2_models = ["esm2_t12_35M_UR50D", "esm2_t33_650M_UR50D"]
    esm3_models = ["esm3_sm_open_v1", "esm3_open_small"]

    if featurizer in esm2_models:
        return _get_protein_features_esm2(protein_dict, featurizer, layer)
    elif featurizer in esm3_models:
        return _get_protein_features_esm3(protein_dict, featurizer, layer)
    else:
        all_models = esm2_models + esm3_models
        raise ValueError(f"Unsupported featurizer: {featurizer}. Available models: {all_models}")


def _get_protein_features_esm2(
    protein_dict: Dict[str, str], featurizer: str, layer: Optional[int] = None
) -> np.ndarray:
    """Optimized ESM2 feature computation with model caching."""
    global _MODEL_CACHE

    # Check if model is cached
    cache_key = f"esm2_{featurizer}"
    if cache_key not in _MODEL_CACHE:
        print(f"Loading ESM2 model {featurizer} (will be cached for future use)...")
        model, alphabet = torch.hub.load("facebookresearch/esm", featurizer)
        batch_converter = alphabet.get_batch_converter()
        model.eval()  # disables dropout for deterministic results

        _MODEL_CACHE[cache_key] = {"model": model, "alphabet": alphabet, "batch_converter": batch_converter}
    else:
        print(f"Using cached ESM2 model {featurizer}")
        cached = _MODEL_CACHE[cache_key]
        model = cached["model"]
        alphabet = cached["alphabet"]
        batch_converter = cached["batch_converter"]

    # Convert protein dictionary to list of (id, sequence) tuples
    data = list(protein_dict.items())
    print(f"Processing batch of {len(data)} proteins with ESM2")

    batch_labels, batch_strs, batch_tokens = batch_converter(data)

    # Calculate sequence lengths excluding padding tokens
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

    if layer is None:
        # Extract layer number from model name
        if "t12" in featurizer:
            layer = 12
        elif "t33" in featurizer:
            layer = 33
        else:
            # Default to the last layer for unknown models
            layer = 33

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


def _get_protein_features_esm3(
    protein_dict: Dict[str, str], featurizer: str, layer: Optional[int] = None
) -> np.ndarray:
    """Optimized ESM3 feature computation with model caching and improved batch processing.

    This function implements better batching for ESM3 models by trying to process multiple
    proteins together when possible, and caches the model to avoid reloading.
    """
    try:
        from esm.models.esm3 import ESM3
        from esm.sdk.api import ESM3InferenceClient, ESMProtein, LogitsConfig
    except ImportError as e:
        raise ImportError(
            "ESM3 not found. Please install the esm package with ESM3 support: pip install esm"
        ) from e

    global _MODEL_CACHE

    # Check if model is cached
    cache_key = f"esm3_{featurizer}"
    if cache_key not in _MODEL_CACHE:
        print(f"Loading ESM3 model {featurizer} (will be cached for future use)...")
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            client: ESM3InferenceClient = ESM3.from_pretrained(featurizer).to(device)
            _MODEL_CACHE[cache_key] = {"client": client}
        except Exception as e:
            raise RuntimeError(f"Failed to initialize ESM3 model {featurizer}: {e}") from e
    else:
        print(f"Using cached ESM3 model {featurizer}")
        client = _MODEL_CACHE[cache_key]["client"]

    print(f"Processing batch of {len(protein_dict)} proteins with ESM3")
    sequence_representations = []

    # Try to process proteins in batches where possible
    # Note: ESM3 API may have limitations on batch processing, so we process one by one
    # but with the cached model this is still much more efficient

    for protein_id, sequence in protein_dict.items():
        if not sequence or not isinstance(sequence, str):
            raise ValueError(f"Invalid sequence for protein {protein_id}: {sequence}")

        try:
            # Create ESMProtein object from sequence
            protein = ESMProtein(sequence=sequence)

            # Encode the protein to get internal representation
            protein_tensor = client.encode(protein)

            # For ESM3, we can extract embeddings directly from the encoded tensor
            # This is more efficient than using logits for just getting embeddings
            if hasattr(protein_tensor, "sequence") and protein_tensor.sequence is not None:
                hidden_states = protein_tensor.sequence
                # Average over sequence dimension to get a single vector per protein
                protein_embedding = hidden_states.mean(dim=1).squeeze()

                # Convert to numpy if it's a torch tensor
                if hasattr(protein_embedding, "cpu"):
                    protein_embedding = protein_embedding.cpu().numpy()
                elif hasattr(protein_embedding, "detach"):
                    protein_embedding = protein_embedding.detach().cpu().numpy()
                elif hasattr(protein_embedding, "numpy"):
                    protein_embedding = protein_embedding.numpy()

                sequence_representations.append(protein_embedding)
            else:
                # Fallback to logits method if direct access doesn't work
                logits_config = LogitsConfig(sequence=True)
                embeddings = client.logits(protein_tensor, logits_config)

                if hasattr(embeddings, "sequence_logits") and embeddings.sequence_logits is not None:
                    sequence_logits = embeddings.sequence_logits
                    protein_embedding = sequence_logits.mean(dim=1).squeeze()

                    if hasattr(protein_embedding, "cpu"):
                        protein_embedding = protein_embedding.cpu().numpy()
                    elif hasattr(protein_embedding, "detach"):
                        protein_embedding = protein_embedding.detach().cpu().numpy()
                    elif hasattr(protein_embedding, "numpy"):
                        protein_embedding = protein_embedding.numpy()

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


def clear_protein_model_cache() -> None:
    """Clear the global protein model cache to free memory."""
    global _MODEL_CACHE
    _MODEL_CACHE.clear()
    print("Protein model cache cleared")


def get_cached_models() -> List[str]:
    """Get list of currently cached protein models."""
    global _MODEL_CACHE
    return list(_MODEL_CACHE.keys())


# Keep the original function name for backwards compatibility
def get_protein_features_esm3(
    protein_dict: Dict[str, str], featurizer: str = "esm3_sm_open_v1", layer: Optional[int] = None
) -> np.ndarray:
    """Legacy function - now redirects to the optimized dispatcher.

    Deprecated: Use get_protein_features() instead which handles both ESM2 and ESM3.
    """
    return get_protein_features(protein_dict, featurizer, layer)
