"""This script is used to compute the embedding for molecules in the test tasks and train tasks.
The embedding is computed using the specified available featurizers. Then for each featurizer, a dictionary
is created that maps the task name to the embedding of the molecules in that task. The dictionary is like the following:
{'CheMBL1234:{'ecfp': torch.Tensor, 'labels': torch.Tensor, 'smiles': np.array}, 'CheMBL5678': {'ecfp': torch.Tensor, 'labels': torch.Tensor, 'smiles': np.array}, ...}
The dictionary is then saved to a pickle file (for train and test tasks separately).

Usage:
    python scripts/task_embedding_molecules.py --featurizer [featurizer_name] --output_path [path/to/output] --n_jobs [num_jobs]
    
    - featurizer: Name of featurizer to use (or "all" to use all available featurizers)
    - output_path: Path to save the output pickle files (default: dataset/embedding/ecfp_train.pkl)
    - n_jobs: Number of parallel jobs for featurization (default: 32)
"""

import os
import sys
import pickle
from argparse import ArgumentParser

# Setting up local details:
# This should be the location of the checkout of the THEMAP repository:
repo_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKOUT_PATH = repo_path
DATASET_PATH = os.path.join(repo_path, "datasets")

os.chdir(CHECKOUT_PATH)
sys.path.insert(0, CHECKOUT_PATH)


from fs_mol.data import FSMolDataset, DataFold
from themap.utils.distance_utils import compute_features_smiles_labels
from tqdm import tqdm
import pandas as pd
import torch
import datamol as dm
from molfeat.calc import FPCalculator
from molfeat.trans.fp import FPVecTransformer
from molfeat.trans import MoleculeTransformer
from molfeat.trans.pretrained import GraphormerTransformer, PretrainedDGLTransformer
from molfeat.trans.pretrained.hf_transformers import PretrainedHFTransformer


# Available featurizers for molecule embedding generation
# These featurizers fall into several categories:
# 1. Fingerprint-based (ecfp, fcfp, maccs): Molecular fingerprints that encode structural features
# 2. Descriptor-based (mordred, desc2D): Chemical descriptors calculated from molecular structure
# 3. Pretrained language models (ChemBERTa, Roberta): Transformer-based text embeddings for SMILES
# 4. Graph neural networks (gin_supervised_*): Graph embeddings from pretrained GNN models
# 5. Multimodal models (MolT5): Models that can work with both text and molecular representations
AVAILABLE_FEATURIZERS = [
    "ecfp",             # Extended-Connectivity Fingerprints
    "mordred",          # Mordred molecular descriptors 
    "desc2D",           # 2D molecular descriptors
    "maccs",            # MACCS structural keys fingerprint
    "ChemBERTa-77M-MLM", # ChemBERTa model with masked language modeling pretraining
    "ChemBERTa-77M-MTR", # ChemBERTa model with molecular translation pretraining
    "Roberta-Zinc480M-102M", # RoBERTa model pretrained on ZINC dataset
    "gin_supervised_infomax",      # GIN with supervised InfoMax pretraining
    "gin_supervised_contextpred",  # GIN with supervised context prediction pretraining
    "gin_supervised_edgepred",     # GIN with supervised edge prediction pretraining
    "gin_supervised_masking",      # GIN with supervised masking pretraining
    "MolT5",            # Text-to-molecule and molecule-to-text model
]

"""
Known issues with featurizers:
- desc3D not working: Requires 3D conformers which may not be available or consistent
- usrcat not working: Similar issues with 3D structure requirements
"""


def parse_args():
    """Parse command line arguments for the script.
    
    Returns:
        argparse.Namespace: Object containing the parsed arguments
    """
    parser = ArgumentParser(description="Generate molecular embeddings for train and test tasks")
    parser.add_argument("--output_path", type=str, default="dataset/embedding/ecfp_train.pkl", 
                        help="Path to save the output embeddings")
    parser.add_argument("--n_jobs", type=int, default=32, 
                        help="Number of parallel jobs for featurization")
    parser.add_argument("--featurizer", type=str, default="", 
                        help="Featurizer to use (one of the AVAILABLE_FEATURIZERS or 'all')")
    args = parser.parse_args()
    return args


def main():
    """Main function to generate and save molecular embeddings for train and test tasks.
    
    This function:
    1. Loads the FSMol dataset
    2. Reads all test and train tasks
    3. For each specified featurizer:
       a. Initializes the appropriate transformer
       b. Computes features for all molecules in test and train tasks
       c. Organizes features by task name
       d. Saves the embeddings to pickle files
    """
    args = parse_args()
    
    # Load the FSMol dataset from the specified directory
    dataset = FSMolDataset.from_directory(
        DATASET_PATH, task_list_file=os.path.join(DATASET_PATH, "fsmol-0.1.json")
    )

    # Process test tasks
    # Each task contains `MoleculeDatapoint` objects with molecule info
    test_tasks = []
    print("Reading test tasks ...")
    test_task_iterable = dataset.get_task_reading_iterable(DataFold.TEST)
    for task in tqdm(iter(test_task_iterable)):
        test_tasks.append(task)

    # Process train tasks
    print("Reading train tasks ...")
    train_tasks = []
    train_task_iterable = dataset.get_task_reading_iterable(DataFold.TRAIN)
    for task in tqdm(iter(train_task_iterable)):
        train_tasks.append(task)

    # Determine which featurizers to use
    if args.featurizer == "all":
        FEATURIZERS_LIST = AVAILABLE_FEATURIZERS
    else:
        FEATURIZERS_LIST = [args.featurizer]

    # Process each featurizer
    for featurizer in tqdm(FEATURIZERS_LIST):
        print(f"Processing featurizer: {featurizer}")
        
        # Initialize the appropriate transformer based on featurizer type
        if featurizer in ["ecfp", "fcfp", "mordred"]:
            # Fingerprint or descriptor-based featurizers
            if featurizer in ["ecfp"]:
                calc = FPCalculator(featurizer)
                transformer = MoleculeTransformer(calc, n_jobs=args.n_jobs)
            else:
                transformer = MoleculeTransformer(featurizer, n_jobs=args.n_jobs)

        elif featurizer in ["desc2D", "desc3D", "maccs", "usrcat"]:
            # Vector-based fingerprint or descriptor featurizers
            transformer = FPVecTransformer(kind=featurizer, dtype=float, n_jobs=args.n_jobs)

        elif featurizer in ["pcqm4mv2_graphormer_base"]:
            # Graphormer-based transformers (graph neural networks)
            transformer = GraphormerTransformer(kind=featurizer, dtype=float, n_jobs=args.n_jobs)

        elif featurizer in [
            "ChemBERTa-77M-MLM",
            "ChemBERTa-77M-MTR",
            "Roberta-Zinc480M-102M",
            "MolT5",
        ]:
            # Hugging Face transformer models for molecular representations
            transformer = PretrainedHFTransformer(
                kind=featurizer, notation="smiles", dtype=float, n_jobs=args.n_jobs
            )

        elif featurizer in [
            "gin_supervised_infomax",
            "gin_supervised_contextpred",
            "gin_supervised_edgepred",
            "gin_supervised_masking",
        ]:
            # DGL-based Graph Neural Network transformers
            transformer = PretrainedDGLTransformer(kind=featurizer, dtype=float, n_jobs=args.n_jobs)

        # Compute embeddings for test tasks
        print("Computing features for test tasks ...")
        test_features = [compute_features_smiles_labels(task, transformer) for task in tqdm(test_tasks)]

        # Compute embeddings for train tasks
        print("Computing features for train tasks ...")
        train_features = [compute_features_smiles_labels(task, transformer) for task in tqdm(train_tasks)]

        # Organize features by task name for test tasks
        features_test = {
            test_tasks[i].name: {featurizer: feature[0], "labels": feature[1], "smiles": feature[2]}
            for i, feature in enumerate(test_features)
        }

        # Organize features by task name for train tasks
        features_train = {
            train_tasks[i].name: {
                featurizer: feature[0],
                "labels": feature[1],
                "smiles": feature[2],
            }
            for i, feature in enumerate(train_features)
        }

        # Define output paths for saving embeddings
        path_to_save_embedding_test = os.path.join(DATASET_PATH, "embeddings", f"{featurizer}_test.pkl")
        path_to_save_embedding_train = os.path.join(DATASET_PATH, "embeddings", f"{featurizer}_train.pkl")

        # Create the embeddings directory if it doesn't exist
        os.makedirs(os.path.join(DATASET_PATH, "embeddings"), exist_ok=True)
        
        # Save test embeddings to pickle file
        print(f"Saving test embeddings to {path_to_save_embedding_test}")
        with open(path_to_save_embedding_test, "wb") as f:
            pickle.dump(features_test, f)

        # Save train embeddings to pickle file
        print(f"Saving train embeddings to {path_to_save_embedding_train}")
        with open(path_to_save_embedding_train, "wb") as f:
            pickle.dump(features_train, f)


if __name__ == "__main__":
    main()
