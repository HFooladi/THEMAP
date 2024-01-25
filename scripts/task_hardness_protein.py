""" This script is used to compute the hardness of the test tasks from protein perspective/distance.
    The hardenss is computed based on the distance between the protein embedding of the test task and the protein embedding of the training tasks.
    The protein embedding is computed using the ESM2 model. The distance is computed using the Euclidean distance.
"""

import os
import pickle
import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import torch
from tqdm import tqdm

# Setting up local details:
# This should be the location of the checkout of the THEMAP repository:
repo_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKOUT_PATH = repo_path
DATASET_PATH = os.path.join(repo_path, "datasets")

os.chdir(CHECKOUT_PATH)
sys.path.insert(0, CHECKOUT_PATH)


from fs_mol.utils import compute_task_hardness_from_distance_matrix, normalize

"""
ESM2_Models = ["esm2_t6_8M_UR50D",
               "esm2_t12_35M_UR50S",
               "esm2_t30_150M_UR50S",
               "esm2_t33_650M_UR50D", 
               "esm2_t36_3B_UR50D"],
"""


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--esm2_model", type=str, default="esm2_t36_3B_UR50D", help="")
    parser.add_argument("--layer", type=int, default=36, help="")
    parser.add_argument("--k_nearest", type=float, default=10, help="number of training tasks to be considered for determining the hardness of a test task")

    args = parser.parse_args()
    return args


def get_protein_embedding(esm2_model: str = "esm2_t36_3B_UR50D", layer: int = 36) -> Tuple:
    ESM_EMBEDDING_PATH = os.path.join(DATASET_PATH, "targets", "esm2_output", esm2_model)
    target_train_df = pd.read_csv(
        os.path.join(DATASET_PATH, "targets", "train_proteins.csv")
    )
    target_valid_df = pd.read_csv(
        os.path.join(DATASET_PATH, "targets", "valid_proteins.csv")
    )
    target_test_df = pd.read_csv(os.path.join(FS_MOL_DATASET_PATH, "targets", "test_proteins.csv"))

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

    train_accession_ids = []
    valid_accession_ids = []
    test_accession_ids = []

    train_emb_tensor = torch.empty(0)
    valid_emb_tensor = torch.empty(0)
    test_emb_tensor = torch.empty(0)

    for file in tqdm(train_files):
        train_emb.append(torch.load(file))
        # train_emb_tensor = torch.cat((train_emb_tensor, train_emb[-1]['mean_representations'][layer][None, :]), 0)
        train_emb_label.append(train_emb[-1]["label"])

    for file in tqdm(valid_files):
        valid_emb.append(torch.load(file))
        # valid_emb_tensor = torch.cat((valid_emb_tensor, valid_emb[-1]['mean_representations'][layer][None, :]), 0)
        valid_emb_label.append(valid_emb[-1]["label"])

    for file in tqdm(test_files):
        test_emb.append(torch.load(file))
        # test_emb_tensor = torch.cat((test_emb_tensor, test_emb[-1]['mean_representations'][layer][None, :]), 0)
        test_emb_label.append(test_emb[-1]["label"])

    train_ids = [item.split("|")[1] for item in train_emb_label]
    valid_ids = [item.split("|")[1] for item in valid_emb_label]
    test_ids = [item.split("|")[1] for item in test_emb_label]

    for chembl_id in tqdm(target_train_df["chembl_id"]):
        target_accession_id = target_train_df["target_accession_id"][
            target_train_df["chembl_id"] == chembl_id
        ].item()
        id = train_ids.index(target_accession_id)
        train_emb_tensor = torch.cat(
            (train_emb_tensor, train_emb[id]["mean_representations"][layer][None, :]), 0
        )
        train_accession_ids.append(target_accession_id)

    for chembl_id in tqdm(target_valid_df["chembl_id"]):
        target_accession_id = target_valid_df["target_accession_id"][
            target_valid_df["chembl_id"] == chembl_id
        ].item()
        id = valid_ids.index(target_accession_id)
        valid_emb_tensor = torch.cat(
            (valid_emb_tensor, valid_emb[id]["mean_representations"][layer][None, :]), 0
        )
        valid_accession_ids.append(target_accession_id)

    for chembl_id in tqdm(target_test_df["chembl_id"]):
        target_accession_id = target_test_df["target_accession_id"][
            target_test_df["chembl_id"] == chembl_id
        ].item()
        id = test_ids.index(target_accession_id)
        test_emb_tensor = torch.cat(
            (test_emb_tensor, test_emb[id]["mean_representations"][layer][None, :]), 0
        )
        test_accession_ids.append(target_accession_id)

    return (
        train_emb_tensor,
        valid_emb_tensor,
        test_emb_tensor,
        train_accession_ids,
        valid_accession_ids,
        test_accession_ids,
    )


def main():
    args = parse_args()
    (
        train_emb_tensor,
        valid_emb_tensor,
        test_emb_tensor,
        train_accession_ids,
        valid_accession_ids,
        test_accession_ids,
    ) = get_protein_embedding(args.esm2_model, args.layer)

    ##ToDO: This is just reprs for unique proteins in the train set.
    ## I should add reprs for all the other proteins in the training set.

    distance_matrix = torch.cdist(train_emb_tensor, test_emb_tensor, p=2)

    hardness_protien = compute_task_hardness_from_distance_matrix(
        distance_matrix, aggr="mean_median", proportion=args.k_nearest
    )
    target_test_df = pd.read_csv(os.path.join(DATASET_PATH, "targets", "test_proteins.csv"))
    chembl_ids = []
    for item in test_accession_ids:
        chembl_ids.append(
            target_test_df["chembl_id"][target_test_df["target_accession_id"] == item].item()
        )

    hardness_protien_mean_norm = normalize(hardness_protien[0])
    hardness_protien_median_norm = normalize(hardness_protien[1])

    protein_hardness_df = pd.DataFrame(
        {
            "protein_hardness_mean": hardness_protien[0],
            "protien_hardness_median": hardness_protien[1],
            "protein_hardness_mean_norm": hardness_protien_mean_norm,
            "protein_hardness_median_norm": hardness_protien_median_norm,
            "accession_id": test_accession_ids,
            "assay": chembl_ids,
        }
    )

    protein_hardness_df.to_csv(
        os.path.join(DATASET_PATH, "targets", "esm2_output", args.esm2_model, f"protein_hardness_{args.k_nearest}.csv"), index=False
    )

    evaluation_output_directory = os.path.join(
        CHECKOUT_PATH, "outputs", "FSMol_Eval_ProtoNet_2023-02-15_12-21-54"
    )
    output_results = pd.read_csv(
        os.path.join(
            evaluation_output_directory, "summary", "ProtoNet_summary_num_train_requested_128.csv"
        )
    )

    df = protein_hardness_df.merge(output_results[["assay", "delta_auprc", "roc_auc"]], on="assay")
    print(
        "correlation between protein hardness and delta_auprc: ",
        df["protein_hardness_mean"].corr(df["delta_auprc"]),
    )
    print(
        "correlation between protein hardness and roc_auc: ",
        df["protein_hardness_mean"].corr(df["roc_auc"]),
    )


if __name__ == "__main__":
    main()
