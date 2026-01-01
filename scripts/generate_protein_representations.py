#!/usr/bin/env python3
"""
Generate ESM2 protein representations from FASTA files and save as JSON.

This script processes protein FASTA files from source and target folders,
generates ESM2 representations, and saves them as JSON files where:
- Keys are CHEMBL IDs
- Values are lists containing the protein representations from ESM2

Usage:
    python generate_protein_representations.py
"""

import json
import sys
from pathlib import Path
from typing import Dict, List

from Bio import SeqIO
from tqdm import tqdm

# Add the current repository to Python path
REPO_PATH = Path(__file__).parent.absolute()
sys.path.insert(0, str(REPO_PATH))

from themap.utils.logging import get_logger, setup_logging
from themap.utils.protein_utils import get_protein_features

# Initialize logging
setup_logging()
logger = get_logger(__name__)


def extract_sequence_from_fasta(fasta_file: Path) -> str:
    """Extract protein sequence from FASTA file.

    Args:
        fasta_file: Path to the FASTA file

    Returns:
        Protein sequence as string
    """
    try:
        with open(fasta_file, "r") as f:
            records = list(SeqIO.parse(f, "fasta"))
            if records:
                return str(records[0].seq)
            else:
                logger.warning(f"No sequences found in {fasta_file}")
                return ""
    except Exception as e:
        logger.error(f"Error reading {fasta_file}: {e}")
        return ""


def extract_chembl_id_from_filename(fasta_file: Path) -> str:
    """Extract CHEMBL ID from filename.

    Args:
        fasta_file: Path to the FASTA file

    Returns:
        CHEMBL ID extracted from filename
    """
    # Filename format is expected to be CHEMBL*.fasta
    return fasta_file.stem  # This removes the .fasta extension


def process_folder(folder_path: Path, esm2_model: str = "esm2_t33_650M_UR50D") -> Dict[str, List[float]]:
    """Process all FASTA files in a folder and generate ESM2 representations.

    Args:
        folder_path: Path to folder containing FASTA files
        esm2_model: ESM2 model to use for generating representations

    Returns:
        Dictionary mapping CHEMBL IDs to their ESM2 representations
    """
    logger.info(f"Processing folder: {folder_path}")

    # Find all individual CHEMBL FASTA files in the folder (exclude combined files)
    all_fasta_files = list(folder_path.glob("*.fasta"))
    fasta_files = [f for f in all_fasta_files if f.stem.startswith("CHEMBL")]
    logger.info(
        f"Found {len(fasta_files)} individual CHEMBL FASTA files (filtered from {len(all_fasta_files)} total)"
    )

    if not fasta_files:
        logger.warning(f"No FASTA files found in {folder_path}")
        return {}

    # Extract sequences and CHEMBL IDs
    protein_sequences = {}
    chembl_ids = []

    for fasta_file in tqdm(fasta_files, desc="Reading FASTA files"):
        chembl_id = extract_chembl_id_from_filename(fasta_file)
        sequence = extract_sequence_from_fasta(fasta_file)

        if sequence:
            protein_sequences[chembl_id] = sequence
            chembl_ids.append(chembl_id)
        else:
            logger.warning(f"Skipping {fasta_file} - no valid sequence found")

    if not protein_sequences:
        logger.error(f"No valid protein sequences found in {folder_path}")
        return {}

    logger.info(f"Successfully loaded {len(protein_sequences)} protein sequences")

    # Generate ESM2 representations
    logger.info(f"Generating ESM2 representations using model: {esm2_model}")
    try:
        # The get_protein_features function returns a numpy array
        representations = get_protein_features(protein_sequences, featurizer=esm2_model)
        logger.info(f"Generated representations with shape: {representations.shape}")

        # Convert numpy array to dictionary mapping CHEMBL ID to representation list
        result = {}
        for i, chembl_id in enumerate(chembl_ids):
            # Convert numpy array to list for JSON serialization
            result[chembl_id] = representations[i].tolist()

        return result

    except Exception as e:
        logger.error(f"Error generating ESM2 representations: {e}")
        raise


def save_representations_to_json(representations: Dict[str, List[float]], output_file: Path):
    """Save protein representations to JSON file.

    Args:
        representations: Dictionary mapping CHEMBL IDs to representations
        output_file: Path to output JSON file
    """
    logger.info(f"Saving {len(representations)} representations to {output_file}")

    try:
        with open(output_file, "w") as f:
            json.dump(representations, f, indent=2)
        logger.info(f"Successfully saved representations to {output_file}")
    except Exception as e:
        logger.error(f"Error saving to {output_file}: {e}")
        raise


def main():
    """Main function to process protein FASTA files and generate ESM2 representations."""

    logger.info("🧬 Starting protein representation generation with ESM2")

    # Define paths
    datasets_dir = REPO_PATH / "datasets"
    source_folder = datasets_dir / "train"  # Source folder (train)
    target_folder = datasets_dir / "test"  # Target folder (test)

    # Output JSON files
    source_output = REPO_PATH / "protein_representations_source.json"
    target_output = REPO_PATH / "protein_representations_target.json"

    # Check if folders exist
    if not source_folder.exists():
        logger.error(f"Source folder not found: {source_folder}")
        return

    if not target_folder.exists():
        logger.error(f"Target folder not found: {target_folder}")
        return

    # ESM2 model to use
    esm2_model = "esm2_t33_650M_UR50D"  # Using the larger model for better representations

    try:
        # Process source folder (train)
        logger.info("=" * 50)
        logger.info("Processing SOURCE folder (train)")
        logger.info("=" * 50)
        source_representations = process_folder(source_folder, esm2_model)

        if source_representations:
            save_representations_to_json(source_representations, source_output)
        else:
            logger.error("No representations generated for source folder")
            return

        # Process target folder (test)
        logger.info("=" * 50)
        logger.info("Processing TARGET folder (test)")
        logger.info("=" * 50)
        target_representations = process_folder(target_folder, esm2_model)

        if target_representations:
            save_representations_to_json(target_representations, target_output)
        else:
            logger.error("No representations generated for target folder")
            return

        # Summary
        logger.info("=" * 50)
        logger.info("SUMMARY")
        logger.info("=" * 50)
        logger.info(f"Source (train) representations: {len(source_representations)} proteins")
        logger.info(f"Target (test) representations: {len(target_representations)} proteins")
        logger.info(f"Source JSON saved to: {source_output}")
        logger.info(f"Target JSON saved to: {target_output}")
        logger.info("✅ Protein representation generation completed successfully!")

        # Show example of generated data structure
        if source_representations:
            example_chembl = list(source_representations.keys())[0]
            example_repr = source_representations[example_chembl]
            logger.info(f"Example - {example_chembl}: representation length = {len(example_repr)}")

    except Exception as e:
        logger.error(f"Error during processing: {e}")
        raise


if __name__ == "__main__":
    main()
