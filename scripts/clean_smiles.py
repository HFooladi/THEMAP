#!/usr/bin/env python3
"""
SMILES Cleaning Utility for THEMAP Data Files

This script reads JSONL.GZ files (with SMILES field) or CSV files (with SMILES column),
validates SMILES strings using RDKit, and creates cleaned versions with only valid,
parsable SMILES.

Usage:
    python scripts/clean_smiles.py input.jsonl.gz
    python scripts/clean_smiles.py input.csv --smiles-column SMILES
    python scripts/clean_smiles.py input.jsonl.gz --output cleaned_input.jsonl.gz
    python scripts/clean_smiles.py datasets/train/CHEMBL1613776.jsonl.gz --dry-run
    python scripts/clean_smiles.py datasets/ --recursive
"""

import argparse
import csv
import gzip
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

try:
    from rdkit import Chem, RDLogger
    from rdkit.Chem import SanitizeMol

    # Suppress RDKit warnings during SMILES validation
    RDLogger.DisableLog("rdApp.*")
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    # Only exit if not showing help
    if len(sys.argv) == 1 or "--help" not in sys.argv and "-h" not in sys.argv:
        print("Error: RDKit is required but not installed. Please install with:")
        print("  conda install -c conda-forge rdkit")
        sys.exit(1)

try:
    import pandas as pd

    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False


class SmilesCleaningStats:
    """Track statistics during SMILES cleaning."""

    def __init__(self):
        self.total_molecules = 0
        self.valid_molecules = 0
        self.invalid_smiles = []
        self.parsing_errors = []

    @property
    def invalid_count(self) -> int:
        return len(self.invalid_smiles)

    @property
    def success_rate(self) -> float:
        if self.total_molecules == 0:
            return 0.0
        return (self.valid_molecules / self.total_molecules) * 100

    def add_valid_molecule(self):
        self.total_molecules += 1
        self.valid_molecules += 1

    def add_invalid_molecule(self, smiles: str, error: str):
        self.total_molecules += 1
        self.invalid_smiles.append(smiles)
        self.parsing_errors.append(error)

    def print_summary(self, file_path: Union[str, Path]):
        print(f"\n📊 Cleaning Summary for {file_path}")
        print(f"   Total molecules: {self.total_molecules}")
        print(f"   Valid molecules: {self.valid_molecules}")
        print(f"   Invalid molecules: {self.invalid_count}")
        print(f"   Success rate: {self.success_rate:.1f}%")

        if self.invalid_count > 0:
            print("\n❌ Invalid SMILES found:")
            for i, (smiles, error) in enumerate(zip(self.invalid_smiles[:5], self.parsing_errors[:5])):
                print(f"   {i + 1}. '{smiles}' - {error}")
            if self.invalid_count > 5:
                print(f"   ... and {self.invalid_count - 5} more")


def validate_smiles(smiles: str, strict: bool = True) -> Tuple[bool, Optional[str]]:
    """
    Validate a SMILES string using RDKit.

    Args:
        smiles: SMILES string to validate
        strict: If True, perform full sanitization checks

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not smiles or not isinstance(smiles, str):
        return False, "Empty or non-string SMILES"

    # Handle pandas NaN values
    if PANDAS_AVAILABLE and pd.isna(smiles):
        return False, "NaN SMILES value"

    try:
        # Parse SMILES
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        if mol is None:
            return False, "RDKit cannot parse this SMILES"

        if strict:
            # Perform full sanitization
            try:
                SanitizeMol(mol)
            except Exception as e:
                return False, f"Sanitization failed: {str(e)}"

        # Additional basic checks
        if mol.GetNumAtoms() == 0:
            return False, "Molecule has no atoms"

        return True, None

    except Exception as e:
        return False, f"Validation error: {str(e)}"


def clean_jsonl_gz_file(
    input_path: Path,
    output_path: Optional[Path] = None,
    strict: bool = True,
    dry_run: bool = False,
) -> SmilesCleaningStats:
    """
    Clean a JSONL.GZ file by removing invalid SMILES.

    Args:
        input_path: Path to input JSONL.GZ file
        output_path: Path for cleaned output file (default: adds '_cleaned' suffix)
        strict: Whether to perform strict RDKit validation
        dry_run: If True, only analyze without writing output

    Returns:
        SmilesCleaningStats object with cleaning statistics
    """
    if output_path is None:
        # Create default output path
        stem = input_path.stem.replace(".jsonl", "")  # Remove .jsonl from .jsonl.gz
        output_path = input_path.parent / f"{stem}_cleaned.jsonl.gz"

    stats = SmilesCleaningStats()

    # Validate input file exists
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    logging.info(f"Processing JSONL.GZ file: {input_path}")
    if not dry_run:
        logging.info(f"Output will be saved to {output_path}")

    # Process the file
    try:
        with gzip.open(input_path, "rt", encoding="utf-8") as infile:
            if dry_run:
                # Dry run - just analyze
                for line_num, line in enumerate(infile, 1):
                    try:
                        data = json.loads(line.strip())
                        smiles = data.get("SMILES", "")

                        is_valid, error_msg = validate_smiles(smiles, strict=strict)
                        if is_valid:
                            stats.add_valid_molecule()
                        else:
                            stats.add_invalid_molecule(smiles, error_msg)

                    except json.JSONDecodeError as e:
                        logging.warning(f"JSON decode error at line {line_num}: {e}")
                        stats.add_invalid_molecule("", f"JSON parse error: {e}")
                    except Exception as e:
                        logging.warning(f"Error processing line {line_num}: {e}")
                        stats.add_invalid_molecule("", f"Processing error: {e}")
            else:
                # Real run - clean and write
                with gzip.open(output_path, "wt", encoding="utf-8") as outfile:
                    for line_num, line in enumerate(infile, 1):
                        try:
                            data = json.loads(line.strip())
                            smiles = data.get("SMILES", "")

                            is_valid, error_msg = validate_smiles(smiles, strict=strict)
                            if is_valid:
                                # Write valid molecule to output
                                outfile.write(line)
                                stats.add_valid_molecule()
                            else:
                                stats.add_invalid_molecule(smiles, error_msg)
                                logging.debug(
                                    f"Skipped invalid SMILES at line {line_num}: {smiles} - {error_msg}"
                                )

                        except json.JSONDecodeError as e:
                            logging.warning(f"JSON decode error at line {line_num}: {e}")
                            stats.add_invalid_molecule("", f"JSON parse error: {e}")
                        except Exception as e:
                            logging.warning(f"Error processing line {line_num}: {e}")
                            stats.add_invalid_molecule("", f"Processing error: {e}")

    except Exception as e:
        logging.error(f"Error processing file {input_path}: {e}")
        raise

    return stats


def clean_csv_file(
    input_path: Path,
    output_path: Optional[Path] = None,
    smiles_column: str = "SMILES",
    strict: bool = True,
    dry_run: bool = False,
) -> SmilesCleaningStats:
    """
    Clean a CSV file by removing rows with invalid SMILES.

    Args:
        input_path: Path to input CSV file
        output_path: Path for cleaned output file (default: adds '_cleaned' suffix)
        smiles_column: Name of the column containing SMILES strings
        strict: Whether to perform strict RDKit validation
        dry_run: If True, only analyze without writing output

    Returns:
        SmilesCleaningStats object with cleaning statistics
    """
    if output_path is None:
        # Create default output path
        stem = input_path.stem
        suffix = input_path.suffix
        output_path = input_path.parent / f"{stem}_cleaned{suffix}"

    stats = SmilesCleaningStats()

    # Validate input file exists
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    logging.info(f"Processing CSV file: {input_path}")
    logging.info(f"Using SMILES column: '{smiles_column}'")
    if not dry_run:
        logging.info(f"Output will be saved to {output_path}")

    try:
        if PANDAS_AVAILABLE:
            # Use pandas for better CSV handling
            df = pd.read_csv(input_path)

            if smiles_column not in df.columns:
                available_cols = ", ".join(df.columns.tolist())
                raise ValueError(
                    f"SMILES column '{smiles_column}' not found in CSV. Available columns: {available_cols}"
                )

            valid_rows = []

            for idx, row in df.iterrows():
                smiles = row[smiles_column]
                is_valid, error_msg = validate_smiles(smiles, strict=strict)

                if is_valid:
                    valid_rows.append(idx)
                    stats.add_valid_molecule()
                else:
                    stats.add_invalid_molecule(str(smiles), error_msg)
                    logging.debug(f"Skipped invalid SMILES at row {idx + 1}: {smiles} - {error_msg}")

            if not dry_run and valid_rows:
                # Write cleaned data
                cleaned_df = df.iloc[valid_rows]
                cleaned_df.to_csv(output_path, index=False)

        else:
            # Use standard CSV module
            valid_rows = []
            header = None
            smiles_col_idx = None

            with open(input_path, "r", encoding="utf-8") as infile:
                reader = csv.reader(infile)

                # Read header
                header = next(reader)
                if smiles_column not in header:
                    available_cols = ", ".join(header)
                    raise ValueError(
                        f"SMILES column '{smiles_column}' not found in CSV. Available columns: {available_cols}"
                    )

                smiles_col_idx = header.index(smiles_column)

                # Process rows
                for row_num, row in enumerate(reader, 2):  # Start at 2 because of header
                    if len(row) > smiles_col_idx:
                        smiles = row[smiles_col_idx]
                        is_valid, error_msg = validate_smiles(smiles, strict=strict)

                        if is_valid:
                            valid_rows.append(row)
                            stats.add_valid_molecule()
                        else:
                            stats.add_invalid_molecule(smiles, error_msg)
                            logging.debug(f"Skipped invalid SMILES at row {row_num}: {smiles} - {error_msg}")
                    else:
                        stats.add_invalid_molecule("", f"Row too short at line {row_num}")

            # Write cleaned data
            if not dry_run and (valid_rows or header):
                with open(output_path, "w", encoding="utf-8", newline="") as outfile:
                    writer = csv.writer(outfile)
                    writer.writerow(header)
                    writer.writerows(valid_rows)

    except Exception as e:
        logging.error(f"Error processing file {input_path}: {e}")
        raise

    return stats


def detect_file_type(file_path: Path) -> str:
    """Detect file type based on extension."""
    suffixes = "".join(file_path.suffixes).lower()

    if suffixes.endswith(".jsonl.gz"):
        return "jsonl_gz"
    elif suffixes.endswith(".csv") or suffixes.endswith(".csv.gz"):
        return "csv"
    else:
        raise ValueError(f"Unsupported file type: {file_path}. Supported: .jsonl.gz, .csv")


def clean_file(
    input_path: Path,
    output_path: Optional[Path] = None,
    smiles_column: str = "SMILES",
    strict: bool = True,
    dry_run: bool = False,
) -> SmilesCleaningStats:
    """
    Clean a file by removing invalid SMILES (auto-detects file type).

    Args:
        input_path: Path to input file
        output_path: Path for cleaned output file
        smiles_column: Name of SMILES column (for CSV files)
        strict: Whether to perform strict RDKit validation
        dry_run: If True, only analyze without writing output

    Returns:
        SmilesCleaningStats object with cleaning statistics
    """
    file_type = detect_file_type(input_path)

    if file_type == "jsonl_gz":
        return clean_jsonl_gz_file(input_path, output_path, strict, dry_run)
    elif file_type == "csv":
        return clean_csv_file(input_path, output_path, smiles_column, strict, dry_run)
    else:
        raise ValueError(f"Unsupported file type detected: {file_type}")


def process_directory(
    directory: Path,
    output_directory: Optional[Path] = None,
    smiles_column: str = "SMILES",
    strict: bool = True,
    dry_run: bool = False,
) -> Dict[Path, SmilesCleaningStats]:
    """
    Process all supported files in a directory.

    Args:
        directory: Directory containing files to clean
        output_directory: Directory for cleaned files (default: same as input)
        smiles_column: Name of SMILES column (for CSV files)
        strict: Whether to perform strict RDKit validation
        dry_run: If True, only analyze without writing output

    Returns:
        Dictionary mapping file paths to their cleaning statistics
    """
    if output_directory is None:
        output_directory = directory

    results = {}

    # Find all supported files
    supported_files = []
    for pattern in ["**/*.jsonl.gz", "**/*.csv"]:
        supported_files.extend(directory.glob(pattern))

    if not supported_files:
        logging.warning(f"No supported files found in {directory}")
        return results

    logging.info(f"Found {len(supported_files)} files to process")

    for file_path in supported_files:
        try:
            # Create relative output path
            relative_path = file_path.relative_to(directory)
            output_path = output_directory / relative_path

            # Ensure output directory exists
            if not dry_run:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                # Update filename for cleaned version
                if file_path.suffix == ".gz" and file_path.stem.endswith(".jsonl"):
                    # Handle .jsonl.gz files
                    base_stem = file_path.stem.replace(".jsonl", "")
                    output_path = output_path.parent / f"{base_stem}_cleaned.jsonl.gz"
                elif file_path.suffix == ".csv":
                    # Handle .csv files
                    output_path = output_path.parent / f"{file_path.stem}_cleaned.csv"
            else:
                output_path = None

            stats = clean_file(file_path, output_path, smiles_column, strict=strict, dry_run=dry_run)
            results[file_path] = stats

        except Exception as e:
            logging.error(f"Failed to process {file_path}: {e}")
            # Create error stats
            error_stats = SmilesCleaningStats()
            error_stats.add_invalid_molecule("", str(e))
            results[file_path] = error_stats

    return results


def create_parser() -> argparse.ArgumentParser:
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Clean JSONL.GZ and CSV files by removing invalid SMILES",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Clean a JSONL.GZ file
  python scripts/clean_smiles.py datasets/train/CHEMBL1613776.jsonl.gz

  # Clean a CSV file
  python scripts/clean_smiles.py data/molecules.csv --smiles-column SMILES

  # Clean with custom output
  python scripts/clean_smiles.py input.csv --output cleaned.csv

  # Dry run to analyze without cleaning
  python scripts/clean_smiles.py input.jsonl.gz --dry-run

  # Process all files in directory recursively
  python scripts/clean_smiles.py datasets/ --recursive

  # Process with less strict validation
  python scripts/clean_smiles.py input.csv --lenient --smiles-column smiles
        """,
    )

    parser.add_argument("input_path", help="Input file (.jsonl.gz or .csv) or directory to clean")

    parser.add_argument("--output", "-o", help='Output file path (default: adds "_cleaned" suffix)')

    parser.add_argument(
        "--smiles-column", default="SMILES", help="Name of SMILES column in CSV files (default: SMILES)"
    )

    parser.add_argument(
        "--recursive", "-r", action="store_true", help="Process all supported files in directory recursively"
    )

    parser.add_argument(
        "--dry-run", action="store_true", help="Analyze files without writing cleaned versions"
    )

    parser.add_argument(
        "--lenient", action="store_true", help="Use less strict SMILES validation (skip sanitization)"
    )

    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)",
    )

    parser.add_argument(
        "--quiet", "-q", action="store_true", help="Only show summary, suppress detailed output"
    )

    return parser


def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()

    # Check if RDKit is available when actually running (not just help)
    if not RDKIT_AVAILABLE:
        print("Error: RDKit is required but not installed. Please install with:")
        print("  conda install -c conda-forge rdkit")
        return 1

    # Setup logging
    log_level = getattr(logging, args.log_level.upper())
    if args.quiet:
        log_level = logging.WARNING

    logging.basicConfig(level=log_level, format="%(levelname)s: %(message)s")

    input_path = Path(args.input_path)

    if not input_path.exists():
        print(f"❌ Error: Path does not exist: {input_path}")
        return 1

    strict_validation = not args.lenient

    try:
        if input_path.is_file():
            # Process single file
            output_path = Path(args.output) if args.output else None
            stats = clean_file(
                input_path,
                output_path,
                smiles_column=args.smiles_column,
                strict=strict_validation,
                dry_run=args.dry_run,
            )
            stats.print_summary(input_path)

        elif input_path.is_dir() and args.recursive:
            # Process directory
            output_dir = Path(args.output) if args.output else None
            results = process_directory(
                input_path,
                output_dir,
                smiles_column=args.smiles_column,
                strict=strict_validation,
                dry_run=args.dry_run,
            )

            # Print summary for all files
            total_files = len(results)
            total_molecules = sum(stats.total_molecules for stats in results.values())
            total_valid = sum(stats.valid_molecules for stats in results.values())
            total_invalid = sum(stats.invalid_count for stats in results.values())

            print("\n📊 Overall Summary")
            print(f"   Files processed: {total_files}")
            print(f"   Total molecules: {total_molecules}")
            print(f"   Valid molecules: {total_valid}")
            print(f"   Invalid molecules: {total_invalid}")
            if total_molecules > 0:
                print(f"   Overall success rate: {(total_valid / total_molecules) * 100:.1f}%")

            # Show individual file results
            print("\n📁 Individual File Results:")
            for file_path, stats in results.items():
                if stats.invalid_count > 0:
                    print(f"   ❌ {file_path.name}: {stats.invalid_count}/{stats.total_molecules} invalid")
                else:
                    print(f"   ✅ {file_path.name}: All {stats.total_molecules} molecules valid")
        else:
            print(f"❌ Error: {input_path} is a directory. Use --recursive to process directories.")
            return 1

        return 0

    except KeyboardInterrupt:
        print("\n⏹️  Operation cancelled by user")
        return 1
    except Exception as e:
        print(f"❌ Error: {e}")
        logging.exception("Detailed error information:")
        return 1


if __name__ == "__main__":
    sys.exit(main())
