#!/usr/bin/env python3
"""
CSV to JSONL.GZ Conversion Utility for THEMAP

This script converts CSV files containing molecular data to the JSONL.GZ format
expected by THEMAP. It includes integrated SMILES validation and cleaning.

Usage:
    python scripts/csv_to_jsonl.py input.csv CHEMBL123456
    python scripts/csv_to_jsonl.py input.csv CHEMBL123456 --output datasets/train/CHEMBL123456.jsonl.gz
    python scripts/csv_to_jsonl.py input.csv CHEMBL123456 --smiles-column SMILES --activity-column pIC50
    python scripts/csv_to_jsonl.py chembl_data.csv CHEMBL123456 --auto-detect
    python scripts/csv_to_jsonl.py input.csv CHEMBL123456 --dry-run
"""

import argparse
import csv
import gzip
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    from rdkit import Chem, RDLogger
    from rdkit.Chem import SanitizeMol

    # Suppress RDKit warnings during SMILES validation
    RDLogger.DisableLog("rdApp.*")
except ImportError:
    print("Error: RDKit is required but not installed. Please install with:")
    print("  conda install -c conda-forge rdkit")
    sys.exit(1)

try:
    import pandas as pd

    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    print("Warning: pandas not available. Using standard csv module (slower).")


class ConversionStats:
    """Track statistics during CSV to JSONL conversion."""

    def __init__(self):
        self.total_rows = 0
        self.valid_molecules = 0
        self.invalid_smiles = []
        self.parsing_errors = []
        self.missing_data = []

    @property
    def invalid_count(self) -> int:
        return len(self.invalid_smiles)

    @property
    def missing_count(self) -> int:
        return len(self.missing_data)

    @property
    def success_rate(self) -> float:
        if self.total_rows == 0:
            return 0.0
        return (self.valid_molecules / self.total_rows) * 100

    def add_valid_molecule(self):
        self.total_rows += 1
        self.valid_molecules += 1

    def add_invalid_molecule(self, smiles: str, error: str):
        self.total_rows += 1
        self.invalid_smiles.append(smiles)
        self.parsing_errors.append(error)

    def add_missing_data(self, reason: str):
        self.total_rows += 1
        self.missing_data.append(reason)

    def print_summary(self, input_path: Union[str, Path], output_path: Union[str, Path]):
        print("\n📊 Conversion Summary")
        print(f"   Input: {input_path}")
        print(f"   Output: {output_path}")
        print(f"   Total rows: {self.total_rows}")
        print(f"   Valid molecules: {self.valid_molecules}")
        print(f"   Invalid SMILES: {self.invalid_count}")
        print(f"   Missing data: {self.missing_count}")
        print(f"   Success rate: {self.success_rate:.1f}%")

        if self.invalid_count > 0:
            print("\n❌ Invalid SMILES found:")
            for i, (smiles, error) in enumerate(zip(self.invalid_smiles[:5], self.parsing_errors[:5])):
                print(f"   {i + 1}. '{smiles}' - {error}")
            if self.invalid_count > 5:
                print(f"   ... and {self.invalid_count - 5} more")

        if self.missing_count > 0:
            print("\n⚠️ Missing data issues:")
            for i, reason in enumerate(self.missing_data[:5]):
                print(f"   {i + 1}. {reason}")
            if self.missing_count > 5:
                print(f"   ... and {self.missing_count - 5} more")


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


def auto_detect_columns(df_or_header: Union[pd.DataFrame, List[str]]) -> Dict[str, Optional[str]]:
    """
    Auto-detect common column names for SMILES and activity data.

    Args:
        df_or_header: pandas DataFrame or list of column names

    Returns:
        Dictionary with detected column mappings
    """
    if PANDAS_AVAILABLE and isinstance(df_or_header, pd.DataFrame):
        columns = df_or_header.columns.tolist()
    else:
        columns = df_or_header

    # Convert to lowercase for matching
    lower_columns = [col.lower() for col in columns]

    # Common SMILES column names
    smiles_candidates = [
        "smiles",
        "smi",
        "canonical_smiles",
        "smiles_string",
        "molecule_smiles",
        "mol_smiles",
        "compound_smiles",
    ]

    # Common activity column names
    activity_candidates = [
        "activity",
        "property",
        "pic50",
        "pki",
        "pic",
        "kd",
        "ki",
        "ec50",
        "logp",
        "activity_value",
        "standard_value",
        "bioactivity",
        "target",
        "value",
        "measurement",
        "potency",
        "concentration",
        "label",
        "y",
        "target_value",
        "endpoint",
        "response",
    ]

    # Find SMILES column
    smiles_col = None
    for candidate in smiles_candidates:
        if candidate in lower_columns:
            smiles_col = columns[lower_columns.index(candidate)]
            break

    # Find activity column
    activity_col = None
    for candidate in activity_candidates:
        if candidate in lower_columns:
            activity_col = columns[lower_columns.index(candidate)]
            break

    # Look for additional common columns
    additional_cols = {}

    # Assay ID / ChEMBL ID
    assay_candidates = ["assay_id", "chembl_id", "target_id", "assay", "id", "compound_id"]
    for candidate in assay_candidates:
        if candidate in lower_columns:
            additional_cols["assay_id"] = columns[lower_columns.index(candidate)]
            break

    # Relation/operator (=, <, >, etc.)
    relation_candidates = ["relation", "operator", "standard_relation", "inequality"]
    for candidate in relation_candidates:
        if candidate in lower_columns:
            additional_cols["relation"] = columns[lower_columns.index(candidate)]
            break

    # Units
    unit_candidates = ["units", "standard_units", "unit", "activity_units"]
    for candidate in unit_candidates:
        if candidate in lower_columns:
            additional_cols["units"] = columns[lower_columns.index(candidate)]
            break

    # Assay type
    type_candidates = ["assay_type", "type", "bioassay_type", "activity_type"]
    for candidate in type_candidates:
        if candidate in lower_columns:
            additional_cols["assay_type"] = columns[lower_columns.index(candidate)]
            break

    result = {"smiles": smiles_col, "activity": activity_col, **additional_cols}

    return result


def safe_float_convert(value: Any) -> Optional[float]:
    """Safely convert a value to float, handling various edge cases."""
    if value is None or (isinstance(value, str) and value.strip() == ""):
        return None

    if PANDAS_AVAILABLE and pd.isna(value):
        return None

    try:
        if isinstance(value, (int, float)):
            if math.isnan(value) or math.isinf(value):
                return None
            return float(value)

        # Handle string conversion
        str_val = str(value).strip()
        if str_val.lower() in ["", "nan", "na", "null", "none", "-"]:
            return None

        # Try to parse as float
        return float(str_val)
    except (ValueError, TypeError):
        return None


def convert_csv_to_jsonl_gz(
    input_path: Path,
    output_path: Path,
    assay_id: str,
    smiles_column: str = "SMILES",
    activity_column: Optional[str] = None,
    auto_detect: bool = False,
    additional_columns: Optional[Dict[str, str]] = None,
    clean_smiles: bool = True,
    strict_validation: bool = True,
    dry_run: bool = False,
) -> ConversionStats:
    """
    Convert CSV file to JSONL.GZ format for THEMAP.

    Args:
        input_path: Path to input CSV file
        output_path: Path for output JSONL.GZ file
        assay_id: ChEMBL assay ID or dataset identifier
        smiles_column: Name of SMILES column
        activity_column: Name of activity/property column
        auto_detect: Whether to auto-detect column names
        additional_columns: Additional column mappings
        clean_smiles: Whether to validate and clean SMILES
        strict_validation: Whether to use strict SMILES validation
        dry_run: If True, only analyze without writing output

    Returns:
        ConversionStats object with conversion statistics
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    stats = ConversionStats()
    additional_columns = additional_columns or {}

    logging.info(f"Converting {input_path} to {output_path}")
    logging.info(f"Using assay ID: {assay_id}")

    try:
        if PANDAS_AVAILABLE:
            # Use pandas for better CSV handling
            df = pd.read_csv(input_path)

            # Auto-detect columns if requested
            if auto_detect:
                detected = auto_detect_columns(df)
                logging.info(f"Auto-detected columns: {detected}")

                # Use detected columns if not explicitly provided
                if detected["smiles"]:
                    smiles_column = detected["smiles"]
                    logging.info(f"Using SMILES column: {smiles_column}")

                if activity_column is None and detected["activity"]:
                    activity_column = detected["activity"]
                    logging.info(f"Using activity column: {activity_column}")

                # Add other detected columns
                for key, value in detected.items():
                    if key not in ["smiles", "activity"] and value:
                        additional_columns[key] = value

            if smiles_column not in df.columns:
                available_cols = ", ".join(df.columns.tolist())
                raise ValueError(f"SMILES column '{smiles_column}' not found. Available: {available_cols}")

            # Process each row
            if not dry_run:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                outfile = gzip.open(output_path, "wt", encoding="utf-8")

            try:
                for idx, row in df.iterrows():
                    smiles = row[smiles_column]

                    # Validate SMILES if requested
                    if clean_smiles:
                        is_valid, error_msg = validate_smiles(smiles, strict=strict_validation)
                        if not is_valid:
                            stats.add_invalid_molecule(str(smiles), error_msg)
                            continue

                    # Check for missing SMILES
                    if pd.isna(smiles) or str(smiles).strip() == "":
                        stats.add_missing_data("Missing SMILES")
                        continue

                    # Build JSON record
                    record = {"SMILES": str(smiles).strip(), "Assay_ID": assay_id}

                    # Add activity/property if available
                    if activity_column and activity_column in df.columns:
                        activity_value = safe_float_convert(row[activity_column])
                        if activity_value is not None:
                            record["Property"] = str(activity_value)
                            record["RegressionProperty"] = str(activity_value)
                            # Add log property if value is positive
                            if activity_value > 0:
                                record["LogRegressionProperty"] = str(math.log10(activity_value))
                        else:
                            record["Property"] = "0.0"  # Default value
                    else:
                        record["Property"] = "1.0"  # Default binary activity

                    # Add additional columns
                    for json_key, csv_col in additional_columns.items():
                        if csv_col in df.columns and not pd.isna(row[csv_col]):
                            value = row[csv_col]
                            if json_key == "relation":
                                record["Relation"] = str(value)
                            elif json_key == "assay_type":
                                record["AssayType"] = str(value)
                            elif json_key == "units":
                                record["Units"] = str(value)
                            else:
                                record[json_key] = str(value)

                    # Set default values for required fields
                    if "Relation" not in record:
                        record["Relation"] = "="
                    if "AssayType" not in record:
                        record["AssayType"] = "B"  # Binding assay

                    # Add placeholder arrays for computed features (will be computed by THEMAP)
                    record["fingerprints"] = []
                    record["descriptors"] = []
                    record["motifs"] = []
                    record["graph"] = {
                        "adjacency_lists": [],
                        "node_types": [],
                        "node_features": [],
                        "node_contexts": [],
                    }

                    # Write record
                    if not dry_run:
                        outfile.write(json.dumps(record) + "\n")

                    stats.add_valid_molecule()

            finally:
                if not dry_run and "outfile" in locals():
                    outfile.close()

        else:
            # Use standard CSV module
            with open(input_path, "r", encoding="utf-8") as infile:
                # Detect delimiter
                sniffer = csv.Sniffer()
                sample = infile.read(1024)
                infile.seek(0)
                delimiter = sniffer.sniff(sample).delimiter

                reader = csv.DictReader(infile, delimiter=delimiter)

                # Auto-detect columns if requested
                if auto_detect:
                    detected = auto_detect_columns(list(reader.fieldnames))
                    logging.info(f"Auto-detected columns: {detected}")

                    if detected["smiles"]:
                        smiles_column = detected["smiles"]
                        logging.info(f"Using SMILES column: {smiles_column}")

                    if activity_column is None and detected["activity"]:
                        activity_column = detected["activity"]
                        logging.info(f"Using activity column: {activity_column}")

                if smiles_column not in reader.fieldnames:
                    available_cols = ", ".join(reader.fieldnames)
                    raise ValueError(
                        f"SMILES column '{smiles_column}' not found. Available: {available_cols}"
                    )

                # Process rows
                if not dry_run:
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    outfile = gzip.open(output_path, "wt", encoding="utf-8")

                try:
                    for row in reader:
                        smiles = row.get(smiles_column, "").strip()

                        # Validate SMILES if requested
                        if clean_smiles:
                            is_valid, error_msg = validate_smiles(smiles, strict=strict_validation)
                            if not is_valid:
                                stats.add_invalid_molecule(smiles, error_msg)
                                continue

                        # Check for missing SMILES
                        if not smiles:
                            stats.add_missing_data("Missing SMILES")
                            continue

                        # Build JSON record
                        record = {"SMILES": smiles, "Assay_ID": assay_id, "Relation": "=", "AssayType": "B"}

                        # Add activity if available
                        if activity_column and activity_column in row:
                            activity_value = safe_float_convert(row[activity_column])
                            if activity_value is not None:
                                record["Property"] = str(activity_value)
                                record["RegressionProperty"] = str(activity_value)
                                if activity_value > 0:
                                    record["LogRegressionProperty"] = str(math.log10(activity_value))
                            else:
                                record["Property"] = "0.0"
                        else:
                            record["Property"] = "1.0"

                        # Add additional columns
                        for json_key, csv_col in additional_columns.items():
                            if csv_col in row and row[csv_col].strip():
                                record[json_key] = row[csv_col].strip()

                        # Add placeholder arrays for computed features
                        record["fingerprints"] = []
                        record["descriptors"] = []
                        record["motifs"] = []
                        record["graph"] = {
                            "adjacency_lists": [],
                            "node_types": [],
                            "node_features": [],
                            "node_contexts": [],
                        }

                        # Write record
                        if not dry_run:
                            outfile.write(json.dumps(record) + "\n")

                        stats.add_valid_molecule()

                finally:
                    if not dry_run and "outfile" in locals():
                        outfile.close()

    except Exception as e:
        logging.error(f"Error during conversion: {e}")
        raise

    return stats


def create_parser() -> argparse.ArgumentParser:
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Convert CSV files to JSONL.GZ format for THEMAP",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic conversion
  python scripts/csv_to_jsonl.py input.csv CHEMBL123456

  # Specify custom columns
  python scripts/csv_to_jsonl.py data.csv CHEMBL123456 --smiles-column SMILES --activity-column pIC50

  # Auto-detect column names
  python scripts/csv_to_jsonl.py chembl_data.csv CHEMBL123456 --auto-detect

  # Custom output location
  python scripts/csv_to_jsonl.py input.csv CHEMBL123456 --output datasets/train/CHEMBL123456.jsonl.gz

  # Dry run to analyze data
  python scripts/csv_to_jsonl.py input.csv CHEMBL123456 --dry-run

  # Skip SMILES validation (faster but less safe)
  python scripts/csv_to_jsonl.py input.csv CHEMBL123456 --no-clean-smiles

Output Structure:
  The generated JSONL.GZ files contain one JSON object per line with fields:
  - SMILES: Molecular structure
  - Property: Activity/property value (string)
  - Assay_ID: Dataset/assay identifier
  - RegressionProperty: Numeric activity value
  - LogRegressionProperty: Log10 of activity (if positive)
  - Relation: Relationship operator (=, <, >, etc.)
  - AssayType: Type of assay (B=binding, F=functional, etc.)
  - fingerprints, descriptors, motifs, graph: Placeholder arrays for computed features
        """,
    )

    parser.add_argument("input_csv", help="Input CSV file to convert")

    parser.add_argument("assay_id", help="ChEMBL assay ID or dataset identifier (e.g., CHEMBL123456)")

    parser.add_argument(
        "--output", "-o", help="Output JSONL.GZ file path (default: assay_id.jsonl.gz in current directory)"
    )

    parser.add_argument(
        "--smiles-column", default="SMILES", help="Name of SMILES column in CSV (default: SMILES)"
    )

    parser.add_argument("--activity-column", help="Name of activity/property column in CSV")

    parser.add_argument(
        "--auto-detect", action="store_true", help="Auto-detect common column names (SMILES, activity, etc.)"
    )

    parser.add_argument(
        "--additional-columns",
        help='Additional column mappings as JSON (e.g., {"relation": "standard_relation"})',
    )

    parser.add_argument(
        "--no-clean-smiles",
        action="store_true",
        help="Skip SMILES validation and cleaning (faster but less safe)",
    )

    parser.add_argument(
        "--lenient", action="store_true", help="Use less strict SMILES validation (skip sanitization)"
    )

    parser.add_argument("--dry-run", action="store_true", help="Analyze input file without creating output")

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

    # Setup logging
    log_level = getattr(logging, args.log_level.upper())
    if args.quiet:
        log_level = logging.WARNING

    logging.basicConfig(level=log_level, format="%(levelname)s: %(message)s")

    input_path = Path(args.input_csv)
    if not input_path.exists():
        print(f"❌ Error: Input file not found: {input_path}")
        return 1

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path(f"{args.assay_id}.jsonl.gz")

    # Parse additional columns
    additional_columns = {}
    if args.additional_columns:
        try:
            additional_columns = json.loads(args.additional_columns)
        except json.JSONDecodeError as e:
            print(f"❌ Error: Invalid JSON in --additional-columns: {e}")
            return 1

    try:
        # Convert CSV to JSONL.GZ
        stats = convert_csv_to_jsonl_gz(
            input_path=input_path,
            output_path=output_path,
            assay_id=args.assay_id,
            smiles_column=args.smiles_column,
            activity_column=args.activity_column,
            auto_detect=args.auto_detect,
            additional_columns=additional_columns,
            clean_smiles=not args.no_clean_smiles,
            strict_validation=not args.lenient,
            dry_run=args.dry_run,
        )

        # Print summary
        stats.print_summary(input_path, output_path if not args.dry_run else "N/A (dry run)")

        if not args.dry_run and stats.valid_molecules > 0:
            print(f"\n✅ Successfully created {output_path}")
            print("   Ready for use with THEMAP pipeline!")
        elif args.dry_run:
            print(f"\n📋 Dry run completed. Use without --dry-run to create {output_path}")
        else:
            print("\n❌ No valid molecules found. Check your input data and column mappings.")
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
