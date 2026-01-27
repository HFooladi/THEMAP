"""
CSV to JSONL.GZ conversion utilities for THEMAP.

This module provides programmatic CSV to JSONL.GZ conversion with
RDKit-based SMILES validation, designed for use by the DatasetLoader.
"""

import csv
import gzip
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from ..utils.logging import get_logger

logger = get_logger(__name__)

# Check for RDKit
try:
    from rdkit import Chem, RDLogger
    from rdkit.Chem import SanitizeMol

    RDLogger.DisableLog("rdApp.*")
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    logger.warning("RDKit not available. SMILES validation will be limited.")

# Check for pandas
try:
    import pandas as pd

    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False


@dataclass
class ConversionStats:
    """Statistics from CSV to JSONL conversion."""

    total_rows: int = 0
    valid_molecules: int = 0
    invalid_smiles: List[str] = field(default_factory=list)
    parsing_errors: List[str] = field(default_factory=list)
    missing_data: List[str] = field(default_factory=list)

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


def validate_smiles(smiles: str, strict: bool = True) -> Tuple[bool, Optional[str]]:
    """Validate a SMILES string using RDKit.

    Args:
        smiles: SMILES string to validate.
        strict: If True, perform full sanitization checks.

    Returns:
        Tuple of (is_valid, error_message).
    """
    if not smiles or not isinstance(smiles, str):
        return False, "Empty or non-string SMILES"

    if PANDAS_AVAILABLE:
        try:
            if pd.isna(smiles):
                return False, "NaN SMILES value"
        except (TypeError, ValueError):
            # pd.isna() can raise on certain input types; continue with validation
            pass

    if not RDKIT_AVAILABLE:
        # Basic validation without RDKit
        return len(smiles.strip()) > 0, None

    try:
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        if mol is None:
            return False, "RDKit cannot parse this SMILES"

        if strict:
            try:
                SanitizeMol(mol)
            except Exception as e:
                return False, f"Sanitization failed: {str(e)}"

        if mol.GetNumAtoms() == 0:
            return False, "Molecule has no atoms"

        return True, None

    except Exception as e:
        return False, f"Validation error: {str(e)}"


def auto_detect_columns(columns: List[str]) -> Dict[str, Optional[str]]:
    """Auto-detect common column names for SMILES and activity data.

    Args:
        columns: List of column names from the CSV.

    Returns:
        Dictionary with detected column mappings.
    """
    lower_columns = [col.lower() for col in columns]

    # SMILES column candidates
    smiles_candidates = [
        "smiles",
        "smi",
        "canonical_smiles",
        "smiles_string",
        "molecule_smiles",
        "mol_smiles",
        "compound_smiles",
    ]

    # Activity column candidates
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
        "label",
        "y",
        "target_value",
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

    return {"smiles": smiles_col, "activity": activity_col}


def safe_float_convert(value: Any) -> Optional[float]:
    """Safely convert a value to float."""
    if value is None:
        return None

    if isinstance(value, str) and value.strip() == "":
        return None

    if PANDAS_AVAILABLE:
        try:
            if pd.isna(value):
                return None
        except (TypeError, ValueError):
            # pd.isna() can raise on certain input types; continue with conversion
            pass

    try:
        if isinstance(value, (int, float)):
            if math.isnan(value) or math.isinf(value):
                return None
            return float(value)

        str_val = str(value).strip()
        if str_val.lower() in ["", "nan", "na", "null", "none", "-"]:
            return None

        return float(str_val)
    except (ValueError, TypeError):
        return None


class CSVConverter:
    """Convert CSV files to JSONL.GZ format for THEMAP.

    Supports auto-detection of SMILES and activity columns,
    RDKit-based SMILES validation, and various CSV formats.

    Examples:
        >>> converter = CSVConverter()
        >>> stats = converter.convert("input.csv", "output.jsonl.gz", "CHEMBL123")
        >>> print(f"Converted {stats.valid_molecules} molecules")
    """

    def __init__(
        self,
        validate_smiles: bool = True,
        strict_validation: bool = True,
        auto_detect_columns: bool = True,
    ):
        """Initialize the converter.

        Args:
            validate_smiles: Whether to validate SMILES with RDKit.
            strict_validation: If True, use strict sanitization.
            auto_detect_columns: If True, auto-detect column names.
        """
        self.validate_smiles_flag = validate_smiles
        self.strict_validation = strict_validation
        self.auto_detect_columns_flag = auto_detect_columns

    def read_csv(
        self,
        path: Union[str, Path],
        smiles_column: Optional[str] = None,
        activity_column: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Read CSV file and extract SMILES and labels.

        Args:
            path: Path to the CSV file.
            smiles_column: Name of the SMILES column (auto-detected if None).
            activity_column: Name of the activity column (auto-detected if None).

        Returns:
            Dictionary with 'smiles', 'labels', 'numeric_labels' keys.
        """
        path = Path(path)

        if PANDAS_AVAILABLE:
            df = pd.read_csv(path)
            columns = df.columns.tolist()
        else:
            with open(path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                columns = reader.fieldnames or []
                rows = list(reader)
            df = None

        # Auto-detect columns if needed
        if self.auto_detect_columns_flag:
            detected = auto_detect_columns(columns)
            if smiles_column is None and detected["smiles"]:
                smiles_column = detected["smiles"]
                logger.info(f"Auto-detected SMILES column: {smiles_column}")
            if activity_column is None and detected["activity"]:
                activity_column = detected["activity"]
                logger.info(f"Auto-detected activity column: {activity_column}")

        if smiles_column is None:
            raise ValueError(f"Could not detect SMILES column. Available columns: {columns}")

        # Extract data
        smiles_list = []
        labels = []
        numeric_labels = []

        if PANDAS_AVAILABLE and df is not None:
            for _, row in df.iterrows():
                smiles = row[smiles_column]

                # Validate SMILES
                if self.validate_smiles_flag:
                    is_valid, _ = validate_smiles(str(smiles), strict=self.strict_validation)
                    if not is_valid:
                        continue

                if pd.isna(smiles) or str(smiles).strip() == "":
                    continue

                smiles_list.append(str(smiles).strip())

                # Get activity value
                if activity_column and activity_column in df.columns:
                    activity_value = safe_float_convert(row[activity_column])
                    if activity_value is not None:
                        # Convert to binary label (default threshold: 0 = positive)
                        labels.append(1 if activity_value > 0 else 0)
                        numeric_labels.append(activity_value)
                    else:
                        labels.append(1)  # Default to positive
                        numeric_labels.append(None)
                else:
                    labels.append(1)
                    numeric_labels.append(None)
        else:
            for row in rows:
                smiles = row.get(smiles_column, "").strip()

                if self.validate_smiles_flag:
                    is_valid, _ = validate_smiles(smiles, strict=self.strict_validation)
                    if not is_valid:
                        continue

                if not smiles:
                    continue

                smiles_list.append(smiles)

                if activity_column and activity_column in row:
                    activity_value = safe_float_convert(row[activity_column])
                    if activity_value is not None:
                        labels.append(1 if activity_value > 0 else 0)
                        numeric_labels.append(activity_value)
                    else:
                        labels.append(1)
                        numeric_labels.append(None)
                else:
                    labels.append(1)
                    numeric_labels.append(None)

        return {
            "smiles": smiles_list,
            "labels": labels,
            "numeric_labels": numeric_labels if any(v is not None for v in numeric_labels) else None,
        }

    def convert(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        task_id: str,
        smiles_column: Optional[str] = None,
        activity_column: Optional[str] = None,
    ) -> ConversionStats:
        """Convert CSV file to JSONL.GZ format.

        Args:
            input_path: Path to input CSV file.
            output_path: Path for output JSONL.GZ file.
            task_id: Task/assay ID for the dataset.
            smiles_column: Name of SMILES column (auto-detected if None).
            activity_column: Name of activity column (auto-detected if None).

        Returns:
            ConversionStats with conversion statistics.
        """
        input_path = Path(input_path)
        output_path = Path(output_path)

        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        stats = ConversionStats()
        logger.info(f"Converting {input_path} to {output_path}")

        # Read and detect columns
        if PANDAS_AVAILABLE:
            df = pd.read_csv(input_path)
            columns = df.columns.tolist()
        else:
            with open(input_path, "r", encoding="utf-8") as f:
                sniffer = csv.Sniffer()
                sample = f.read(1024)
                f.seek(0)
                dialect = sniffer.sniff(sample)
                reader = csv.DictReader(f, dialect=dialect)
                columns = reader.fieldnames or []
                rows = list(reader)
            df = None

        # Auto-detect columns
        if self.auto_detect_columns_flag:
            detected = auto_detect_columns(columns)
            if smiles_column is None and detected["smiles"]:
                smiles_column = detected["smiles"]
            if activity_column is None and detected["activity"]:
                activity_column = detected["activity"]

        if smiles_column is None or smiles_column not in columns:
            raise ValueError(f"SMILES column '{smiles_column}' not found. Available: {columns}")

        # Create output directory
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Process and write
        with gzip.open(output_path, "wt", encoding="utf-8") as outfile:
            if PANDAS_AVAILABLE and df is not None:
                for _, row in df.iterrows():
                    stats.total_rows += 1
                    record = self._process_row(row, smiles_column, activity_column, task_id, stats)
                    if record:
                        outfile.write(json.dumps(record) + "\n")
            else:
                for row in rows:
                    stats.total_rows += 1
                    record = self._process_row_dict(row, smiles_column, activity_column, task_id, stats)
                    if record:
                        outfile.write(json.dumps(record) + "\n")

        logger.info(
            f"Converted {stats.valid_molecules}/{stats.total_rows} molecules "
            f"({stats.success_rate:.1f}% success rate)"
        )

        return stats

    def _process_row(
        self,
        row: "pd.Series",
        smiles_column: str,
        activity_column: Optional[str],
        task_id: str,
        stats: ConversionStats,
    ) -> Optional[Dict[str, Any]]:
        """Process a pandas DataFrame row."""
        smiles = row[smiles_column]

        if PANDAS_AVAILABLE and pd.isna(smiles):
            stats.missing_data.append("Missing SMILES")
            return None

        smiles = str(smiles).strip()
        if not smiles:
            stats.missing_data.append("Empty SMILES")
            return None

        # Validate SMILES
        if self.validate_smiles_flag:
            is_valid, error_msg = validate_smiles(smiles, strict=self.strict_validation)
            if not is_valid:
                stats.invalid_smiles.append(smiles)
                stats.parsing_errors.append(error_msg or "Unknown error")
                return None

        # Build record
        record = {
            "SMILES": smiles,
            "Assay_ID": task_id,
            "Relation": "=",
            "AssayType": "B",
        }

        # Add activity
        if activity_column and activity_column in row.index:
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

        # Add placeholder arrays
        record["fingerprints"] = []
        record["descriptors"] = []
        record["motifs"] = []
        record["graph"] = {
            "adjacency_lists": [],
            "node_types": [],
            "node_features": [],
            "node_contexts": [],
        }

        stats.valid_molecules += 1
        return record

    def _process_row_dict(
        self,
        row: Dict[str, str],
        smiles_column: str,
        activity_column: Optional[str],
        task_id: str,
        stats: ConversionStats,
    ) -> Optional[Dict[str, Any]]:
        """Process a dictionary row (from csv.DictReader)."""
        smiles = row.get(smiles_column, "").strip()

        if not smiles:
            stats.missing_data.append("Missing SMILES")
            return None

        # Validate SMILES
        if self.validate_smiles_flag:
            is_valid, error_msg = validate_smiles(smiles, strict=self.strict_validation)
            if not is_valid:
                stats.invalid_smiles.append(smiles)
                stats.parsing_errors.append(error_msg or "Unknown error")
                return None

        # Build record
        record = {
            "SMILES": smiles,
            "Assay_ID": task_id,
            "Relation": "=",
            "AssayType": "B",
        }

        # Add activity
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

        # Add placeholder arrays
        record["fingerprints"] = []
        record["descriptors"] = []
        record["motifs"] = []
        record["graph"] = {
            "adjacency_lists": [],
            "node_types": [],
            "node_features": [],
            "node_contexts": [],
        }

        stats.valid_molecules += 1
        return record


def convert_csv_to_jsonl(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    task_id: str,
    smiles_column: Optional[str] = None,
    activity_column: Optional[str] = None,
    validate: bool = True,
) -> ConversionStats:
    """Convenience function to convert CSV to JSONL.GZ.

    Args:
        input_path: Path to input CSV file.
        output_path: Path for output JSONL.GZ file.
        task_id: Task/assay ID for the dataset.
        smiles_column: Name of SMILES column (auto-detected if None).
        activity_column: Name of activity column (auto-detected if None).
        validate: Whether to validate SMILES with RDKit.

    Returns:
        ConversionStats with conversion statistics.
    """
    converter = CSVConverter(validate_smiles=validate)
    return converter.convert(input_path, output_path, task_id, smiles_column, activity_column)
