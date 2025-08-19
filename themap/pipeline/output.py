"""
Output management for pipeline results.

This module handles saving and formatting pipeline results in various formats
including JSON, CSV, Parquet, and pickle files.
"""

import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


class OutputManager:
    """Manages pipeline output in multiple formats."""

    def __init__(self, output_config):
        """Initialize output manager with configuration."""
        from .config import OutputConfig

        self.config: OutputConfig = output_config
        self.output_dir = Path(self.config.directory)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Track all generated files
        self.generated_files: List[Path] = []

    def save_results(self, results: Dict[str, Any], filename_prefix: str = "results") -> Dict[str, Path]:
        """
        Save results in all configured formats.

        Args:
            results: Dictionary containing pipeline results
            filename_prefix: Prefix for output filenames

        Returns:
            Dictionary mapping format names to file paths
        """
        saved_files = {}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        for fmt in self.config.formats:
            filename = f"{filename_prefix}_{timestamp}.{fmt}"
            filepath = self.output_dir / filename

            if fmt == "json":
                saved_files["json"] = self._save_json(results, filepath)
            elif fmt == "csv":
                saved_files["csv"] = self._save_csv(results, filepath)
            elif fmt == "parquet":
                saved_files["parquet"] = self._save_parquet(results, filepath)
            elif fmt == "pickle":
                saved_files["pickle"] = self._save_pickle(results, filepath)
            else:
                raise ValueError(f"Unsupported format: {fmt}")

        return saved_files

    def save_distance_matrix(
        self,
        matrix: np.ndarray,
        row_labels: List[str],
        col_labels: List[str],
        filename_prefix: str = "distance_matrix",
    ) -> Dict[str, Path]:
        """
        Save distance matrix with labels.

        Args:
            matrix: Distance matrix
            row_labels: Labels for rows (source datasets)
            col_labels: Labels for columns (target datasets)
            filename_prefix: Prefix for output filenames

        Returns:
            Dictionary mapping format names to file paths
        """
        if not self.config.save_matrices:
            return {}

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        saved_files = {}

        # Create DataFrame for easier handling
        df = pd.DataFrame(matrix, index=row_labels, columns=col_labels)

        for fmt in self.config.formats:
            if fmt == "json":
                # Convert to nested dict for JSON
                matrix_dict = {
                    "matrix": df.to_dict(),
                    "shape": matrix.shape,
                    "row_labels": row_labels,
                    "col_labels": col_labels,
                    "timestamp": timestamp,
                }
                filename = f"{filename_prefix}_{timestamp}.json"
                filepath = self.output_dir / filename
                saved_files["json"] = self._save_json(matrix_dict, filepath)

            elif fmt == "csv":
                filename = f"{filename_prefix}_{timestamp}.csv"
                filepath = self.output_dir / filename
                df.to_csv(filepath)
                saved_files["csv"] = filepath
                self.generated_files.append(filepath)

            elif fmt == "parquet":
                filename = f"{filename_prefix}_{timestamp}.parquet"
                filepath = self.output_dir / filename
                df.to_parquet(filepath)
                saved_files["parquet"] = filepath
                self.generated_files.append(filepath)

            elif fmt == "pickle":
                matrix_data = {
                    "matrix": matrix,
                    "row_labels": row_labels,
                    "col_labels": col_labels,
                    "timestamp": timestamp,
                }
                filename = f"{filename_prefix}_{timestamp}.pkl"
                filepath = self.output_dir / filename
                saved_files["pickle"] = self._save_pickle(matrix_data, filepath)

        return saved_files

    def save_intermediate_results(
        self, results: Dict[str, Any], stage: str, dataset_id: str = ""
    ) -> Optional[Path]:
        """
        Save intermediate results if configured.

        Args:
            results: Intermediate results to save
            stage: Processing stage name
            dataset_id: Optional dataset identifier

        Returns:
            Path to saved file if saved, None otherwise
        """
        if not self.config.save_intermediate:
            return None

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix = f"_{dataset_id}" if dataset_id else ""
        filename = f"intermediate_{stage}{suffix}_{timestamp}.json"
        filepath = self.output_dir / filename

        return self._save_json(results, filepath)

    def create_summary_report(self, pipeline_results: Dict[str, Any]) -> Path:
        """
        Create a summary report of pipeline execution.

        Args:
            pipeline_results: Complete pipeline results

        Returns:
            Path to summary report file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        summary = {
            "pipeline_name": pipeline_results.get("config", {}).get("name", "Unknown"),
            "execution_timestamp": timestamp,
            "total_runtime_seconds": pipeline_results.get("runtime_seconds", 0),
            "datasets_processed": self._extract_dataset_summary(pipeline_results),
            "distance_computations": self._extract_distance_summary(pipeline_results),
            "files_generated": [str(f) for f in self.generated_files],
            "config": pipeline_results.get("config", {}),
            "errors": pipeline_results.get("errors", []),
        }

        filename = f"pipeline_summary_{timestamp}.json"
        filepath = self.output_dir / filename

        return self._save_json(summary, filepath)

    def _save_json(self, data: Dict[str, Any], filepath: Path) -> Path:
        """Save data as JSON with proper encoding."""

        def json_serializer(obj):
            """Custom JSON serializer for numpy arrays and other types."""
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, Path):
                return str(obj)
            elif hasattr(obj, "__dict__"):
                return obj.__dict__
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, default=json_serializer)

        self.generated_files.append(filepath)
        return filepath

    def _save_csv(self, data: Dict[str, Any], filepath: Path) -> Path:
        """Save data as CSV, converting nested structures to DataFrame."""
        # Try to convert to DataFrame
        try:
            if "distance_results" in data:
                # Extract distance results for CSV format
                distance_data = []
                for result in data["distance_results"]:
                    row = {
                        "source_dataset": result.get("source_dataset", ""),
                        "target_dataset": result.get("target_dataset", ""),
                        "modality": result.get("modality", ""),
                        "method": result.get("method", ""),
                        "distance": result.get("distance", 0),
                        "computation_time": result.get("computation_time", 0),
                    }
                    distance_data.append(row)

                df = pd.DataFrame(distance_data)
            else:
                # Generic flattening for other data types
                df = pd.json_normalize(data)

            df.to_csv(filepath, index=False)
        except Exception:
            # Fallback: save as JSON if CSV conversion fails
            json_path = filepath.with_suffix(".json")
            return self._save_json(data, json_path)

        self.generated_files.append(filepath)
        return filepath

    def _save_parquet(self, data: Dict[str, Any], filepath: Path) -> Path:
        """Save data as Parquet format."""
        try:
            # Convert to DataFrame first (similar to CSV logic)
            if "distance_results" in data:
                distance_data = []
                for result in data["distance_results"]:
                    row = {
                        "source_dataset": result.get("source_dataset", ""),
                        "target_dataset": result.get("target_dataset", ""),
                        "modality": result.get("modality", ""),
                        "method": result.get("method", ""),
                        "distance": result.get("distance", 0),
                        "computation_time": result.get("computation_time", 0),
                    }
                    distance_data.append(row)

                df = pd.DataFrame(distance_data)
            else:
                df = pd.json_normalize(data)

            df.to_parquet(filepath, index=False)
        except Exception:
            # Fallback: save as pickle if parquet conversion fails
            pickle_path = filepath.with_suffix(".pkl")
            return self._save_pickle(data, pickle_path)

        self.generated_files.append(filepath)
        return filepath

    def _save_pickle(self, data: Any, filepath: Path) -> Path:
        """Save data as pickle file."""
        with open(filepath, "wb") as f:
            pickle.dump(data, f)

        self.generated_files.append(filepath)
        return filepath

    def _extract_dataset_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract summary of datasets processed."""
        summary = {"total_datasets": 0, "molecule_datasets": 0, "protein_datasets": 0, "metadata_datasets": 0}

        if "datasets_info" in results:
            for dataset_info in results["datasets_info"]:
                summary["total_datasets"] += 1
                if "molecule" in dataset_info.get("modalities", []):
                    summary["molecule_datasets"] += 1
                if "protein" in dataset_info.get("modalities", []):
                    summary["protein_datasets"] += 1
                if "metadata" in dataset_info.get("modalities", []):
                    summary["metadata_datasets"] += 1

        return summary

    def _extract_distance_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract summary of distance computations."""
        summary = {"total_computations": 0, "by_modality": {}, "by_method": {}, "average_computation_time": 0}

        if "distance_results" in results:
            total_time = 0
            for result in results["distance_results"]:
                summary["total_computations"] += 1

                modality = result.get("modality", "unknown")
                method = result.get("method", "unknown")
                comp_time = result.get("computation_time", 0)

                summary["by_modality"][modality] = summary["by_modality"].get(modality, 0) + 1
                summary["by_method"][method] = summary["by_method"].get(method, 0) + 1
                total_time += comp_time

            if summary["total_computations"] > 0:
                summary["average_computation_time"] = total_time / summary["total_computations"]

        return summary
