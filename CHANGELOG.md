# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [v0.3.0] - 2025-08-19
### Added
- **Pipeline Infrastructure**: Complete configuration-driven pipeline system for distance computation workflows
  - `run_pipeline.py` - Main pipeline runner with CLI interface
  - `themap.pipeline` module with CLI, configuration management, and execution engine
  - Support for both directory-based dataset discovery and explicit dataset specification
  - YAML/JSON configuration files with validation and comprehensive examples

- **Distance Computation Fixes**: Corrected dataset-level distance computation methodology
  - Fixed Euclidean and Cosine distance implementations to work properly with variable-sized datasets
  - Implemented proper pairwise distance computation between individual molecules across datasets
  - Fixed method naming inconsistencies (`compute_distance()` → `get_distance()`)

- **Utility Scripts**: New data processing and conversion tools
  - `scripts/csv_to_jsonl.py` - Convert CSV files to THEMAP's native JSONL.GZ format with SMILES validation
  - `scripts/clean_smiles.py` - SMILES validation and cleanup utility for datasets

- **Examples Reorganization**: Structured example system by complexity level
  - `examples/basic/` - Introductory examples for new users
  - `examples/distance/` - Distance computation workflows
  - `examples/advanced/` - Complex research applications
  - Comprehensive example configurations in `configs/examples/`

- **Enhanced Documentation**: Pipeline usage guides and workflow documentation
  - `docs/PIPELINE_GUIDE.md` - Complete pipeline usage documentation
  - `docs/README_PIPELINE_SCRIPTS.md` - Utility scripts documentation

- **New Test Coverage**: Comprehensive test suite for new functionality
  - Unit tests for distance computation modules
  - Pipeline component tests
  - Configuration validation tests

### Fixed
- **Critical Distance Computation Bug**: Resolved incorrect Euclidean/Cosine distance implementation
  - Previous implementation incorrectly flattened entire feature matrices from different-sized datasets
  - Now correctly computes pairwise distances between individual molecules across datasets
  - Uses mean of pairwise distances as dataset-level distance metric

- **Pipeline Method Calls**: Fixed incorrect method names in pipeline execution
  - Updated all distance computation calls to use correct `get_distance()` method
  - Fixed TaskDistance instantiation and usage patterns

- **Dataset Loading**: Fixed explicit dataset specification mode
  - Individual dataset loading now creates proper Task objects for distance computation
  - Both directory-based and explicit dataset modes now work correctly

### Changed
- **Examples Structure**: Moved examples from `scripts/` to organized `examples/` directory
- **Distance Module**: Refactored distance computation classes for better consistency and correctness
- **Configuration System**: Enhanced pipeline configuration with better validation and error handling

### Removed
- Deprecated example scripts from `scripts/` directory (moved to `examples/`)
- Legacy `tasks_distance.py` module (functionality integrated into new distance classes)

## [v0.2.0] - YYYY-MM-DD
### Added
- Initial changelog entry. Describe new features, changes, and fixes here.

## [v0.1.0] - YYYY-MM-DD
### Added
- Initial release.
