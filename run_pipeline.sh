#!/bin/bash
# THEMAP Pipeline Runner Script
#
# This script provides a convenient way to run the THEMAP pipeline with configuration files.
# It automatically activates the conda environment and runs the pipeline with proper error handling.

set -e  # Exit on any error

# Script configuration
CONDA_ENV="themap"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_DATA_PATH="datasets"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_usage() {
    cat << EOF
THEMAP Pipeline Runner

Usage: $0 [OPTIONS] <config_file>

ARGUMENTS:
    config_file          Path to pipeline configuration file (YAML or JSON)

OPTIONS:
    --data-path PATH     Base path to dataset files (default: datasets)
    --output-dir PATH    Override output directory from config
    --log-level LEVEL    Logging level: DEBUG, INFO, WARNING, ERROR (default: INFO)
    --sample-size N      Override sample size from config (useful for testing)
    --max-workers N      Override number of workers from config
    --validate-only      Only validate configuration without running
    --dry-run           Show what would be computed without running
    --list-examples     List available example configurations
    --help              Show this help message

EXAMPLES:
    # Run a simple pipeline
    $0 configs/examples/simple_directory_discovery.yaml

    # Run with custom data path and output directory
    $0 --data-path /path/to/my/data --output-dir results/my_run configs/my_config.yaml

    # Test with small sample size and debug logging
    $0 --sample-size 100 --log-level DEBUG configs/examples/comprehensive_multimodal.yaml

    # Validate configuration without running
    $0 --validate-only configs/my_config.yaml

    # See what would be computed
    $0 --dry-run configs/examples/directory_based_discovery.yaml

    # List available examples
    $0 --list-examples

ENVIRONMENT:
    The script automatically activates the '$CONDA_ENV' conda environment.
    Make sure you have installed THEMAP in this environment.

EOF
}

# Check if conda is available
check_conda() {
    if ! command -v conda &> /dev/null; then
        print_error "Conda is not available. Please install Anaconda/Miniconda."
        exit 1
    fi
}

# Check if conda environment exists
check_environment() {
    if ! conda env list | grep -q "^${CONDA_ENV} "; then
        print_error "Conda environment '${CONDA_ENV}' not found."
        print_info "Please create the environment or modify the CONDA_ENV variable in this script."
        exit 1
    fi
}

# Activate conda environment and run pipeline
run_pipeline() {
    local args=("$@")

    print_info "Activating conda environment: ${CONDA_ENV}"

    # Use conda run to execute in the environment
    if command -v conda &> /dev/null; then
        print_info "Running THEMAP pipeline..."
        conda run -n "${CONDA_ENV}" python -m themap.pipeline "${args[@]}"
    else
        print_error "Failed to activate conda environment"
        exit 1
    fi
}

# Main script logic
main() {
    # Handle help and special cases
    if [[ $# -eq 0 ]] || [[ "$1" == "--help" ]] || [[ "$1" == "-h" ]]; then
        print_usage
        exit 0
    fi

    # Check prerequisites
    check_conda
    check_environment

    # Change to script directory to ensure relative paths work
    cd "${SCRIPT_DIR}"

    # Handle list-examples specially
    if [[ "$1" == "--list-examples" ]]; then
        print_info "Available example configurations:"
        run_pipeline --list-examples
        exit 0
    fi

    # Validate that at least one argument (config file) is provided
    local config_file=""
    local has_config=false

    # Look for config file in arguments (not starting with --)
    for arg in "$@"; do
        if [[ ! "$arg" =~ ^-- ]] && [[ "$arg" != --* ]]; then
            config_file="$arg"
            has_config=true
            break
        fi
    done

    if [[ "$has_config" == false ]]; then
        print_error "No configuration file specified."
        print_usage
        exit 1
    fi

    # Check if config file exists
    if [[ ! -f "$config_file" ]]; then
        print_error "Configuration file not found: $config_file"
        exit 1
    fi

    print_info "Using configuration: $config_file"

    # Run the pipeline with all arguments
    run_pipeline "$@"

    local exit_code=$?

    if [[ $exit_code -eq 0 ]]; then
        print_success "Pipeline completed successfully!"
    else
        print_error "Pipeline failed with exit code: $exit_code"
        exit $exit_code
    fi
}

# Run main function with all arguments
main "$@"
