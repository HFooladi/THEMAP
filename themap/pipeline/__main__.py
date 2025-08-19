"""
Main entry point for running the pipeline module as a script.

Usage:
    python -m themap.pipeline config.yaml
    python -m themap.pipeline --list-examples
"""

from .cli import main

if __name__ == "__main__":
    main()
