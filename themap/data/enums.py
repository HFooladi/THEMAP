"""
Enums used throughout THEMAP.

This module contains enums that define constants used across the codebase.
"""

from enum import IntEnum


class DataFold(IntEnum):
    """Enum for data fold types.

    This enum represents the different data splits used in machine learning:
    - TRAIN (0): Training/source tasks
    - VALIDATION (1): Validation/development tasks
    - TEST (2): Test/target tasks

    By inheriting from IntEnum, each fold type is assigned an integer value
    which allows for easy indexing and comparison operations.
    """

    TRAIN = 0
    VALIDATION = 1
    TEST = 2
