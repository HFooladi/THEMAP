"""
This is the OTTD model, based on the following implementation:
https://github.com/microsoft/otdd.

If you use this model, please cite the corresponding paper:
@article{alvarez2020geometric,
  title={Geometric dataset distances via optimal transport},
  author={Alvarez-Melis, David and Fusi, Nicolo},
  journal={Advances in Neural Information Processing Systems},
  volume={33},
  pages={21428--21439},
  year={2020}
}

"""

import logging
import os
from os.path import abspath, dirname

# Defaults
ROOT_DIR = dirname(dirname(abspath(__file__)))  # Project Root
HOME_DIR = os.getenv("HOME")  # User home dir
DATA_DIR = os.path.join(ROOT_DIR, "data")
OUTPUT_DIR = os.path.join(ROOT_DIR, "out")
MODELS_DIR = os.path.join(ROOT_DIR, "models")
from .utils import launch_logger
