"""
Geometry module for defining domains.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple, List, Union, Callable

import jax
import jax.numpy as jnp
