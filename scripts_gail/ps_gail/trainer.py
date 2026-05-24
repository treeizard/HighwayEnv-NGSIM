"""Compatibility facade for modular PS-GAIL training helpers.

The implementation lives in :mod:`scripts_gail.ps_gail.training`; this module
keeps the historical import path stable for training scripts and tests.
"""

from .training import *
from .training import __all__
