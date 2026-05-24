"""Compatibility facade for highway-env observation implementations.

Observation classes now live under :mod:`highway_env.envs.common.observations`,
while this module preserves the historical import path used by environments,
scripts, tests, and documentation.
"""

from highway_env.envs.common.observations import *
from highway_env.envs.common.observations import __all__
