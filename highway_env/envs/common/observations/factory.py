"""Observation factory for highway-env observation configurations."""

from __future__ import annotations

from .base import ObservationType
from .camera import (
    LaneCameraObservation,
    LidarCameraObservations,
    SharedMultiAgentLidarCameraObservations,
)
from .classic import (
    AttributesObservation,
    ExitObservation,
    GrayscaleObservation,
    KinematicObservation,
    KinematicsGoalObservation,
    MultiAgentObservation,
    OccupancyGridObservation,
    TimeToCollisionObservation,
    TupleObservation,
)
from .lidar import LidarObservation


def observation_factory(env: AbstractEnv, config: dict) -> ObservationType:
    """Build the configured observation object for an environment."""
    if config["type"] == "TimeToCollision":
        return TimeToCollisionObservation(env, **config)
    elif config["type"] == "Kinematics":
        return KinematicObservation(env, **config)
    elif config["type"] == "OccupancyGrid":
        return OccupancyGridObservation(env, **config)
    elif config["type"] == "KinematicsGoal":
        return KinematicsGoalObservation(env, **config)
    elif config["type"] == "GrayscaleObservation":
        return GrayscaleObservation(env, **config)
    elif config["type"] == "AttributesObservation":
        return AttributesObservation(env, **config)
    elif (
        config["type"] == "MultiAgentObservation"
        and isinstance(config.get("observation_config"), dict)
        and config["observation_config"].get("type") == "LidarCameraObservations"
    ):
        shared_cfg = config["observation_config"]
        return SharedMultiAgentLidarCameraObservations(
            env,
            lidar=shared_cfg.get("lidar"),
            camera=shared_cfg.get("camera"),
        )
    elif config["type"] == "MultiAgentObservation":
        return MultiAgentObservation(env, **config)
    elif config["type"] == "SharedMultiAgentLidarCameraObservations":
        return SharedMultiAgentLidarCameraObservations(env, **config)
    elif config["type"] == "TupleObservation":
        return TupleObservation(env, **config)
    elif config["type"] == "LidarObservation":
        return LidarObservation(env, **config)
    elif config["type"] == "LaneCameraObservation":
        return LaneCameraObservation(env, **config)
    elif config["type"] == "LidarCameraObservation":
        return LaneCameraObservation(env, **config)
    elif config["type"] == "LidarCameraObservations":
        return LidarCameraObservations(env, **config)
    elif config["type"] == "ExitObservation":
        return ExitObservation(env, **config)
    else:
        raise ValueError("Unknown observation type")


__all__ = ["observation_factory"]
