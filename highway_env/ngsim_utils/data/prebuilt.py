"""Load, select, and prepare prebuilt NGSIM episode data."""

# Modified by: Yide Tao (yide.tao@monash.edu)
# Reference: @article{huang2021driving,
#   title={Driving Behavior Modeling Using Naturalistic Human Driving Data With Inverse Reinforcement Learning},
#   author={Huang, Zhiyu and Wu, Jingda and Lv, Chen},
#   journal={IEEE Transactions on Intelligent Transportation Systems},
#   year={2021},
#   publisher={IEEE}
# }
# @misc{highway-env,
#   author = {Leurent, Edouard},
#   title = {An Environment for Autonomous Driving Decision-Making},
#   year = {2018},
#   publisher = {GitHub},
#   journal = {GitHub repository},
#   howpublished = {\url{https://github.com/eleurent/highway-env}},
# }


from __future__ import annotations

import hashlib
import json
import os
import re
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping

import numpy as np

from highway_env.ngsim_utils.core.constants import (
    MANIFEST_RUNTIME_ENVIRONMENT_IDS,
    MORINOMIYA_MANIFEST_ENVIRONMENT_IDS,
)
from highway_env.ngsim_utils.data.trajectory_gen import trajectory_has_min_continuous_occupancy


PrebuiltCache = dict[
    tuple[str, str, str, float],
    tuple[dict[str, np.ndarray], dict[str, dict[Any, Any]], list[str]],
]


DATASET_MANIFEST_V2 = "dataset_manifest_v2"
EPISODE_STORE_V2 = "episode_store_v2"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
ELIGIBILITY_REASON_BITS = {
    "inactive": 1,
    "provider_interpolated": 2,
    "unsupported_or_forbidden_lane": 4,
    "road_mapping_failed": 8,
    "illegal_successor": 16,
    "low_confidence_topology": 32,
    "invalid_pose_or_heading": 64,
    "section_context_or_boundary_only": 128,
    "source_quarantine": 256,
}
ELIGIBILITY_REASON_ALLOWED_MASK = sum(ELIGIBILITY_REASON_BITS.values())
MANIFEST_CONTROLLED_WINDOW_FRAMES = 200


def _reject_nonfinite_json(value: str) -> None:
    raise ValueError(f"Non-finite JSON number {value!r} is not permitted.")


def _required_nonempty_string(payload: Mapping[str, Any], key: str, *, label: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} requires a non-empty string field {key!r}.")
    return value.strip()


def _optional_nonempty_string(
    payload: Mapping[str, Any], key: str, *, label: str
) -> str | None:
    if key not in payload or payload[key] is None:
        return None
    return _required_nonempty_string(payload, key, label=label)


def _validate_manifest_environment_identity(
    *, site_id: str, environment_id: str | None, label: str
) -> None:
    if environment_id in MORINOMIYA_MANIFEST_ENVIRONMENT_IDS and site_id != "morinomiya":
        raise ValueError(
            f"{label} environment_id {environment_id!r} requires site_id='morinomiya'."
        )
    if environment_id == "i-80" and site_id != "i-80":
        raise ValueError(f"{label} environment_id='i-80' requires site_id='i-80'.")
    if environment_id == "us-101" and site_id != "us-101":
        raise ValueError(
            f"{label} environment_id='us-101' requires site_id='us-101'."
        )
    if site_id == "morinomiya" and environment_id is not None and (
        environment_id not in MORINOMIYA_MANIFEST_ENVIRONMENT_IDS
    ):
        raise ValueError(
            f"{label} has unsupported Morinomiya environment_id {environment_id!r}."
        )


def _required_int(payload: Mapping[str, Any], key: str, *, label: str, minimum: int) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{label} requires an integer field {key!r}.")
    result = int(value)
    if result < int(minimum):
        raise ValueError(f"{label} field {key!r} must be >= {minimum}.")
    return result


def _has_exact_schema_version(
    payload: Mapping[str, Any],
    expected: int,
) -> bool:
    value = payload.get("schema_version")
    return isinstance(value, int) and not isinstance(value, bool) and value == expected


def _load_json_mapping(
    source: str | os.PathLike[str] | Mapping[str, Any],
    *,
    label: str,
) -> tuple[dict[str, Any], Path | None]:
    if isinstance(source, Mapping):
        payload = dict(source)
        path = None
    else:
        path = Path(source).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        try:
            payload = json.loads(
                path.read_text(encoding="utf-8"),
                parse_constant=_reject_nonfinite_json,
            )
        except json.JSONDecodeError as error:
            raise ValueError(f"{label} is not valid JSON: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    try:
        json.dumps(payload, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must contain finite JSON values.") from error
    return payload, path


def _validate_reference_field(
    payload: Mapping[str, Any],
    key: str,
    *,
    label: str,
    allowed_types: tuple[type, ...],
) -> Any:
    if key not in payload:
        raise ValueError(f"{label} is missing required field {key!r}.")
    value = payload[key]
    if str in allowed_types and isinstance(value, str):
        if not value.strip():
            raise ValueError(f"{label} field {key!r} may not be empty.")
        return value
    if dict in allowed_types and isinstance(value, Mapping):
        return dict(value)
    if list in allowed_types and isinstance(value, list):
        return list(value)
    expected = ", ".join(
        {str: "string", dict: "object", list: "list"}[allowed]
        for allowed in allowed_types
    )
    raise ValueError(
        f"{label} field {key!r} must be {expected}, got {type(value).__name__}."
    )


def _validate_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{label} requires a lowercase hexadecimal SHA-256.")
    return value


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_hashed_artifact_reference(value: Any, *, label: str) -> None:
    if not isinstance(value, Mapping):
        raise ValueError(
            f"{label} requires an object with relative_path and sha256 for a "
            "five-environment dataset."
        )
    _required_nonempty_string(value, "relative_path", label=label)
    _validate_sha256(value.get("sha256"), label=label)


@dataclass(frozen=True)
class DatasetManifestV2:
    """Validated, extra-field-tolerant site dataset contract.

    This type validates the frozen V2 envelope only. Producer-specific metadata
    inside source/session/split references is intentionally retained in
    ``payload`` without being interpreted by the simulator.
    """

    dataset_id: str
    site_id: str
    scene: str
    environment_id: str | None
    source_artifacts: Any
    sessions: Any
    coordinate_frame: Any
    split_contract: Any
    episode_store: Any
    road_geometry: Any
    test_accessed: bool
    payload: Mapping[str, Any] = field(repr=False, compare=False)
    path: Path | None = field(default=None, repr=False, compare=False)
    schema_version: int = field(default=2, init=False)
    contract_id: str = field(default=DATASET_MANIFEST_V2, init=False)

    @classmethod
    def from_source(
        cls,
        source: str | os.PathLike[str] | Mapping[str, Any],
    ) -> "DatasetManifestV2":
        payload, path = _load_json_mapping(source, label="DatasetManifestV2")
        if not _has_exact_schema_version(payload, 2):
            raise ValueError("DatasetManifestV2 requires schema_version=2.")
        if payload.get("contract_id") != DATASET_MANIFEST_V2:
            raise ValueError(
                "DatasetManifestV2 requires contract_id='dataset_manifest_v2'."
            )
        test_accessed = payload.get("test_accessed")
        if not isinstance(test_accessed, bool):
            raise ValueError("DatasetManifestV2 requires boolean test_accessed.")
        site_id = _required_nonempty_string(
            payload, "site_id", label="DatasetManifestV2"
        )
        environment_id = _optional_nonempty_string(
            payload, "environment_id", label="DatasetManifestV2"
        )
        _validate_manifest_environment_identity(
            site_id=site_id,
            environment_id=environment_id,
            label="DatasetManifestV2",
        )
        scene = _required_nonempty_string(
            payload, "scene", label="DatasetManifestV2"
        )
        if scene in MANIFEST_RUNTIME_ENVIRONMENT_IDS and environment_id is None:
            raise ValueError(
                "DatasetManifestV2 requires environment_id for the five manifest "
                f"runtime scenes; scene={scene!r}."
            )
        if environment_id in MANIFEST_RUNTIME_ENVIRONMENT_IDS and scene != environment_id:
            raise ValueError(
                "DatasetManifestV2 scene must equal environment_id for the five "
                f"manifest runtime environments: {scene!r} != {environment_id!r}."
            )
        episode_store = _validate_reference_field(
            payload,
            "episode_store",
            label="DatasetManifestV2",
            allowed_types=(str, dict),
        )
        road_geometry = _validate_reference_field(
            payload,
            "road_geometry",
            label="DatasetManifestV2",
            allowed_types=(str, dict),
        )
        if environment_id in MANIFEST_RUNTIME_ENVIRONMENT_IDS:
            _validate_hashed_artifact_reference(
                road_geometry, label="DatasetManifestV2 road_geometry"
            )
            if not isinstance(episode_store, Mapping) or not episode_store:
                raise ValueError(
                    "Five-environment DatasetManifestV2 episode_store must map "
                    "splits to hashed artifact references."
                )
            for split, reference in episode_store.items():
                if not isinstance(split, str) or not split.strip():
                    raise ValueError(
                        "DatasetManifestV2 episode_store split keys must be non-empty strings."
                    )
                _validate_hashed_artifact_reference(
                    reference,
                    label=f"DatasetManifestV2 episode_store[{split!r}]",
                )
        return cls(
            dataset_id=_required_nonempty_string(
                payload, "dataset_id", label="DatasetManifestV2"
            ),
            site_id=site_id,
            scene=scene,
            environment_id=environment_id,
            source_artifacts=_validate_reference_field(
                payload,
                "source_artifacts",
                label="DatasetManifestV2",
                allowed_types=(list,),
            ),
            sessions=_validate_reference_field(
                payload,
                "sessions",
                label="DatasetManifestV2",
                allowed_types=(list,),
            ),
            coordinate_frame=_validate_reference_field(
                payload,
                "coordinate_frame",
                label="DatasetManifestV2",
                allowed_types=(dict,),
            ),
            split_contract=_validate_reference_field(
                payload,
                "split_contract",
                label="DatasetManifestV2",
                allowed_types=(dict,),
            ),
            episode_store=episode_store,
            road_geometry=road_geometry,
            test_accessed=test_accessed,
            payload=payload,
            path=path,
        )

    def episode_store_path(self, split: str) -> Path | None:
        """Resolve a split-specific store reference when this manifest has a path."""

        reference = self.episode_store
        if isinstance(reference, Mapping):
            if split not in reference:
                raise ValueError(
                    f"DatasetManifestV2 episode_store does not declare split {split!r}."
                )
            reference = reference[split]
        return self._resolve_artifact_reference(reference, label="episode_store")

    def road_geometry_path(self) -> Path | None:
        """Resolve the road-geometry reference when represented as a file path."""

        return self._resolve_artifact_reference(
            self.road_geometry,
            label="road_geometry",
        )

    def episode_store_sha256(self, split: str) -> str | None:
        reference = self.episode_store
        if isinstance(reference, Mapping) and "relative_path" not in reference:
            if split not in reference:
                raise ValueError(
                    f"DatasetManifestV2 episode_store does not declare split {split!r}."
                )
            reference = reference[split]
        return self._artifact_sha256(reference, label="episode_store")

    def road_geometry_sha256(self) -> str | None:
        return self._artifact_sha256(self.road_geometry, label="road_geometry")

    @staticmethod
    def _artifact_sha256(reference: Any, *, label: str) -> str | None:
        if not isinstance(reference, Mapping):
            return None
        if "sha256" not in reference:
            return None
        return _validate_sha256(reference["sha256"], label=f"{label} reference")

    def _resolve_artifact_reference(self, reference: Any, *, label: str) -> Path | None:
        if isinstance(reference, Mapping):
            for key in ("relative_path", "path", "manifest"):
                if key in reference:
                    reference = reference[key]
                    break
            else:
                return None
        if not isinstance(reference, str):
            return None
        if self.path is None:
            return None
        candidate = Path(reference).expanduser()
        if not candidate.is_absolute():
            candidate = self.path.parent / candidate
        return candidate.resolve()


@dataclass(frozen=True)
class EpisodeStoreIndexEntryV2:
    episode_id: str
    relative_path: str
    sha256: str
    session_id: str
    split: str
    start_time_ms: int
    frame_count: int
    vehicle_count: int
    environment_id: str | None
    controlled_candidate_vehicle_count: int | None
    payload: Mapping[str, Any] = field(repr=False, compare=False)


@dataclass(frozen=True)
class EpisodeDataV2:
    """One validated episode loaded from an EpisodeStoreV2 shard."""

    entry: EpisodeStoreIndexEntryV2
    timestamps_ms: np.ndarray
    vehicle_ids: np.ndarray
    dimensions_m: np.ndarray
    states: np.ndarray
    active_mask: np.ndarray
    provider_mask: np.ndarray
    controlled_vehicle_eligible_mask: np.ndarray | None
    road_valid_mask: np.ndarray | None
    training_eligible_mask: np.ndarray | None
    eligibility_reason_mask: np.ndarray | None
    source_front_center_xy_m: np.ndarray | None
    heading_rad: np.ndarray | None
    heading_valid_mask: np.ndarray | None
    heading_derivation: np.ndarray | None
    contiguous_track_instance: np.ndarray | None

    def as_legacy_trajectory_dict(self) -> dict[int, dict[str, Any]]:
        """Expose the selected SI episode in the legacy per-vehicle container.

        The returned records are explicitly marked as simulator-SI so NGSimEnv
        can bypass the legacy US-feet/Japanese-km/h conversion path.
        """

        result: dict[int, dict[str, Any]] = {}
        for column, vehicle_id in enumerate(self.vehicle_ids):
            trajectory = np.asarray(self.states[:, column, :], dtype=float).copy()
            inactive = ~np.asarray(self.active_mask[:, column], dtype=bool)
            trajectory[inactive] = 0.0
            record = {
                "length": float(self.dimensions_m[column, 0]),
                "width": float(self.dimensions_m[column, 1]),
                "trajectory": trajectory,
                "active_mask": np.asarray(
                    self.active_mask[:, column], dtype=bool
                ).copy(),
                "provider_observation_mask": np.asarray(
                    self.provider_mask[:, column], dtype=np.int8
                ).copy(),
                "trajectory_units": "simulator_si_v1",
            }
            if self.controlled_vehicle_eligible_mask is not None:
                record["controlled_vehicle_eligible"] = bool(
                    self.controlled_vehicle_eligible_mask[column]
                )
            if self.road_valid_mask is not None:
                record.update(
                    {
                        "road_valid_mask": np.asarray(
                            self.road_valid_mask[:, column], dtype=bool
                        ).copy(),
                        "training_eligible_mask": np.asarray(
                            self.training_eligible_mask[:, column], dtype=bool
                        ).copy(),
                        "eligibility_reason_mask": np.asarray(
                            self.eligibility_reason_mask[:, column], dtype=np.uint16
                        ).copy(),
                    }
                )
            if self.source_front_center_xy_m is not None:
                record.update(
                    {
                        "source_front_center_xy_m": np.asarray(
                            self.source_front_center_xy_m[:, column, :],
                            dtype=np.float64,
                        ).copy(),
                    }
                )
            if self.heading_rad is not None:
                record.update(
                    {
                        "heading_rad": np.asarray(
                            self.heading_rad[:, column], dtype=np.float64
                        ).copy(),
                        "heading_valid_mask": np.asarray(
                            self.heading_valid_mask[:, column], dtype=bool
                        ).copy(),
                        "heading_derivation": np.asarray(
                            self.heading_derivation[:, column], dtype=np.int8
                        ).copy(),
                    }
                )
            if self.contiguous_track_instance is not None:
                record["contiguous_track_instance"] = np.asarray(
                    self.contiguous_track_instance[:, column], dtype=np.int32
                ).copy()
            result[int(vehicle_id)] = record
        return result


class EpisodeStoreV2(Mapping[str, EpisodeDataV2]):
    """Lazy, checksum-verifying reader for per-episode NPZ shards.

    Construction reads only the small JSON manifest and JSONL index. Episode
    arrays are opened and validated on first access, then retained in a bounded
    LRU cache.
    """

    REQUIRED_ARRAYS = frozenset(
        {
            "timestamps_ms",
            "vehicle_ids",
            "dimensions_m",
            "states",
            "active_mask",
            "provider_mask",
        }
    )
    I80_REFERENCE_ARRAYS = frozenset(
        {
            "source_front_center_xy_m",
            "heading_rad",
            "heading_valid_mask",
            "heading_derivation",
        }
    )
    HEADING_ARRAYS = frozenset(
        {"heading_rad", "heading_valid_mask", "heading_derivation"}
    )
    ELIGIBILITY_ARRAYS = frozenset(
        {
            "road_valid_mask",
            "training_eligible_mask",
            "eligibility_reason_mask",
        }
    )
    CONTROLLED_ELIGIBILITY_ARRAY = "controlled_vehicle_eligible_mask"
    OFFICIAL_I80_SITE_ID = "i-80"
    US101_ENVIRONMENT_ID = "us-101"
    US101_IDENTITY_ARRAY = "contiguous_track_instance"
    US101_IDENTITY_CONTRACT = "us101_contiguous_track_instance_split_guard_v1"
    I80_STATE_POSITION_REFERENCE = "vehicle_geometric_body_center"
    I80_SOURCE_POSITION_REFERENCE = "vehicle_front_center"

    def __init__(
        self,
        manifest: str | os.PathLike[str] | Mapping[str, Any],
        *,
        base_dir: str | os.PathLike[str] | None = None,
        max_cached_episodes: int = 2,
    ) -> None:
        payload, manifest_path = _load_json_mapping(
            manifest, label="EpisodeStoreV2 manifest"
        )
        if not _has_exact_schema_version(payload, 2):
            raise ValueError("EpisodeStoreV2 requires schema_version=2.")
        if payload.get("contract_id") != EPISODE_STORE_V2:
            raise ValueError(
                "EpisodeStoreV2 requires contract_id='episode_store_v2'."
            )
        self.site_id = _required_nonempty_string(
            payload, "site_id", label="EpisodeStoreV2"
        )
        self.environment_id = _optional_nonempty_string(
            payload, "environment_id", label="EpisodeStoreV2"
        )
        _validate_manifest_environment_identity(
            site_id=self.site_id,
            environment_id=self.environment_id,
            label="EpisodeStoreV2",
        )
        self.requires_manifest_runtime_arrays = (
            self.environment_id in MANIFEST_RUNTIME_ENVIRONMENT_IDS
        )
        self.split = _required_nonempty_string(
            payload, "split", label="EpisodeStoreV2"
        )
        self.episode_count = _required_int(
            payload, "episode_count", label="EpisodeStoreV2", minimum=0
        )
        index_file = _required_nonempty_string(
            payload, "index_file", label="EpisodeStoreV2"
        )
        if manifest_path is not None:
            root = manifest_path.parent
        elif base_dir is not None:
            root = Path(base_dir).expanduser().resolve()
        else:
            raise ValueError(
                "EpisodeStoreV2 constructed from an object requires base_dir."
            )
        self.root = root.resolve()
        self.manifest_path = manifest_path
        self.payload = payload
        self.requires_i80_reference_arrays = self.site_id == self.OFFICIAL_I80_SITE_ID
        if self.requires_i80_reference_arrays:
            if payload.get("state_position_reference") != self.I80_STATE_POSITION_REFERENCE:
                raise ValueError(
                    "Official I-80 EpisodeStoreV2 requires "
                    "state_position_reference='vehicle_geometric_body_center'."
                )
            if payload.get("source_position_reference") != self.I80_SOURCE_POSITION_REFERENCE:
                raise ValueError(
                    "Official I-80 EpisodeStoreV2 requires "
                    "source_position_reference='vehicle_front_center'."
                )
        self.state_position_reference = _optional_nonempty_string(
            payload, "state_position_reference", label="EpisodeStoreV2"
        )
        self.source_position_reference = _optional_nonempty_string(
            payload, "source_position_reference", label="EpisodeStoreV2"
        )
        self.heading_contract = self._heading_contract(payload)
        self.identity_contract = _optional_nonempty_string(
            payload, "identity_contract", label="EpisodeStoreV2"
        )
        self.mask_contract = _optional_nonempty_string(
            payload, "mask_contract", label="EpisodeStoreV2"
        )
        self.eligibility_reason_bits = payload.get("eligibility_reason_bits")
        self.index_sha256 = payload.get("index_sha256")
        self.controlled_candidate_vehicle_count = payload.get(
            "controlled_candidate_vehicle_count"
        )
        if self.requires_manifest_runtime_arrays:
            if self.state_position_reference is None:
                raise ValueError(
                    "Five-environment EpisodeStoreV2 requires "
                    "state_position_reference."
                )
            if self.source_position_reference is None:
                raise ValueError(
                    "Five-environment EpisodeStoreV2 requires "
                    "source_position_reference."
                )
            if self.heading_contract is None:
                raise ValueError(
                    "Five-environment EpisodeStoreV2 requires heading_contract "
                    "at top level or inside reference_point_conversion."
                )
            if self.mask_contract is None:
                raise ValueError(
                    "Five-environment EpisodeStoreV2 requires mask_contract."
                )
            if self.eligibility_reason_bits != ELIGIBILITY_REASON_BITS:
                raise ValueError(
                    "Five-environment EpisodeStoreV2 requires the frozen "
                    "eligibility_reason_bits mapping."
                )
            if self.environment_id in MORINOMIYA_MANIFEST_ENVIRONMENT_IDS and (
                self.state_position_reference != "vehicle_body_center"
                or self.source_position_reference != "vehicle_body_center"
            ):
                raise ValueError(
                    "Morinomiya EpisodeStoreV2 requires source/state position "
                    "reference='vehicle_body_center'."
                )
            if self.environment_id == self.US101_ENVIRONMENT_ID and (
                self.identity_contract != self.US101_IDENTITY_CONTRACT
                or payload.get("contiguous_track_instance_array")
                != "int32[T,V]_minus_one_inactive"
            ):
                raise ValueError(
                    "US-101 EpisodeStoreV2 requires its contiguous-track identity contract."
                )
        self.index_path = self._resolve_relative_path(index_file, label="index_file")
        if not self.index_path.is_file():
            raise FileNotFoundError(self.index_path)
        if self.index_sha256 is not None:
            expected_index_sha256 = _validate_sha256(
                self.index_sha256, label="EpisodeStoreV2 index_sha256"
            )
            actual_index_sha256 = self._sha256(self.index_path)
            if actual_index_sha256 != expected_index_sha256:
                raise ValueError(
                    "EpisodeStoreV2 index_sha256 mismatch: "
                    f"{actual_index_sha256} != {expected_index_sha256}."
                )
        elif self.requires_manifest_runtime_arrays:
            raise ValueError(
                "Five-environment EpisodeStoreV2 requires index_sha256."
            )
        if self.requires_manifest_runtime_arrays and self.split in {
            "train",
            "validation",
            "val",
        }:
            if self.controlled_candidate_vehicle_count is None:
                raise ValueError(
                    "Five-environment non-test EpisodeStoreV2 requires "
                    "controlled_candidate_vehicle_count."
                )
            declared_controlled_count = _required_int(
                payload,
                "controlled_candidate_vehicle_count",
                label="EpisodeStoreV2",
                minimum=0,
            )
            if self.episode_count > 0 and declared_controlled_count <= 0:
                raise ValueError(
                    "A nonempty five-environment production store requires at least "
                    "one controlled candidate vehicle."
                )
            self.controlled_candidate_vehicle_count = declared_controlled_count
        if isinstance(max_cached_episodes, bool) or not isinstance(
            max_cached_episodes, (int, np.integer)
        ):
            raise ValueError("max_cached_episodes must be an integer >= 0.")
        if int(max_cached_episodes) < 0:
            raise ValueError("max_cached_episodes must be >= 0.")
        self.max_cached_episodes = int(max_cached_episodes)
        self._entries = self._load_index()
        self._cache: OrderedDict[str, EpisodeDataV2] = OrderedDict()

    @staticmethod
    def _heading_contract(payload: Mapping[str, Any]) -> str | None:
        direct = _optional_nonempty_string(
            payload, "heading_contract", label="EpisodeStoreV2"
        )
        conversion = payload.get("reference_point_conversion")
        nested = None
        if isinstance(conversion, Mapping):
            nested = _optional_nonempty_string(
                conversion, "heading_contract", label="EpisodeStoreV2"
            )
        if direct is not None and nested is not None and direct != nested:
            raise ValueError(
                "EpisodeStoreV2 top-level and reference-point heading_contract "
                "values do not match."
            )
        return direct if direct is not None else nested

    @property
    def canonical_sha256(self) -> str:
        manifest_bytes = json.dumps(
            self.payload,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        digest = hashlib.sha256(manifest_bytes)
        digest.update(b"\0")
        digest.update(self._sha256(self.index_path).encode("ascii"))
        return digest.hexdigest()

    @property
    def episode_ids(self) -> list[str]:
        return list(self._entries)

    @property
    def loaded_episode_ids(self) -> tuple[str, ...]:
        return tuple(self._cache)

    def __contains__(self, episode_id: object) -> bool:
        return isinstance(episode_id, str) and episode_id in self._entries

    def __getitem__(self, episode_id: str) -> EpisodeDataV2:
        return self.load_episode(episode_id)

    def __iter__(self) -> Iterator[str]:
        return iter(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def entry(self, episode_id: str) -> EpisodeStoreIndexEntryV2:
        try:
            return self._entries[str(episode_id)]
        except KeyError as error:
            raise KeyError(f"Unknown EpisodeStoreV2 episode {episode_id!r}.") from error

    def _resolve_relative_path(self, value: str, *, label: str) -> Path:
        relative = Path(value)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"EpisodeStoreV2 {label} must be a safe relative path.")
        resolved = (self.root / relative).resolve()
        try:
            resolved.relative_to(self.root)
        except ValueError as error:
            raise ValueError(
                f"EpisodeStoreV2 {label} escapes the store root: {value!r}."
            ) from error
        return resolved

    def _load_index(self) -> dict[str, EpisodeStoreIndexEntryV2]:
        entries: dict[str, EpisodeStoreIndexEntryV2] = {}
        paths: set[Path] = set()
        for line_number, raw_line in enumerate(
            self.index_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not raw_line.strip():
                continue
            try:
                payload = json.loads(raw_line, parse_constant=_reject_nonfinite_json)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"EpisodeStoreV2 index has invalid JSON on line {line_number}."
                ) from error
            if not isinstance(payload, dict):
                raise ValueError(
                    f"EpisodeStoreV2 index line {line_number} must be an object."
                )
            label = f"EpisodeStoreV2 index line {line_number}"
            episode_id = _required_nonempty_string(payload, "episode_id", label=label)
            relative_path = _required_nonempty_string(
                payload, "relative_path", label=label
            )
            resolved_path = self._resolve_relative_path(
                relative_path, label="relative_path"
            )
            if Path(relative_path).suffix.lower() != ".npz":
                raise ValueError(f"{label} relative_path must name an NPZ shard.")
            sha256 = _required_nonempty_string(payload, "sha256", label=label)
            if _SHA256_RE.fullmatch(sha256) is None:
                raise ValueError(f"{label} requires a lowercase hexadecimal SHA-256.")
            session_id = _required_nonempty_string(payload, "session_id", label=label)
            split = _required_nonempty_string(payload, "split", label=label)
            if split != self.split:
                raise ValueError(
                    f"{label} split {split!r} does not match store split {self.split!r}."
                )
            environment_id = _optional_nonempty_string(
                payload, "environment_id", label=label
            )
            if self.requires_manifest_runtime_arrays and environment_id is None:
                raise ValueError(
                    f"{label} requires environment_id for a five-environment store."
                )
            if environment_id != self.environment_id:
                raise ValueError(
                    f"{label} environment_id {environment_id!r} does not match "
                    f"store environment_id {self.environment_id!r}."
                )
            for key, expected in (
                ("state_position_reference", self.state_position_reference),
                ("source_position_reference", self.source_position_reference),
                ("heading_contract", self.heading_contract),
                ("mask_contract", self.mask_contract),
            ):
                if self.requires_manifest_runtime_arrays:
                    actual = _required_nonempty_string(payload, key, label=label)
                    if actual != expected:
                        raise ValueError(
                            f"{label} field {key!r} does not match the store manifest."
                        )
            if self.requires_manifest_runtime_arrays and (
                payload.get("eligibility_reason_bits") != ELIGIBILITY_REASON_BITS
            ):
                raise ValueError(
                    f"{label} requires the frozen eligibility_reason_bits mapping."
                )
            controlled_count = payload.get("controlled_candidate_vehicle_count")
            if self.requires_manifest_runtime_arrays and self.split in {
                "train",
                "validation",
                "val",
            }:
                controlled_count = _required_int(
                    payload,
                    "controlled_candidate_vehicle_count",
                    label=label,
                    minimum=0,
                )
            elif controlled_count is not None:
                controlled_count = _required_int(
                    payload,
                    "controlled_candidate_vehicle_count",
                    label=label,
                    minimum=0,
                )
            entry = EpisodeStoreIndexEntryV2(
                episode_id=episode_id,
                relative_path=relative_path,
                sha256=sha256,
                session_id=session_id,
                split=split,
                start_time_ms=_required_int(
                    payload, "start_time_ms", label=label, minimum=0
                ),
                frame_count=_required_int(
                    payload, "frame_count", label=label, minimum=1
                ),
                vehicle_count=_required_int(
                    payload, "vehicle_count", label=label, minimum=1
                ),
                environment_id=environment_id,
                controlled_candidate_vehicle_count=controlled_count,
                payload=payload,
            )
            if episode_id in entries:
                raise ValueError(
                    f"EpisodeStoreV2 index contains duplicate episode_id {episode_id!r}."
                )
            if resolved_path in paths:
                raise ValueError(
                    f"EpisodeStoreV2 index reuses relative_path {relative_path!r}."
                )
            entries[episode_id] = entry
            paths.add(resolved_path)
        if len(entries) != self.episode_count:
            raise ValueError(
                "EpisodeStoreV2 episode_count does not match JSONL index rows: "
                f"{self.episode_count} != {len(entries)}."
            )
        if self.requires_manifest_runtime_arrays and self.split in {
            "train",
            "validation",
            "val",
        }:
            indexed_total = sum(
                int(entry.controlled_candidate_vehicle_count or 0)
                for entry in entries.values()
            )
            if indexed_total != int(self.controlled_candidate_vehicle_count):
                raise ValueError(
                    "EpisodeStoreV2 controlled_candidate_vehicle_count does not "
                    f"match the JSONL total: {self.controlled_candidate_vehicle_count} "
                    f"!= {indexed_total}."
                )
        return entries

    @staticmethod
    def _sha256(path: Path) -> str:
        return sha256_file(path)

    @staticmethod
    def _boolean_mask(values: np.ndarray, *, name: str) -> np.ndarray:
        array = np.asarray(values)
        if array.dtype == np.bool_:
            return array.astype(bool, copy=False)
        if not np.issubdtype(array.dtype, np.integer) or not np.all(
            np.isin(array, [0, 1])
        ):
            raise ValueError(f"EpisodeStoreV2 {name} must contain only boolean/0/1 values.")
        return array.astype(bool, copy=False)

    def load_episode(self, episode_id: str) -> EpisodeDataV2:
        episode_id = str(episode_id)
        cached = self._cache.get(episode_id)
        if cached is not None:
            self._cache.move_to_end(episode_id)
            return cached
        entry = self.entry(episode_id)
        path = self._resolve_relative_path(entry.relative_path, label="relative_path")
        if not path.is_file():
            raise FileNotFoundError(path)
        actual_sha256 = self._sha256(path)
        if actual_sha256 != entry.sha256:
            raise ValueError(
                f"EpisodeStoreV2 checksum mismatch for {episode_id!r}: "
                f"{actual_sha256} != {entry.sha256}."
            )
        with np.load(path, allow_pickle=False) as arrays:
            missing = sorted(self.REQUIRED_ARRAYS.difference(arrays.files))
            if missing:
                raise ValueError(
                    f"EpisodeStoreV2 episode {episode_id!r} is missing arrays: {missing}."
                )
            available_heading_arrays = self.HEADING_ARRAYS.intersection(arrays.files)
            if available_heading_arrays and available_heading_arrays != self.HEADING_ARRAYS:
                missing_heading = sorted(self.HEADING_ARRAYS.difference(arrays.files))
                partial_label = (
                    "partial I-80 reference-point contract"
                    if self.requires_i80_reference_arrays
                    else "partial persisted-heading contract"
                )
                raise ValueError(
                    f"EpisodeStoreV2 episode {episode_id!r} has a {partial_label}; "
                    f"missing arrays: {missing_heading}."
                )
            has_heading_arrays = available_heading_arrays == self.HEADING_ARRAYS
            has_source_front = "source_front_center_xy_m" in arrays.files
            if self.requires_i80_reference_arrays and not (
                has_heading_arrays and has_source_front
            ):
                raise ValueError(
                    f"Official I-80 EpisodeStoreV2 episode {episode_id!r} is missing "
                    f"required reference-point arrays: {sorted(self.I80_REFERENCE_ARRAYS)}."
                )
            if has_source_front and not has_heading_arrays:
                raise ValueError(
                    f"EpisodeStoreV2 episode {episode_id!r} cannot expose "
                    "source_front_center_xy_m without the persisted-heading arrays."
                )
            if self.requires_manifest_runtime_arrays and not has_heading_arrays:
                raise ValueError(
                    f"Five-environment EpisodeStoreV2 episode {episode_id!r} is missing "
                    f"required heading arrays: {sorted(self.HEADING_ARRAYS)}."
                )
            available_eligibility_arrays = self.ELIGIBILITY_ARRAYS.intersection(
                arrays.files
            )
            if (
                available_eligibility_arrays
                and available_eligibility_arrays != self.ELIGIBILITY_ARRAYS
            ):
                missing_eligibility = sorted(
                    self.ELIGIBILITY_ARRAYS.difference(arrays.files)
                )
                raise ValueError(
                    f"EpisodeStoreV2 episode {episode_id!r} has a partial eligibility "
                    f"contract; missing arrays: {missing_eligibility}."
                )
            has_eligibility_arrays = (
                available_eligibility_arrays == self.ELIGIBILITY_ARRAYS
            )
            if self.requires_manifest_runtime_arrays and not has_eligibility_arrays:
                raise ValueError(
                    f"Five-environment EpisodeStoreV2 episode {episode_id!r} is missing "
                    f"required eligibility arrays: {sorted(self.ELIGIBILITY_ARRAYS)}."
                )
            has_controlled_eligibility = (
                self.CONTROLLED_ELIGIBILITY_ARRAY in arrays.files
            )
            if self.requires_manifest_runtime_arrays and not has_controlled_eligibility:
                raise ValueError(
                    f"Five-environment EpisodeStoreV2 episode {episode_id!r} is "
                    "missing required array controlled_vehicle_eligible_mask."
                )
            timestamps_ms = np.asarray(arrays["timestamps_ms"])
            vehicle_ids = np.asarray(arrays["vehicle_ids"])
            dimensions_m = np.asarray(arrays["dimensions_m"], dtype=np.float64)
            states = np.asarray(arrays["states"], dtype=np.float64)
            active_mask = self._boolean_mask(arrays["active_mask"], name="active_mask")
            provider_mask = np.asarray(arrays["provider_mask"])
            if has_controlled_eligibility:
                controlled_raw = np.asarray(
                    arrays[self.CONTROLLED_ELIGIBILITY_ARRAY]
                )
                if controlled_raw.dtype != np.bool_:
                    raise ValueError(
                        "EpisodeStoreV2 controlled_vehicle_eligible_mask must "
                        "have dtype bool."
                    )
                controlled_vehicle_eligible_mask = controlled_raw
            else:
                controlled_vehicle_eligible_mask = None
            has_contiguous_track_instance = self.US101_IDENTITY_ARRAY in arrays.files
            if (
                self.environment_id == self.US101_ENVIRONMENT_ID
                and not has_contiguous_track_instance
            ):
                raise ValueError(
                    "US-101 EpisodeStoreV2 is missing contiguous_track_instance."
                )
            if has_contiguous_track_instance:
                contiguous_track_instance_raw = np.asarray(
                    arrays[self.US101_IDENTITY_ARRAY]
                )
                if contiguous_track_instance_raw.dtype != np.int32:
                    raise ValueError(
                        "EpisodeStoreV2 contiguous_track_instance must have dtype int32."
                    )
                contiguous_track_instance = contiguous_track_instance_raw
            else:
                contiguous_track_instance = None
            if has_eligibility_arrays:
                road_valid_raw = np.asarray(arrays["road_valid_mask"])
                training_eligible_raw = np.asarray(arrays["training_eligible_mask"])
                eligibility_reason_raw = np.asarray(arrays["eligibility_reason_mask"])
                if road_valid_raw.dtype != np.bool_:
                    raise ValueError("EpisodeStoreV2 road_valid_mask must have dtype bool.")
                if training_eligible_raw.dtype != np.bool_:
                    raise ValueError(
                        "EpisodeStoreV2 training_eligible_mask must have dtype bool."
                    )
                if eligibility_reason_raw.dtype != np.uint16:
                    raise ValueError(
                        "EpisodeStoreV2 eligibility_reason_mask must have dtype uint16."
                    )
                road_valid_mask = road_valid_raw
                training_eligible_mask = training_eligible_raw
                eligibility_reason_mask = eligibility_reason_raw
            else:
                road_valid_mask = None
                training_eligible_mask = None
                eligibility_reason_mask = None
            if has_heading_arrays:
                heading_raw = np.asarray(arrays["heading_rad"])
                heading_valid_raw = np.asarray(arrays["heading_valid_mask"])
                heading_derivation_raw = np.asarray(arrays["heading_derivation"])
                if heading_raw.dtype != np.float64:
                    raise ValueError("EpisodeStoreV2 heading_rad must have dtype float64.")
                if heading_valid_raw.dtype != np.bool_:
                    raise ValueError(
                        "EpisodeStoreV2 heading_valid_mask must have dtype bool."
                    )
                if heading_derivation_raw.dtype != np.int8:
                    raise ValueError(
                        "EpisodeStoreV2 heading_derivation must have dtype int8."
                    )
                heading_rad = heading_raw
                heading_valid_mask = heading_valid_raw
                heading_derivation = heading_derivation_raw
            else:
                heading_rad = None
                heading_valid_mask = None
                heading_derivation = None
            if has_source_front:
                source_front_raw = np.asarray(arrays["source_front_center_xy_m"])
                if source_front_raw.dtype != np.float64:
                    raise ValueError(
                        "EpisodeStoreV2 source_front_center_xy_m must have dtype float64."
                    )
                source_front_center_xy_m = source_front_raw
            else:
                source_front_center_xy_m = None

        frames = entry.frame_count
        vehicles = entry.vehicle_count
        if (
            timestamps_ms.ndim != 1
            or timestamps_ms.shape != (frames,)
            or not np.issubdtype(timestamps_ms.dtype, np.integer)
            or np.any(np.diff(timestamps_ms.astype(np.int64)) <= 0)
        ):
            raise ValueError(
                f"EpisodeStoreV2 timestamps_ms must be {frames} strictly increasing integers."
            )
        if (
            vehicle_ids.ndim != 1
            or vehicle_ids.shape != (vehicles,)
            or not np.issubdtype(vehicle_ids.dtype, np.integer)
            or len(np.unique(vehicle_ids)) != vehicles
        ):
            raise ValueError(
                f"EpisodeStoreV2 vehicle_ids must be {vehicles} unique integers."
            )
        if dimensions_m.shape != (vehicles, 2) or not np.all(
            np.isfinite(dimensions_m) & (dimensions_m > 0.0)
        ):
            raise ValueError(
                f"EpisodeStoreV2 dimensions_m must have finite positive shape {(vehicles, 2)}."
            )
        if states.shape != (frames, vehicles, 4) or not np.all(np.isfinite(states)):
            raise ValueError(
                "EpisodeStoreV2 states must be finite [frame, vehicle, "
                f"(x_m,y_m,speed_mps,raw_lane_id)] with shape {(frames, vehicles, 4)}."
            )
        if active_mask.shape != (frames, vehicles):
            raise ValueError(
                f"EpisodeStoreV2 active_mask must have shape {(frames, vehicles)}."
            )
        if (
            provider_mask.shape != (frames, vehicles)
            or not np.issubdtype(provider_mask.dtype, np.integer)
            or not np.all(np.isin(provider_mask, [-1, 0, 1]))
        ):
            raise ValueError(
                "EpisodeStoreV2 provider_mask must have aligned integer values in {-1,0,1}."
            )
        if controlled_vehicle_eligible_mask is not None:
            if controlled_vehicle_eligible_mask.shape != (vehicles,):
                raise ValueError(
                    "EpisodeStoreV2 controlled_vehicle_eligible_mask must have "
                    f"shape {(vehicles,)}."
                )
            controlled_count = int(
                np.count_nonzero(controlled_vehicle_eligible_mask)
            )
            if (
                entry.controlled_candidate_vehicle_count is not None
                and controlled_count != entry.controlled_candidate_vehicle_count
            ):
                raise ValueError(
                    "EpisodeStoreV2 controlled_vehicle_eligible_mask true count "
                    "does not match the JSONL controlled_candidate_vehicle_count: "
                    f"{controlled_count} != "
                    f"{entry.controlled_candidate_vehicle_count}."
                )
        if np.any(provider_mask[~active_mask] != -1):
            raise ValueError(
                "EpisodeStoreV2 inactive states require provider_mask=-1."
            )
        if np.any(~np.isin(provider_mask[active_mask], [0, 1])):
            raise ValueError(
                "EpisodeStoreV2 active states require provider_mask in {0,1}."
            )
        if road_valid_mask is not None:
            aligned_shape = (frames, vehicles)
            for name, mask in (
                ("road_valid_mask", road_valid_mask),
                ("training_eligible_mask", training_eligible_mask),
                ("eligibility_reason_mask", eligibility_reason_mask),
            ):
                if mask.shape != aligned_shape:
                    raise ValueError(
                        f"EpisodeStoreV2 {name} must have shape {aligned_shape}."
                    )
            if np.any(road_valid_mask[~active_mask]):
                raise ValueError(
                    "EpisodeStoreV2 inactive states require road_valid_mask=false."
                )
            if np.any(training_eligible_mask[~active_mask]):
                raise ValueError(
                    "EpisodeStoreV2 inactive states require "
                    "training_eligible_mask=false."
                )
            unknown_reason_bits = np.bitwise_and(
                eligibility_reason_mask,
                np.uint16(~ELIGIBILITY_REASON_ALLOWED_MASK & 0xFFFF),
            )
            if np.any(unknown_reason_bits != 0):
                raise ValueError(
                    "EpisodeStoreV2 eligibility_reason_mask contains undeclared bits."
                )
            inactive_bit = np.uint16(ELIGIBILITY_REASON_BITS["inactive"])
            if np.any(
                np.bitwise_and(eligibility_reason_mask[~active_mask], inactive_bit)
                == 0
            ):
                raise ValueError(
                    "EpisodeStoreV2 inactive states must include eligibility reason bit 1."
                )
            if np.any(
                np.bitwise_and(eligibility_reason_mask[active_mask], inactive_bit)
                != 0
            ):
                raise ValueError(
                    "EpisodeStoreV2 active states may not include eligibility reason bit 1."
                )
            expected_training_eligible = (
                active_mask & road_valid_mask & (eligibility_reason_mask == 0)
            )
            if not np.array_equal(
                training_eligible_mask, expected_training_eligible
            ):
                raise ValueError(
                    "EpisodeStoreV2 training_eligible_mask must equal active_mask & "
                    "road_valid_mask & (eligibility_reason_mask == 0)."
                )
            if self.environment_id in MORINOMIYA_MANIFEST_ENVIRONMENT_IDS and np.any(
                training_eligible_mask & (provider_mask != 1)
            ):
                raise ValueError(
                    "Morinomiya training-eligible states require provider_mask=1."
                )
        if heading_rad is not None:
            if heading_rad.shape != (frames, vehicles) or not np.all(
                np.isfinite(heading_rad)
            ):
                raise ValueError(
                    "EpisodeStoreV2 heading_rad must be finite with shape "
                    f"{(frames, vehicles)}."
                )
            if np.any(heading_rad < -np.pi) or np.any(heading_rad > np.pi):
                raise ValueError("EpisodeStoreV2 heading_rad must lie in [-pi, pi].")
            if heading_valid_mask.shape != (frames, vehicles):
                raise ValueError(
                    "EpisodeStoreV2 heading_valid_mask must have shape "
                    f"{(frames, vehicles)}."
                )
            if heading_derivation.shape != (frames, vehicles):
                raise ValueError(
                    "EpisodeStoreV2 heading_derivation must have shape "
                    f"{(frames, vehicles)}."
                )
            if np.any(heading_valid_mask[~active_mask]) or np.any(
                heading_derivation[~active_mask] != -1
            ):
                raise ValueError(
                    "EpisodeStoreV2 inactive states require heading_valid_mask=false "
                    "and heading_derivation=-1."
                )
            valid_active = active_mask & heading_valid_mask
            invalid_active = active_mask & ~heading_valid_mask
            if np.any(~np.isin(heading_derivation[valid_active], [1, 2, 3])):
                raise ValueError(
                    "EpisodeStoreV2 valid active headings require "
                    "heading_derivation in {1,2,3}."
                )
            if np.any(heading_derivation[invalid_active] != 0):
                raise ValueError(
                    "EpisodeStoreV2 invalid active headings require "
                    "heading_derivation=0."
                )
            if road_valid_mask is not None:
                invalid_heading_bit = np.uint16(
                    ELIGIBILITY_REASON_BITS["invalid_pose_or_heading"]
                )
                if np.any(
                    np.bitwise_and(
                        eligibility_reason_mask[invalid_active], invalid_heading_bit
                    )
                    == 0
                ):
                    raise ValueError(
                        "EpisodeStoreV2 invalid active headings must include "
                        "eligibility reason bit 64."
                    )
                if np.any(training_eligible_mask & ~heading_valid_mask):
                    raise ValueError(
                        "EpisodeStoreV2 training-eligible states require valid headings."
                    )
        if source_front_center_xy_m is not None:
            if source_front_center_xy_m.shape != (frames, vehicles, 2) or not np.all(
                np.isfinite(source_front_center_xy_m)
            ):
                raise ValueError(
                    "EpisodeStoreV2 source_front_center_xy_m must be finite with "
                    f"shape {(frames, vehicles, 2)}."
                )
            unit_heading = np.stack(
                (np.cos(heading_rad), np.sin(heading_rad)), axis=-1
            )
            reconstructed_front = states[:, :, :2] + (
                0.5 * dimensions_m[None, :, 0, None] * unit_heading
            )
            if not np.allclose(
                reconstructed_front[active_mask & heading_valid_mask],
                source_front_center_xy_m[active_mask & heading_valid_mask],
                rtol=0.0,
                atol=1e-9,
            ):
                max_error = float(
                    np.max(
                        np.abs(
                            reconstructed_front[active_mask & heading_valid_mask]
                            - source_front_center_xy_m[active_mask & heading_valid_mask]
                        )
                    )
                )
                raise ValueError(
                    "EpisodeStoreV2 I-80 front-center/body-center reconstruction "
                    f"mismatch (max_abs_error_m={max_error:.12g})."
                )
        if contiguous_track_instance is not None:
            if contiguous_track_instance.shape != (frames, vehicles):
                raise ValueError(
                    "EpisodeStoreV2 contiguous_track_instance must have shape "
                    f"{(frames, vehicles)}."
                )
            if np.any(contiguous_track_instance[~active_mask] != -1):
                raise ValueError(
                    "EpisodeStoreV2 inactive states require contiguous_track_instance=-1."
                )
            if np.any(contiguous_track_instance[active_mask] <= 0):
                raise ValueError(
                    "EpisodeStoreV2 active states require a positive contiguous track instance."
                )
            consecutive = active_mask[1:] & active_mask[:-1]
            if np.any(
                contiguous_track_instance[1:][consecutive]
                != contiguous_track_instance[:-1][consecutive]
            ):
                raise ValueError(
                    "EpisodeStoreV2 contiguous track identity changed across consecutive frames."
                )
        if int(timestamps_ms[0]) != entry.start_time_ms:
            raise ValueError(
                f"EpisodeStoreV2 start_time_ms mismatch for {episode_id!r}: "
                f"{int(timestamps_ms[0])} != {entry.start_time_ms}."
            )
        frozen_arrays = (
            timestamps_ms.astype(np.int64, copy=False),
            vehicle_ids.astype(np.int64, copy=False),
            dimensions_m,
            states,
            active_mask,
            provider_mask.astype(np.int8, copy=False),
        )
        optional_frozen_arrays = (
            controlled_vehicle_eligible_mask,
            road_valid_mask,
            training_eligible_mask,
            eligibility_reason_mask,
            source_front_center_xy_m,
            heading_rad,
            heading_valid_mask,
            heading_derivation,
            contiguous_track_instance,
        )
        for array in (*frozen_arrays, *optional_frozen_arrays):
            if array is None:
                continue
            array.setflags(write=False)
        episode = EpisodeDataV2(entry, *frozen_arrays, *optional_frozen_arrays)
        if self.max_cached_episodes > 0:
            self._cache[episode_id] = episode
            self._cache.move_to_end(episode_id)
            while len(self._cache) > self.max_cached_episodes:
                self._cache.popitem(last=False)
        return episode


def refine_valid_ids_by_episode(
    raw_valid_ids: dict[str, np.ndarray],
    traj_all: dict[str, dict[Any, Any]],
    *,
    min_occupancy: float,
    data_dt: float = 0.1,
) -> dict[str, np.ndarray]:
    refined: dict[str, np.ndarray] = {}
    for episode_name, veh_dict in traj_all.items():
        candidate_ids = raw_valid_ids.get(episode_name, veh_dict.keys())
        filtered_ids = []
        for veh_id in candidate_ids:
            meta = veh_dict.get(int(veh_id))
            if meta is None:
                continue
            traj = np.asarray(meta.get("trajectory", []), dtype=float)
            if trajectory_has_min_continuous_occupancy(
                traj,
                min_presence_ratio=min_occupancy,
                data_dt=data_dt,
            ):
                filtered_ids.append(int(veh_id))
        refined[episode_name] = np.asarray(filtered_ids, dtype=np.int64)
    return refined


def load_prebuilt_data(
    episode_root: str,
    scene: str,
    prebuilt_split: str,
    *,
    min_occupancy: float,
    cache: PrebuiltCache,
) -> tuple[
    str,
    dict[str, np.ndarray],
    dict[str, dict[Any, Any]],
    list[str],
]:
    episode_root_abs = os.path.abspath(episode_root)
    split = str(prebuilt_split)
    cache_key = (episode_root_abs, scene, split, float(min_occupancy))
    prebuilt_dir = os.path.join(episode_root_abs, scene, "prebuilt")

    cached = cache.get(cache_key)
    if cached is None:
        veh_ids_path = os.path.join(prebuilt_dir, f"veh_ids_{split}.npy")
        traj_path = os.path.join(prebuilt_dir, f"trajectory_{split}.npy")
        if not os.path.exists(veh_ids_path):
            raise FileNotFoundError(
                f"Missing prebuilt vehicle id file for split={split!r}: {veh_ids_path}"
            )
        if not os.path.exists(traj_path):
            raise FileNotFoundError(
                f"Missing prebuilt trajectory file for split={split!r}: {traj_path}"
            )
        raw_valid_ids = np.load(veh_ids_path, allow_pickle=True).item()
        traj_all = np.load(traj_path, allow_pickle=True).item()
        valid_ids = refine_valid_ids_by_episode(
            raw_valid_ids,
            traj_all,
            min_occupancy=min_occupancy,
        )
        episodes = sorted(traj_all.keys())
        cached = (valid_ids, traj_all, episodes)
        cache[cache_key] = cached

    valid_ids_by_episode, traj_all_by_episode, episodes = cached
    return prebuilt_dir, valid_ids_by_episode, traj_all_by_episode, episodes
