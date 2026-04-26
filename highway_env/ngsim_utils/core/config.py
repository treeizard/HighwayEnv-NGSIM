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

from copy import deepcopy

from highway_env.ngsim_utils.core.constants import (
    IDM_PARAMETER_PRESETS,
    SCENE_IDM_PARAMETER_KEY,
)


def deep_update(base: dict, override: dict) -> dict:
    """
    Recursively merge ``override`` into ``base`` and return the merged copy.
    """
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


def resolve_idm_parameters(scene: str, cfg: dict) -> dict:
    preset_key = SCENE_IDM_PARAMETER_KEY.get(str(scene), "US")
    preset = IDM_PARAMETER_PRESETS.get(preset_key, IDM_PARAMETER_PRESETS["US"])
    configured = cfg.get("idm_parameters")
    if configured:
        return deep_update(deepcopy(preset), configured)
    return deepcopy(preset)


def normalize_action_mode(cfg: dict, raw_config: dict | None = None) -> str:
    raw_config = raw_config or {}
    raw_action_cfg = raw_config.get("action", {})
    cfg_action_cfg = cfg.get("action", {})
    raw_action_type = str(raw_action_cfg.get("type", ""))
    cfg_action_type = str(cfg_action_cfg.get("type", ""))
    raw_action_mode = str(raw_config.get("action_mode", "")).lower()

    if raw_action_mode == "teleport":
        cfg["action_mode"] = "teleport"
        cfg["action"] = deep_update(cfg_action_cfg, raw_action_cfg)
        return "teleport"

    if raw_action_type == "MultiAgentAction" or cfg_action_type == "MultiAgentAction":
        nested_action = raw_action_cfg.get(
            "action_config",
            cfg_action_cfg.get("action_config", {}),
        )
        nested_type = str(nested_action.get("type", ""))
        if nested_type == "ContinuousAction":
            control_mode = "continuous"
        elif nested_type == "DiscreteSteerMetaAction":
            control_mode = "discrete"
        else:
            raise ValueError(
                "MultiAgentAction requires action_config.type to be "
                "'ContinuousAction' or 'DiscreteSteerMetaAction'."
            )
        cfg["action"] = deep_update(cfg_action_cfg, raw_action_cfg)
        return control_mode

    if "action_mode" in raw_config:
        control_mode = str(raw_config["action_mode"]).lower()
    else:
        action_type = str(raw_action_cfg.get("type", cfg_action_cfg.get("type", ""))).lower()
        if action_type == "continuousaction":
            control_mode = "continuous"
        elif action_type == "discretesteermetaaction":
            control_mode = "discrete"
        else:
            control_mode = str(cfg.get("action_mode", "continuous")).lower()

    action_types = {
        "continuous": "ContinuousAction",
        "discrete": "DiscreteSteerMetaAction",
        "teleport": "DiscreteSteerMetaAction",
    }
    if control_mode not in action_types:
        raise ValueError(f"Unknown action_mode={control_mode!r}")
    cfg["action"] = {"type": action_types[control_mode]}
    return control_mode
