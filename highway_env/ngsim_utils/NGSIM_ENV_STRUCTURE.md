# NGSimEnv Structure

This note describes how [`NGSimEnv`](/home/tao/Documents/Github_Projects/HighwayEnv-NGSIM/highway_env/envs/ngsim_env.py) is organized and how it interacts with the utilities in [`highway_env/ngsim_utils`](/home/tao/Documents/Github_Projects/HighwayEnv-NGSIM/highway_env/ngsim_utils).

## High-Level Layout

`NGSimEnv` is the main environment class. It is responsible for:

- loading prebuilt trajectory episodes
- selecting an episode and ego vehicle
- building the road for the selected scene
- spawning the ego vehicle and surrounding replay vehicles
- optionally running in `expert_test_mode`, where expert actions override external actions
- stepping the simulation and returning observations/info

## Main Modules

```mermaid
flowchart TD
    A[NGSimEnv\nhighway_env/envs/ngsim_env.py]
    B[helper_ngsim.py\nscene helpers, lane mapping,\ntrajectory loading helpers]
    C[gen_road.py\nroad-network builders]
    D[ego_vehicle.py\ncontrolled ego vehicle]
    E[obs_vehicle.py\nsurrounding replay vehicles]
    F[trajectory_gen.py\nraw trajectory conversion]
    G[trajectory_to_action.py\nexpert tracker and action mapping]

    A --> B
    A --> C
    A --> D
    A --> E
    A --> F
    A --> G
    E --> B
    E --> F
    D --> B
```

## NGSimEnv Responsibilities

```mermaid
flowchart TB
    A[NGSimEnv.__init__]
    B[_normalize_action_mode]
    C[_load_prebuilt_data]
    D[_reset]
    E[_load_trajectory]
    F[_create_road]
    G[_create_vehicles]
    H[_prepare_ego_trajectory]
    I[_setup_expert_tracker]
    J[_build_ego_vehicle]
    K[_spawn_surrounding_vehicles]
    L[step]
    M[_resolve_expert_action]

    A --> B
    A --> C
    D --> E
    D --> F
    D --> G
    G --> H
    H --> I
    G --> J
    G --> K
    L --> M
```

## Reset Lifecycle

The reset path builds a fresh simulation state from recorded data.

```mermaid
sequenceDiagram
    participant User
    participant Env as NGSimEnv
    participant Data as Prebuilt Trajectories
    participant Road as RoadNetwork/Road
    participant Ego as EgoVehicle
    participant Other as NGSIMVehicle[]

    User->>Env: reset()
    Env->>Data: select episode + ego id
    Env->>Road: build scene road
    Env->>Env: prepare ego trajectory
    alt expert_test_mode
        Env->>Env: setup expert tracker
    end
    Env->>Ego: create ego vehicle
    Env->>Other: spawn surrounding vehicles
    Env-->>User: observation, info
```

## Step Lifecycle

The environment supports two modes:

- normal mode: the caller-provided action is used
- `expert_test_mode`: the expert action is computed internally and replaces the caller-provided action

```mermaid
flowchart TD
    A[step(action)]
    B{expert_test_mode?}
    C[_resolve_expert_action]
    D[super().step(action)]
    E[road.act + road.step]
    F[obs/reward/terminated/truncated/info]
    G[attach expert info to info dict]

    A --> B
    B -- Yes --> C
    B -- No --> D
    C --> D
    D --> E
    E --> F
    F --> G
```

## Scene and Road Selection

Road creation is scene-driven.

```mermaid
flowchart LR
    A[scene config]
    B[ROAD_BUILDERS]
    C[create_ngsim_101_road]
    D[create_japanese_road]
    E[RoadNetwork]
    F[Road]

    A --> B
    B --> C
    B --> D
    C --> E
    D --> E
    E --> F
```

Current scene-related responsibilities:

- [`gen_road.py`](/home/tao/Documents/Github_Projects/HighwayEnv-NGSIM/highway_env/ngsim_utils/gen_road.py): builds road topology
- [`helper_ngsim.py`](/home/tao/Documents/Github_Projects/HighwayEnv-NGSIM/highway_env/ngsim_utils/helper_ngsim.py): maps `x` and dataset `lane_id` to valid road edges/lane indices

## Vehicle Roles

### EgoVehicle

[`EgoVehicle`](/home/tao/Documents/Github_Projects/HighwayEnv-NGSIM/highway_env/ngsim_utils/ego_vehicle.py) is the controlled vehicle used by the environment.

It supports:

- continuous low-level control
- discrete meta-actions
- lane-change cooldown and within-lane offset control
- scene-aware lane projection using road-edge helpers

### NGSIMVehicle

[`NGSIMVehicle`](/home/tao/Documents/Github_Projects/HighwayEnv-NGSIM/highway_env/ngsim_utils/obs_vehicle.py) is used for surrounding vehicles.

It supports:

- replaying logged trajectories
- appearing/disappearing from logged data
- replay-time lane assignment using recorded `lane_id`
- IDM/MOBIL handover when replay ends or replay becomes unsafe

## Expert Mode

In `expert_test_mode`, `NGSimEnv` computes an expert action each step using:

- [`setup_expert_tracker()`](/home/tao/Documents/Github_Projects/HighwayEnv-NGSIM/highway_env/ngsim_utils/helper_ngsim.py#L133)
- [`PurePursuitTracker`](/home/tao/Documents/Github_Projects/HighwayEnv-NGSIM/highway_env/ngsim_utils/trajectory_to_action.py)
- [`map_discrete_expert_action()`](/home/tao/Documents/Github_Projects/HighwayEnv-NGSIM/highway_env/ngsim_utils/trajectory_to_action.py)

```mermaid
flowchart TD
    A[expert_test_mode]
    B[PurePursuitTracker.step]
    C{control_mode}
    D[continuous expert action]
    E[discrete expert action]
    F[action sent to AbstractEnv.step]

    A --> B
    B --> C
    C -- continuous --> D
    C -- discrete --> E
    D --> F
    E --> F
```

Important design note:

- In `expert_test_mode`, external actions are deliberately ignored so the environment can test expert-action generation and replay quality directly.

## Supporting Data Flow

```mermaid
flowchart TD
    A[Prebuilt .npy files]
    B[_load_prebuilt_data]
    C[_traj_all_by_episode]
    D[_valid_ids_by_episode]
    E[_load_trajectory]
    F[trajectory_set]
    G[load_ego_trajectory / process_raw_trajectory]

    A --> B
    B --> C
    B --> D
    C --> E
    D --> E
    E --> F
    F --> G
```

## Key Files

- [`ngsim_env.py`](/home/tao/Documents/Github_Projects/HighwayEnv-NGSIM/highway_env/envs/ngsim_env.py): main environment orchestration
- [`helper_ngsim.py`](/home/tao/Documents/Github_Projects/HighwayEnv-NGSIM/highway_env/ngsim_utils/helper_ngsim.py): lane mapping and helper utilities
- [`gen_road.py`](/home/tao/Documents/Github_Projects/HighwayEnv-NGSIM/highway_env/ngsim_utils/gen_road.py): road builders
- [`ego_vehicle.py`](/home/tao/Documents/Github_Projects/HighwayEnv-NGSIM/highway_env/ngsim_utils/ego_vehicle.py): ego control logic
- [`obs_vehicle.py`](/home/tao/Documents/Github_Projects/HighwayEnv-NGSIM/highway_env/ngsim_utils/obs_vehicle.py): surrounding replay vehicles
- [`trajectory_gen.py`](/home/tao/Documents/Github_Projects/HighwayEnv-NGSIM/highway_env/ngsim_utils/trajectory_gen.py): trajectory conversion
- [`trajectory_to_action.py`](/home/tao/Documents/Github_Projects/HighwayEnv-NGSIM/highway_env/ngsim_utils/trajectory_to_action.py): expert tracking/action conversion

## Suggested Reading Order

1. [`ngsim_env.py`](/home/tao/Documents/Github_Projects/HighwayEnv-NGSIM/highway_env/envs/ngsim_env.py)
2. [`helper_ngsim.py`](/home/tao/Documents/Github_Projects/HighwayEnv-NGSIM/highway_env/ngsim_utils/helper_ngsim.py)
3. [`gen_road.py`](/home/tao/Documents/Github_Projects/HighwayEnv-NGSIM/highway_env/ngsim_utils/gen_road.py)
4. [`ego_vehicle.py`](/home/tao/Documents/Github_Projects/HighwayEnv-NGSIM/highway_env/ngsim_utils/ego_vehicle.py)
5. [`obs_vehicle.py`](/home/tao/Documents/Github_Projects/HighwayEnv-NGSIM/highway_env/ngsim_utils/obs_vehicle.py)
6. [`trajectory_to_action.py`](/home/tao/Documents/Github_Projects/HighwayEnv-NGSIM/highway_env/ngsim_utils/trajectory_to_action.py)
