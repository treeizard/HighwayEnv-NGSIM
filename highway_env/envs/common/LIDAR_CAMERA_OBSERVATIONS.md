# `LidarCameraObservations` Output

`LidarCameraObservations` returns a 3-tuple:

```python
(lidar_obs, lane_camera_obs, ego_state_obs)
```

## 1. `lidar_obs`

- Type: `np.ndarray`
- Shape: `(lidar_cells, 2)`
- Meaning per row:
  - `lidar_obs[i, 0]`: beam distance
  - `lidar_obs[i, 1]`: relative speed along beam direction

This is the existing `LidarObservation` output.

## 2. `lane_camera_obs`

- Type: `np.ndarray`
- Shape: `(camera_cells, 3)`
- Meaning per row:
  - `lane_camera_obs[i, 0]`: presence flag (`0` or `1`)
  - `lane_camera_obs[i, 1]`: detected lane-boundary point `x` in ego frame
  - `lane_camera_obs[i, 2]`: detected lane-boundary point `y` in ego frame

Notes:

- The lane camera is topology-only.
- It ignores dynamic objects.
- It only considers lane/road boundary points inside the configured forward field of view and range.
- Each angular camera cell keeps the nearest valid topology point.

## 3. `ego_state_obs`

- Type: `np.ndarray`
- Shape: `(4,)`
- Meaning:
  - `ego_state_obs[0]`: ego speed
  - `ego_state_obs[1]`: ego heading
  - `ego_state_obs[2]`: ego width
  - `ego_state_obs[3]`: ego length

## Example

```python
lidar_obs, lane_camera_obs, ego_state_obs = obs
```
