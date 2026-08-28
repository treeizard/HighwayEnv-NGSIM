(graphics)=

% py:currentmodule::highway_env.envs.common.graphics

# Graphics

Environment rendering is done with [pygame](https://www.pygame.org/news), which must be {ref}`installed separately <installation>`.

A window is created at the first call of `env.render()`. Its dimensions can be configured:

```python
env = gym.make(
    "roundabout-v0",
    config={
        "screen_width": 640,
        "screen_height": 480
    }
)
env.reset()
env.render()
```

## World surface

The simulation is rendered in a {py:class}`~highway_env.envs.common.graphics.RoadSurface` pygame surface, which defines the location and zoom of the rendered location.
By default, the rendered area is always centered on the ego-vehicle.
Its initial scale and offset can be set with the `"scaling"` and `"centering_position"` configurations, and can also be
updated during simulation using the O,L keys and K,M keys, respectively.

## Scene graphics

- Roads are rendered in the {py:class}`~highway_env.road.graphics.RoadGraphics` class.
- Vehicles are rendered in the {py:class}`~highway_env.vehicle.graphics.VehicleGraphics` class.

### Longitudinal marking profiles

`RoadGeometryV3` polyline lanes may override their two base `line_types` over
longitudinal intervals. This supports a solid approach followed by an open or
striped merge without adding artificial road-network edges:

```json
"marking_profile": [
  {"start_s_m": 0.0, "end_s_m": 40.0, "line_types": [3, 3]},
  {"start_s_m": 40.0, "end_s_m": 75.0, "line_types": [0, 1]}
]
```

Intervals must be ordered, positive-length, exactly contiguous, and partition
the full derived lane length from `0.0` to `lane.length`. Each `line_types`
pair uses the base lane ordering and values: `0` none, `1` striped, `2`
continuous, and `3` sampled continuous line. If `marking_profile` is absent,
the renderer uses the base `line_types` unchanged. Marking profiles affect
rendering only; lane geometry, connectivity, and off-road checks are unchanged.

## API

```{eval-rst}
.. automodule:: highway_env.envs.common.graphics
    :members:
```

```{eval-rst}
.. automodule:: highway_env.road.graphics
    :members:
```

```{eval-rst}
.. automodule:: highway_env.vehicle.graphics
    :members:
```
