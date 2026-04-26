from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List, Tuple

import numpy as np

from highway_env.road.lane import AbstractLane, LineType, StraightLane, lane_from_config
from highway_env.vehicle.objects import Landmark


if TYPE_CHECKING:
    from highway_env.vehicle import kinematics, objects

logger = logging.getLogger(__name__)

LaneIndex = Tuple[str, str, int]
Route = List[LaneIndex]


class RoadNetwork:
    graph: dict[str, dict[str, list[AbstractLane]]]

    def __init__(self):
        self.graph = {}

    def add_lane(self, _from: str, _to: str, lane: AbstractLane) -> None:
        """
        A lane is encoded as an edge in the road network.

        :param _from: the node at which the lane starts.
        :param _to: the node at which the lane ends.
        :param AbstractLane lane: the lane geometry.
        """
        if _from not in self.graph:
            self.graph[_from] = {}
        if _to not in self.graph[_from]:
            self.graph[_from][_to] = []
        self.graph[_from][_to].append(lane)

    def get_lane(self, index: LaneIndex) -> AbstractLane:
        """
        Get the lane geometry corresponding to a given index in the road network.

        :param index: a tuple (origin node, destination node, lane id on the road).
        :return: the corresponding lane geometry.
        """
        _from, _to, _id = index
        if _id is None:
            pass
        if _id is None and len(self.graph[_from][_to]) == 1:
            _id = 0
        return self.graph[_from][_to][_id]

    def get_closest_lane_index(
        self, position: np.ndarray, heading: float | None = None
    ) -> LaneIndex:
        """
        Get the index of the lane closest to a world position.

        :param position: a world position [m].
        :param heading: a heading angle [rad].
        :return: the index of the closest lane.
        """
        indexes, distances = [], []
        for _from, to_dict in self.graph.items():
            for _to, lanes in to_dict.items():
                for _id, l in enumerate(lanes):
                    distances.append(l.distance_with_heading(position, heading))
                    indexes.append((_from, _to, _id))
        return indexes[int(np.argmin(distances))]

    def next_lane(
        self,
        current_index: LaneIndex,
        route: Route = None,
        position: np.ndarray = None,
        np_random: np.random.RandomState = np.random,
    ) -> LaneIndex:
        """
        Get the index of the next lane that should be followed after finishing the current lane.

        - If a plan is available and matches with current lane, follow it.
        - Else, pick next road randomly.
        - If it has the same number of lanes as current road, stay in the same lane.
        - Else, pick next road's closest lane.
        :param current_index: the index of the current target lane.
        :param route: the planned route, if any.
        :param position: the vehicle position.
        :param np_random: a source of randomness.
        :return: the index of the next lane to be followed when current lane is finished.
        """
        _from, _to, _id = current_index
        next_to = next_id = None
        # Pick next road according to planned route
        if route:
            if (
                route[0][:2] == current_index[:2]
            ):  # We just finished the first step of the route, drop it.
                route.pop(0)
            if (
                route and route[0][0] == _to
            ):  # Next road in route is starting at the end of current road.
                _, next_to, next_id = route[0]
            elif route:
                logger.warning(
                    "Route {} does not start after current road {}.".format(
                        route[0], current_index
                    )
                )

        # Compute current projected (desired) position
        long, lat = self.get_lane(current_index).local_coordinates(position)
        projected_position = self.get_lane(current_index).position(long, lateral=0)
        # If next route is not known
        if not next_to:
            # Pick the one with the closest lane to projected target position
            try:
                lanes_dists = [
                    (
                        next_to,
                        *self.next_lane_given_next_road(
                            _from, _to, _id, next_to, next_id, projected_position
                        ),
                    )
                    for next_to in self.graph[_to].keys()
                ]  # (next_to, next_id, distance)
                next_to, next_id, _ = min(lanes_dists, key=lambda x: x[-1])
            except KeyError:
                return current_index
        else:
            # If it is known, follow it and get the closest lane
            next_id, _ = self.next_lane_given_next_road(
                _from, _to, _id, next_to, next_id, projected_position
            )
        return _to, next_to, next_id

    def next_lane_given_next_road(
        self,
        _from: str,
        _to: str,
        _id: int,
        next_to: str,
        next_id: int,
        position: np.ndarray,
    ) -> tuple[int, float]:
        # If next road has same number of lane, stay on the same lane
        if len(self.graph[_from][_to]) == len(self.graph[_to][next_to]):
            if next_id is None:
                next_id = _id
        # Else, pick closest lane
        else:
            lanes = range(len(self.graph[_to][next_to]))
            next_id = min(
                lanes, key=lambda l: self.get_lane((_to, next_to, l)).distance(position)
            )
        return next_id, self.get_lane((_to, next_to, next_id)).distance(position)

    def bfs_paths(self, start: str, goal: str) -> list[list[str]]:
        """
        Breadth-first search of all routes from start to goal.

        :param start: starting node
        :param goal: goal node
        :return: list of paths from start to goal.
        """
        queue = [(start, [start])]
        while queue:
            (node, path) = queue.pop(0)
            if node not in self.graph:
                yield []
            for _next in sorted(
                [key for key in self.graph[node].keys() if key not in path]
            ):
                if _next == goal:
                    yield path + [_next]
                elif _next in self.graph:
                    queue.append((_next, path + [_next]))

    def shortest_path(self, start: str, goal: str) -> list[str]:
        """
        Breadth-first search of shortest path from start to goal.

        :param start: starting node
        :param goal: goal node
        :return: shortest path from start to goal.
        """
        return next(self.bfs_paths(start, goal), [])

    def all_side_lanes(self, lane_index: LaneIndex) -> list[LaneIndex]:
        """
        :param lane_index: the index of a lane.
        :return: all lanes belonging to the same road.
        """
        return [
            (lane_index[0], lane_index[1], i)
            for i in range(len(self.graph[lane_index[0]][lane_index[1]]))
        ]

    def side_lanes(self, lane_index: LaneIndex) -> list[LaneIndex]:
        """
        :param lane_index: the index of a lane.
        :return: indexes of lanes next to a an input lane, to its right or left.
        """
        _from, _to, _id = lane_index
        lanes = []
        if _id > 0:
            lanes.append((_from, _to, _id - 1))
        if _id < len(self.graph[_from][_to]) - 1:
            lanes.append((_from, _to, _id + 1))
        return lanes

    @staticmethod
    def is_same_road(
        lane_index_1: LaneIndex, lane_index_2: LaneIndex, same_lane: bool = False
    ) -> bool:
        """Is lane 1 in the same road as lane 2?"""
        return lane_index_1[:2] == lane_index_2[:2] and (
            not same_lane or lane_index_1[2] == lane_index_2[2]
        )

    @staticmethod
    def is_leading_to_road(
        lane_index_1: LaneIndex, lane_index_2: LaneIndex, same_lane: bool = False
    ) -> bool:
        """Is lane 1 leading to of lane 2?"""
        return lane_index_1[1] == lane_index_2[0] and (
            not same_lane or lane_index_1[2] == lane_index_2[2]
        )

    def is_connected_road(
        self,
        lane_index_1: LaneIndex,
        lane_index_2: LaneIndex,
        route: Route = None,
        same_lane: bool = False,
        depth: int = 0,
    ) -> bool:
        """
        Is the lane 2 leading to a road within lane 1's route?

        Vehicles on these lanes must be considered for collisions.
        :param lane_index_1: origin lane
        :param lane_index_2: target lane
        :param route: route from origin lane, if any
        :param same_lane: compare lane id
        :param depth: search depth from lane 1 along its route
        :return: whether the roads are connected
        """
        if RoadNetwork.is_same_road(
            lane_index_2, lane_index_1, same_lane
        ) or RoadNetwork.is_leading_to_road(lane_index_2, lane_index_1, same_lane):
            return True
        if depth > 0:
            if route and route[0][:2] == lane_index_1[:2]:
                # Route is starting at current road, skip it
                return self.is_connected_road(
                    lane_index_1, lane_index_2, route[1:], same_lane, depth
                )
            elif route and route[0][0] == lane_index_1[1]:
                # Route is continuing from current road, follow it
                return self.is_connected_road(
                    route[0], lane_index_2, route[1:], same_lane, depth - 1
                )
            else:
                # Recursively search all roads at intersection
                _from, _to, _id = lane_index_1
                return any(
                    [
                        self.is_connected_road(
                            (_to, l1_to, _id), lane_index_2, route, same_lane, depth - 1
                        )
                        for l1_to in self.graph.get(_to, {}).keys()
                    ]
                )
        return False

    def lanes_list(self) -> list[AbstractLane]:
        return [
            lane for to in self.graph.values() for ids in to.values() for lane in ids
        ]

    def lanes_dict(self) -> dict[str, AbstractLane]:
        return {
            (from_, to_, i): lane
            for from_, tos in self.graph.items()
            for to_, ids in tos.items()
            for i, lane in enumerate(ids)
        }

    @staticmethod
    def straight_road_network(
        lanes: int = 4,
        start: float = 0.0,
        length: float = 10000.0,
        angle: float = 0.0,
        speed_limit: float = 30.0,
        nodes_str: tuple[str, str] | None = None,
        net: RoadNetwork | None = None,
    ) -> RoadNetwork:
        net = net or RoadNetwork()
        nodes_str = nodes_str or ("0", "1")
        for lane in range(lanes):
            origin = np.array([start, lane * StraightLane.DEFAULT_WIDTH])
            end = np.array([start + length, lane * StraightLane.DEFAULT_WIDTH])
            rotation = np.array(
                [[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]]
            )
            origin = rotation @ origin
            end = rotation @ end
            line_types = [
                LineType.CONTINUOUS_LINE if lane == 0 else LineType.STRIPED,
                LineType.CONTINUOUS_LINE if lane == lanes - 1 else LineType.NONE,
            ]
            net.add_lane(
                *nodes_str,
                StraightLane(
                    origin, end, line_types=line_types, speed_limit=speed_limit
                ),
            )
        return net

    def position_heading_along_route(
        self,
        route: Route,
        longitudinal: float,
        lateral: float,
        current_lane_index: LaneIndex,
    ) -> tuple[np.ndarray, float]:
        """
        Get the absolute position and heading along a route composed of several lanes at some local coordinates.

        :param route: a planned route, list of lane indexes
        :param longitudinal: longitudinal position
        :param lateral: : lateral position
        :param current_lane_index: current lane index of the vehicle
        :return: position, heading
        """

        def _get_route_head_with_id(route_):
            lane_index_ = route_[0]
            if lane_index_[2] is None:
                # We know which road segment will be followed by the vehicle, but not which lane.
                # Hypothesis: the vehicle will keep the same lane_id as the current one.
                id_ = (
                    current_lane_index[2]
                    if current_lane_index[2]
                    < len(self.graph[current_lane_index[0]][current_lane_index[1]])
                    else 0
                )
                lane_index_ = (lane_index_[0], lane_index_[1], id_)
            return lane_index_

        lane_index = _get_route_head_with_id(route)
        while len(route) > 1 and longitudinal > self.get_lane(lane_index).length:
            longitudinal -= self.get_lane(lane_index).length
            route = route[1:]
            lane_index = _get_route_head_with_id(route)

        return self.get_lane(lane_index).position(longitudinal, lateral), self.get_lane(
            lane_index
        ).heading_at(longitudinal)

    def random_lane_index(self, np_random: np.random.RandomState) -> LaneIndex:
        _from = np_random.choice(list(self.graph.keys()))
        _to = np_random.choice(list(self.graph[_from].keys()))
        _id = np_random.integers(len(self.graph[_from][_to]))
        return _from, _to, _id

    @classmethod
    def from_config(cls, config: dict) -> None:
        net = cls()
        for _from, to_dict in config.items():
            net.graph[_from] = {}
            for _to, lanes_dict in to_dict.items():
                net.graph[_from][_to] = []
                for lane_dict in lanes_dict:
                    net.graph[_from][_to].append(lane_from_config(lane_dict))
        return net

    def to_config(self) -> dict:
        graph_dict = {}
        for _from, to_dict in self.graph.items():
            graph_dict[_from] = {}
            for _to, lanes in to_dict.items():
                graph_dict[_from][_to] = []
                for lane in lanes:
                    graph_dict[_from][_to].append(lane.to_config())
        return graph_dict


class Road:
    """A road is a set of lanes, and a set of vehicles driving on these lanes."""

    def __init__(
        self,
        network: RoadNetwork = None,
        vehicles: list[kinematics.Vehicle] = None,
        road_objects: list[objects.RoadObject] = None,
        np_random: np.random.RandomState = None,
        record_history: bool = False,
        use_query_fast_path: bool = False,
        query_cell_size: float = 25.0,
    ) -> None:
        """
        New road.

        :param network: the road network describing the lanes
        :param vehicles: the vehicles driving on the road
        :param road_objects: the objects on the road including obstacles and landmarks
        :param np.random.RandomState np_random: a random number generator for vehicle behaviour
        :param record_history: whether the recent trajectories of vehicles should be recorded for display
        """
        self.network = network
        self.vehicles = vehicles or []
        self.objects = road_objects or []
        self.np_random = np_random if np_random else np.random.RandomState()
        self.record_history = record_history
        self.use_query_fast_path = bool(use_query_fast_path)
        self.query_cell_size = float(query_cell_size)
        self._query_cache_dirty = True
        self._cached_counts = (-1, -1)
        self._spatial_index: dict[tuple[int, int], list[objects.RoadObject]] = {}
        self._edge_object_index: dict[tuple[str, str], list[objects.RoadObject]] = {}
        self._edge_bounds_cache = self._collect_edge_bounds()

    def _invalidate_query_cache(self) -> None:
        self._query_cache_dirty = True

    def _cell_key(self, position: np.ndarray) -> tuple[int, int]:
        return (
            int(np.floor(float(position[0]) / self.query_cell_size)),
            int(np.floor(float(position[1]) / self.query_cell_size)),
        )

    def _lane_bounds(self, lane: AbstractLane) -> tuple[float, float, float, float] | None:
        try:
            length = float(getattr(lane, "length", 0.0))
            if not np.isfinite(length) or length <= 0.0:
                sample_s = np.array([0.0], dtype=float)
            else:
                num = max(3, min(9, int(np.ceil(length / 40.0)) + 1))
                sample_s = np.linspace(0.0, length, num=num, dtype=float)

            points = []
            for longitudinal in sample_s:
                width = float(lane.width_at(float(longitudinal)))
                center = lane.position(float(longitudinal), 0.0)
                if center is None or not np.all(np.isfinite(center)) or not np.isfinite(width):
                    continue
                center = np.asarray(center, dtype=float)
                points.append(center)
                points.append(np.asarray(lane.position(float(longitudinal), 0.5 * width), dtype=float))
                points.append(np.asarray(lane.position(float(longitudinal), -0.5 * width), dtype=float))
            if not points:
                return None
            stacked = np.vstack(points)
            return (
                float(np.min(stacked[:, 0])),
                float(np.max(stacked[:, 0])),
                float(np.min(stacked[:, 1])),
                float(np.max(stacked[:, 1])),
            )
        except Exception:
            return None

    def _collect_edge_bounds(
        self,
    ) -> dict[tuple[str, str], tuple[float, float, float, float]] | None:
        if self.network is None:
            return None
        edge_bounds: dict[tuple[str, str], tuple[float, float, float, float]] = {}
        try:
            for src, tos in self.network.graph.items():
                for dst, lanes in tos.items():
                    bounds = [self._lane_bounds(lane) for lane in lanes]
                    bounds = [b for b in bounds if b is not None]
                    if not bounds:
                        continue
                    edge_bounds[(src, dst)] = (
                        min(b[0] for b in bounds),
                        max(b[1] for b in bounds),
                        min(b[2] for b in bounds),
                        max(b[3] for b in bounds),
                    )
        except Exception:
            return None
        return edge_bounds if edge_bounds else None

    def _ensure_query_cache(self) -> None:
        counts = (len(self.vehicles), len(self.objects))
        if (
            not self.use_query_fast_path
            or (
                not self._query_cache_dirty
                and counts == self._cached_counts
            )
        ):
            return

        self._spatial_index = {}
        self._edge_object_index = {}
        for obj in list(self.vehicles) + list(self.objects):
            position = getattr(obj, "position", None)
            if position is None or not np.all(np.isfinite(position)):
                continue
            self._spatial_index.setdefault(
                self._cell_key(np.asarray(position, dtype=float)), []
            ).append(obj)

            lane_index = getattr(obj, "lane_index", None)
            if lane_index is None or lane_index is np.nan:
                continue
            try:
                edge = (lane_index[0], lane_index[1])
            except Exception:
                continue
            self._edge_object_index.setdefault(edge, []).append(obj)

        self._cached_counts = counts
        self._query_cache_dirty = False

    def _close_objects_to_legacy(
        self,
        vehicle: kinematics.Vehicle,
        distance: float,
        count: int | None = None,
        see_behind: bool = True,
        sort: bool = True,
        vehicles_only: bool = False,
    ) -> object:
        candidates = list(self.vehicles)
        if not vehicles_only:
            candidates += list(self.objects)
        return self._filter_close_candidates(
            vehicle,
            candidates,
            distance,
            count=count,
            see_behind=see_behind,
            sort=sort,
            vehicles_only=vehicles_only,
        )

    def _filter_close_candidates(
        self,
        vehicle: kinematics.Vehicle,
        candidates: list[objects.RoadObject],
        distance: float,
        count: int | None = None,
        see_behind: bool = True,
        sort: bool = True,
        vehicles_only: bool = False,
    ) -> list[objects.RoadObject]:
        objects_ = [
            obj
            for obj in candidates
            if obj is not vehicle
            and (obj in self.vehicles or not vehicles_only)
            and np.linalg.norm(obj.position - vehicle.position) < distance
            and (see_behind or -2 * vehicle.LENGTH < vehicle.lane_distance_to(obj))
        ]

        if sort:
            objects_ = sorted(
                objects_, key=lambda obj: abs(vehicle.lane_distance_to(obj))
            )
        if count:
            objects_ = objects_[:count]
        return objects_

    def _candidate_objects_near(
        self,
        position: np.ndarray,
        distance: float,
        *,
        include_objects: bool = True,
    ) -> list[objects.RoadObject]:
        self._ensure_query_cache()
        if not self._spatial_index:
            return []

        cell_radius = max(1, int(np.ceil(float(distance) / self.query_cell_size)))
        cx, cy = self._cell_key(np.asarray(position, dtype=float))
        candidates: list[objects.RoadObject] = []
        seen: set[int] = set()
        for dx in range(-cell_radius, cell_radius + 1):
            for dy in range(-cell_radius, cell_radius + 1):
                for obj in self._spatial_index.get((cx + dx, cy + dy), []):
                    if not include_objects and obj not in self.vehicles:
                        continue
                    key = id(obj)
                    if key not in seen:
                        seen.add(key)
                        candidates.append(obj)
        return candidates

    def _neighbour_vehicles_legacy(
        self, vehicle: kinematics.Vehicle, lane_index: LaneIndex = None
    ) -> tuple[kinematics.Vehicle | None, kinematics.Vehicle | None]:
        lane_index = lane_index or vehicle.lane_index
        if not lane_index:
            return None, None
        lane = self.network.get_lane(lane_index)
        s = self.network.get_lane(lane_index).local_coordinates(vehicle.position)[0]
        s_front = s_rear = None
        v_front = v_rear = None
        for v in self.vehicles + self.objects:
            if (
                v is not vehicle
                and not isinstance(v, Landmark)
                and self._is_neighbour_candidate(v)
            ):
                s_v, lat_v = lane.local_coordinates(v.position)
                if not lane.on_lane(v.position, s_v, lat_v, margin=0.05):
                    continue
                if s <= s_v and (s_front is None or s_v <= s_front):
                    s_front = s_v
                    v_front = v
                if s_v < s and (s_rear is None or s_v > s_rear):
                    s_rear = s_v
                    v_rear = v
        return v_front, v_rear

    @staticmethod
    def _is_neighbour_candidate(obj) -> bool:
        if getattr(obj, "remove_from_road", False):
            return False
        if hasattr(obj, "visible") and not bool(getattr(obj, "visible", True)):
            return False
        if hasattr(obj, "appear") and not bool(getattr(obj, "appear", True)):
            return False
        if hasattr(obj, "scene_collection_is_active") and not bool(
            getattr(obj, "scene_collection_is_active", True)
        ):
            return False
        if float(getattr(obj, "LENGTH", 0.0)) <= 0.0 or float(getattr(obj, "WIDTH", 0.0)) <= 0.0:
            return False
        return True

    def _candidate_objects_for_lane(
        self,
        lane_index: LaneIndex,
        padding: float = 10.0,
    ) -> list[objects.RoadObject] | None:
        self._ensure_query_cache()
        if not self._edge_object_index or self._edge_bounds_cache is None:
            return None

        edge = (lane_index[0], lane_index[1])
        query_bounds = self._edge_bounds_cache.get(edge)
        if query_bounds is None:
            return None

        min_x, max_x, min_y, max_y = query_bounds
        min_x -= padding
        max_x += padding
        min_y -= padding
        max_y += padding

        candidates: list[objects.RoadObject] = []
        seen: set[int] = set()
        for candidate_edge, bounds in self._edge_bounds_cache.items():
            cmin_x, cmax_x, cmin_y, cmax_y = bounds
            if cmax_x < min_x or cmin_x > max_x or cmax_y < min_y or cmin_y > max_y:
                continue
            for obj in self._edge_object_index.get(candidate_edge, []):
                key = id(obj)
                if key not in seen:
                    seen.add(key)
                    candidates.append(obj)
        return candidates

    def close_objects_to(
        self,
        vehicle: kinematics.Vehicle,
        distance: float,
        count: int | None = None,
        see_behind: bool = True,
        sort: bool = True,
        vehicles_only: bool = False,
    ) -> object:
        total_objects = (
            len(self.vehicles)
            if vehicles_only
            else len(self.vehicles) + len(self.objects)
        )
        if (not self.use_query_fast_path) or total_objects < 64:
            return self._close_objects_to_legacy(
                vehicle,
                distance,
                count=count,
                see_behind=see_behind,
                sort=sort,
                vehicles_only=vehicles_only,
            )

        candidates = self._candidate_objects_near(
            vehicle.position,
            distance,
            include_objects=not vehicles_only,
        )
        if len(candidates) > 0.6 * max(1, total_objects):
            return self._close_objects_to_legacy(
                vehicle,
                distance,
                count=count,
                see_behind=see_behind,
                sort=sort,
                vehicles_only=vehicles_only,
            )
        return self._filter_close_candidates(
            vehicle,
            candidates,
            distance,
            count=count,
            see_behind=see_behind,
            sort=sort,
            vehicles_only=vehicles_only,
        )

    def close_vehicles_to(
        self,
        vehicle: kinematics.Vehicle,
        distance: float,
        count: int | None = None,
        see_behind: bool = True,
        sort: bool = True,
    ) -> object:
        return self.close_objects_to(
            vehicle, distance, count, see_behind, sort, vehicles_only=True
        )

    def act(self) -> None:
        """Decide the actions of each entity on the road."""
        self._ensure_query_cache()
        for vehicle in self.vehicles:
            vehicle.act()

    def step(self, dt: float) -> None:
        """
        Step the dynamics of each entity on the road.

        :param dt: timestep [s]
        """
        for vehicle in self.vehicles:
            vehicle.step(dt)
        for i, vehicle in enumerate(self.vehicles):
            for other in self.vehicles[i + 1 :]:
                vehicle.handle_collisions(other, dt)
            for other in self.objects:
                vehicle.handle_collisions(other, dt)
        self._invalidate_query_cache()

    def neighbour_vehicles(
        self, vehicle: kinematics.Vehicle, lane_index: LaneIndex = None
    ) -> tuple[kinematics.Vehicle | None, kinematics.Vehicle | None]:
        """
        Find the preceding and following vehicles of a given vehicle.

        :param vehicle: the vehicle whose neighbours must be found
        :param lane_index: the lane on which to look for preceding and following vehicles.
                     It doesn't have to be the current vehicle lane but can also be another lane, in which case the
                     vehicle is projected on it considering its local coordinates in the lane.
        :return: its preceding vehicle, its following vehicle
        """
        lane_index = lane_index or vehicle.lane_index
        if not lane_index:
            return None, None
        total_objects = len(self.vehicles) + len(self.objects)
        if (not self.use_query_fast_path) or total_objects < 64:
            return self._neighbour_vehicles_legacy(vehicle, lane_index=lane_index)

        lane = self.network.get_lane(lane_index)
        s = lane.local_coordinates(vehicle.position)[0]
        s_front = s_rear = None
        v_front = v_rear = None
        candidates = self._candidate_objects_for_lane(
            lane_index,
            padding=max(10.0, float(vehicle.LENGTH) * 2.0),
        )
        if not candidates:
            return self._neighbour_vehicles_legacy(vehicle, lane_index=lane_index)

        for v in candidates:
            if (
                v is vehicle
                or isinstance(v, Landmark)
                or not self._is_neighbour_candidate(v)
            ):
                continue
            s_v, lat_v = lane.local_coordinates(v.position)
            if not lane.on_lane(v.position, s_v, lat_v, margin=0.05):
                continue
            if s <= s_v and (s_front is None or s_v <= s_front):
                s_front = s_v
                v_front = v
            if s_v < s and (s_rear is None or s_v > s_rear):
                s_rear = s_v
                v_rear = v
        if v_front is None and v_rear is None:
            return self._neighbour_vehicles_legacy(vehicle, lane_index=lane_index)
        return v_front, v_rear

    def __repr__(self):
        return self.vehicles.__repr__()
