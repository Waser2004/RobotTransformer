from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

import numpy as np

from collision_detection import RobotCollisionModel


# Default joint limits in degrees.
# Update these values if your robot uses different mechanical limits.
JOINT_LIMITS_DEG = np.array(
    [
        [-120.0, 120.0],  # j0
        [-90.0, 90.0],    # j1
        [-180.0, 0.0],     # j2
        [-90.0, 90.0],    # j3
        [0.0, 180.0],     # j4
        [-90.0, 90.0],    # j5
    ],
    dtype=float,
)


@dataclass
class RRTTree:
    nodes: List[np.ndarray]
    parents: List[int]

    def __init__(self, root: np.ndarray):
        self.nodes = [np.asarray(root, dtype=float).copy()]
        self.parents = [-1]

    def add(self, config: np.ndarray, parent_index: int) -> int:
        self.nodes.append(np.asarray(config, dtype=float).copy())
        self.parents.append(int(parent_index))
        return len(self.nodes) - 1

    def nearest_index(self, target: np.ndarray) -> int:
        nodes = np.vstack(self.nodes)
        distances = np.linalg.norm(nodes - target, axis=1)
        return int(np.argmin(distances))

    def path_to_root(self, node_index: int) -> List[np.ndarray]:
        path: List[np.ndarray] = []
        while node_index != -1:
            path.append(self.nodes[node_index])
            node_index = self.parents[node_index]
        path.reverse()
        return path


def _coerce_config(config: Sequence[float], name: str) -> np.ndarray:
    arr = np.asarray(config, dtype=float).reshape(-1)
    if arr.shape[0] != 6:
        raise ValueError(f"{name} must have 6 joint values, got shape {arr.shape}")
    return arr


def _sample_uniform(joint_limits: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    low = joint_limits[:, 0]
    high = joint_limits[:, 1]
    return rng.uniform(low, high)


def _steer(from_config: np.ndarray, to_config: np.ndarray, step_size: float) -> np.ndarray:
    direction = to_config - from_config
    distance = float(np.linalg.norm(direction))
    if distance <= step_size:
        return to_config.copy()
    return from_config + (direction / distance) * step_size


def _edge_is_valid(
    collision_model: RobotCollisionModel,
    start: np.ndarray,
    end: np.ndarray,
    step_size: float,
    cube_position: Sequence[float] | None = None,
    cube_z_rotation: float | None = None,
) -> bool:
    if np.allclose(start, end):
        return collision_model.is_state_valid(
            start.tolist(),
            cube_position=cube_position,
            cube_z_rotation=cube_z_rotation,
        )
    return collision_model.is_edge_valid(
        start.tolist(),
        end.tolist(),
        step_size=step_size,
        cube_position=cube_position,
        cube_z_rotation=cube_z_rotation,
    )


def _extend(
    tree: RRTTree,
    target: np.ndarray,
    step_size: float,
    collision_model: RobotCollisionModel,
    collision_step: float,
    cube_position: Sequence[float] | None = None,
    cube_z_rotation: float | None = None,
) -> tuple[str, Optional[int]]:
    near_index = tree.nearest_index(target)
    near_config = tree.nodes[near_index]
    new_config = _steer(near_config, target, step_size)

    if not _edge_is_valid(
        collision_model,
        near_config,
        new_config,
        collision_step,
        cube_position=cube_position,
        cube_z_rotation=cube_z_rotation,
    ):
        return "trapped", None

    new_index = tree.add(new_config, near_index)
    if np.allclose(new_config, target):
        return "reached", new_index
    return "advanced", new_index


def _connect(
    tree: RRTTree,
    target: np.ndarray,
    step_size: float,
    collision_model: RobotCollisionModel,
    collision_step: float,
    cube_position: Sequence[float] | None = None,
    cube_z_rotation: float | None = None,
) -> tuple[str, Optional[int]]:
    status, new_index = _extend(
        tree,
        target,
        step_size,
        collision_model,
        collision_step,
        cube_position=cube_position,
        cube_z_rotation=cube_z_rotation,
    )
    while status == "advanced":
        status, new_index = _extend(
            tree,
            target,
            step_size,
            collision_model,
            collision_step,
            cube_position=cube_position,
            cube_z_rotation=cube_z_rotation,
        )
    return status, new_index


def rrt_connect(
    start: Sequence[float],
    goal: Sequence[float],
    collision_model: RobotCollisionModel,
    joint_limits: Optional[np.ndarray] = None,
    step_size: float = 7.5,
    collision_step: float = 2.0,
    max_iterations: int = 3000,
    goal_bias: float = 0.1,
    cube_position: Sequence[float] | None = None,
    cube_z_rotation: float | None = None,
    rng: Optional[np.random.Generator] = None,
) -> Optional[List[np.ndarray]]:
    """
    Plan a collision-free path with RRT-Connect in joint space.

    Returns a list of joint vectors (np.ndarray) or None if planning fails.
    """
    start_config = _coerce_config(start, "start")
    goal_config = _coerce_config(goal, "goal")

    if joint_limits is None:
        joint_limits = JOINT_LIMITS_DEG
    joint_limits = np.asarray(joint_limits, dtype=float)
    if joint_limits.shape != (6, 2):
        raise ValueError("joint_limits must be shape (6, 2)")

    rng = rng or np.random.default_rng()

    if not collision_model.is_state_valid(
        start_config.tolist(),
        cube_position=cube_position,
        cube_z_rotation=cube_z_rotation,
    ):
        return None
    if not collision_model.is_state_valid(
        goal_config.tolist(),
        cube_position=cube_position,
        cube_z_rotation=cube_z_rotation,
    ):
        return None

    tree_start = RRTTree(start_config)
    tree_goal = RRTTree(goal_config)

    tree_a = tree_start
    tree_b = tree_goal
    tree_a_is_start = True

    for _ in range(max_iterations):
        if rng.random() < goal_bias:
            sample = goal_config.copy()
        else:
            sample = _sample_uniform(joint_limits, rng)

        status, new_index = _extend(
            tree_a,
            sample,
            step_size,
            collision_model,
            collision_step,
            cube_position=cube_position,
            cube_z_rotation=cube_z_rotation,
        )
        if status != "trapped" and new_index is not None:
            new_config = tree_a.nodes[new_index]
            status_b, new_index_b = _connect(
                tree_b,
                new_config,
                step_size,
                collision_model,
                collision_step,
                cube_position=cube_position,
                cube_z_rotation=cube_z_rotation,
            )
            if status_b == "reached" and new_index_b is not None:
                path_a = tree_a.path_to_root(new_index)
                path_b = tree_b.path_to_root(new_index_b)
                if tree_a_is_start:
                    start_path = path_a
                    goal_path = path_b
                else:
                    start_path = path_b
                    goal_path = path_a
                return start_path + list(reversed(goal_path[:-1]))

        tree_a, tree_b = tree_b, tree_a
        tree_a_is_start = not tree_a_is_start

    return None


def _path_length(path: List[np.ndarray]) -> float:
    if len(path) < 2:
        return 0.0
    stacked = np.vstack(path)
    return float(np.sum(np.linalg.norm(np.diff(stacked, axis=0), axis=1)))


def _greedy_shortcut(
    path: List[np.ndarray],
    collision_model: RobotCollisionModel,
    collision_step: float,
    cube_position: Sequence[float] | None = None,
    cube_z_rotation: float | None = None,
) -> List[np.ndarray]:
    if len(path) <= 2:
        return path

    reduced: List[np.ndarray] = [path[0]]
    i = 0
    while i < len(path) - 1:
        j = len(path) - 1
        while j > i + 1:
            if _edge_is_valid(
                collision_model,
                path[i],
                path[j],
                collision_step,
                cube_position=cube_position,
                cube_z_rotation=cube_z_rotation,
            ):
                break
            j -= 1
        reduced.append(path[j])
        i = j
    return reduced


def smooth_path(
    path: Sequence[Sequence[float]],
    collision_model: RobotCollisionModel,
    collision_step: float = 2.0,
    max_iterations: int = 200,
    cube_position: Sequence[float] | None = None,
    cube_z_rotation: float | None = None,
    rng: Optional[np.random.Generator] = None,
) -> List[np.ndarray]:
    """
    Shortcut the path to reduce waypoints and smooth it.
    """
    if path is None:
        return []
    if len(path) == 0:
        return []

    rng = rng or np.random.default_rng()
    working = [np.asarray(p, dtype=float).copy() for p in path]
    working = _greedy_shortcut(
        working,
        collision_model,
        collision_step,
        cube_position=cube_position,
        cube_z_rotation=cube_z_rotation,
    )

    for _ in range(max_iterations):
        if len(working) <= 2:
            break
        i, j = sorted(rng.integers(0, len(working), size=2))
        if j - i <= 1:
            continue
        if not _edge_is_valid(
            collision_model,
            working[i],
            working[j],
            collision_step,
            cube_position=cube_position,
            cube_z_rotation=cube_z_rotation,
        ):
            continue
        candidate = working[: i + 1] + working[j:]
        if len(candidate) < len(working) or _path_length(candidate) + 1e-6 < _path_length(working):
            working = candidate

    return _greedy_shortcut(
        working,
        collision_model,
        collision_step,
        cube_position=cube_position,
        cube_z_rotation=cube_z_rotation,
    )


def plan_and_smooth(
    start: Sequence[float],
    goal: Sequence[float],
    collision_model: RobotCollisionModel,
    joint_limits: Optional[np.ndarray] = None,
    step_size: float = 7.5,
    collision_step: float = 2.0,
    max_iterations: int = 3000,
    goal_bias: float = 0.1,
    smooth_iterations: int = 200,
    cube_position: Sequence[float] | None = None,
    cube_z_rotation: float | None = None,
    rng: Optional[np.random.Generator] = None,
) -> Optional[List[np.ndarray]]:
    """
    Convenience wrapper: plan with RRT-Connect and smooth the result.
    """
    planned = rrt_connect(
        start=start,
        goal=goal,
        collision_model=collision_model,
        joint_limits=joint_limits,
        step_size=step_size,
        collision_step=collision_step,
        max_iterations=max_iterations,
        goal_bias=goal_bias,
        cube_position=cube_position,
        cube_z_rotation=cube_z_rotation,
        rng=rng,
    )
    if planned is None:
        print("RRT-Connect failed to find a path.")
        return None
    return smooth_path(
        planned,
        collision_model=collision_model,
        collision_step=collision_step,
        max_iterations=smooth_iterations,
        cube_position=cube_position,
        cube_z_rotation=cube_z_rotation,
        rng=rng,
    )


def path_to_dicts(path: Iterable[Sequence[float]]) -> List[dict]:
    """
    Convert a joint path into dictionaries that match the API client format.
    """
    output: List[dict] = []
    for pose in path:
        pose_array = _coerce_config(pose, "pose")
        output.append({str(i): float(pose_array[i]) for i in range(6)})
    return output


if __name__ == "__main__":
    # Example usage (adjust start/goal to match your robot's current pose).
    from modules.kinematics.forward_kinematics import RobotFKModel

    fk_model = RobotFKModel()
    collision_model = RobotCollisionModel(fk_model)

    start_angles = np.array([0, 0, 0, 0, 0, 0], dtype=float)
    goal_angles = np.array([30, -20, -90, 10, 45, 30], dtype=float)

    path = plan_and_smooth(
        start_angles,
        goal_angles,
        collision_model=collision_model,
    )

    if path is None:
        print("No collision-free path found.")
    else:
        print(f"Planned path with {len(path)} waypoints.")
        print(path)
