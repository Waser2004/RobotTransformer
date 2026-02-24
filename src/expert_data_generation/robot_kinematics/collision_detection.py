from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import fcl
import trimesh
import numpy as np

from robot_kinematics.forward_kinematics import DHForwardKinematics


@dataclass(frozen=True)
class CollisionHit:
    name_a: str
    name_b: str
    contacts: object | None = None


class RobotCollisionModel:
    """
    Build collision geometry from the robot STL files and run python-fcl collision checks.
    """

    def __init__(
        self,
        fk_model: DHForwardKinematics,
        robot_files_dir: Path | str = Path(__file__).parent / "robot_files",
        mesh_map: dict[str, int | None] | None = None,
        env_meshes: Iterable[str] | None = None,
        exclude_pairs: Iterable[tuple[str, str]] | None = None,
    ):
        self.fk_model = fk_model
        self.robot_files_dir = Path(robot_files_dir)

        if mesh_map is None:
            mesh_map = {
                "Robot Base.stl": -1,
                "primary Arm.stl": 0,
                "secondary Arm part 1.stl": 1,
                "secondary Arm part 2.stl": 2,
                "tertiary arm part 1.stl": 3,
                "tertiary Arm part 2.stl": 4,
                "Finger (left).stl": 4,
                "Finger (right).stl": 4,
            }
        self.mesh_map = dict(mesh_map)

        if env_meshes is None:
            env_meshes = [
                "Kallax_floor.stl",
                "Kallax_wall.stl",
                "Target Cube.stl",
            ]
        self.env_meshes = list(env_meshes)

        # Normalize exclusions to unordered name pairs.
        exclude_pairs = exclude_pairs or [
            # non-colliding robot env pairs by default
            ("Robot Base.stl", "Kallax_floor.stl"),
            ("Robot Base.stl", "Kallax_wall.stl"),
            ("primary Arm.stl", "Kallax_floor.stl"),
            ("primary Arm.stl", "Kallax_wall.stl"),
            ("secondary Arm part 1.stl", "Kallax_wall.stl"),
            ("secondary Arm part 2.stl", "Kallax_wall.stl"),
            # non-colliding robot self pairs by default
            ("Robot Base.stl", "primary Arm.stl"),
            ("Robot Base.stl", "secondary Arm part 1.stl"),
            ("primary Arm.stl", "secondary Arm part 1.stl"),
            ("secondary Arm part 1.stl", "secondary Arm part 2.stl"),
            ("secondary Arm part 1.stl", "tertiary arm part 1.stl"),
            ("secondary Arm part 1.stl", "tertiary arm part 2.stl"),
            ("secondary Arm part 1.stl", "Finger (left).stl"),
            ("secondary Arm part 1.stl", "Finger (right).stl"),
            ("secondary Arm part 2.stl", "tertiary arm part 1.stl"),
            ("secondary Arm part 2.stl", "tertiary arm part 2.stl"),
            ("secondary Arm part 2.stl", "Finger (left).stl"),
            ("secondary Arm part 2.stl", "Finger (right).stl"),
            ("tertiary arm part 1.stl", "tertiary Arm part 2.stl"),
            ("tertiary arm part 1.stl", "Finger (left).stl"),
            ("tertiary arm part 1.stl", "Finger (right).stl"),
            ("tertiary Arm part 2.stl", "Finger (left).stl"),
            ("tertiary Arm part 2.stl", "Finger (right).stl"),
        ]
        self.exclude_pairs = {
            self._normalize_pair(a, b) for (a, b) in (exclude_pairs)
        }

        self._robot_models = {}
        self._env_models = {}
        self._load_models()

    @staticmethod
    def _load_trimesh(mesh_path: Path):
        # Keep export/collision meshes consistent by using trimesh for STL loading.
        mesh = trimesh.load(mesh_path, force="mesh")
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
        return mesh

    def _mesh_to_fcl_model(self, mesh):
        # Build an FCL BVH from raw triangles/vertices for fast collision queries.
        vertices = np.asarray(mesh.vertices, dtype=float)
        faces = np.asarray(mesh.faces, dtype=int)
        model = fcl.BVHModel()
        model.beginModel(len(vertices), len(faces))
        model.addSubModel(vertices, faces)
        model.endModel()
        return model

    def _load_models(self):
        # Load robot link meshes and environment meshes once (geometry is static).
        for mesh_name in self.mesh_map.keys():
            mesh_path = self.robot_files_dir / mesh_name
            if not mesh_path.exists():
                print(f"Missing mesh: {mesh_path}")
                continue
            mesh = self._load_trimesh(mesh_path)
            self._robot_models[mesh_name] = self._mesh_to_fcl_model(mesh)

        # Load environment meshes
        for mesh_name in self.env_meshes:
            mesh_path = self.robot_files_dir / mesh_name
            if not mesh_path.exists():
                print(f"Missing env mesh: {mesh_path}")
                continue
            mesh = self._load_trimesh(mesh_path)
            self._env_models[mesh_name] = self._mesh_to_fcl_model(mesh)

    @staticmethod
    def _rotation_matrix_to_quaternion(r: np.ndarray):
        trace = np.trace(r)
        if trace > 0:
            s = np.sqrt(trace + 1.0) * 2.0
            w = 0.25 * s
            x = (r[2, 1] - r[1, 2]) / s
            y = (r[0, 2] - r[2, 0]) / s
            z = (r[1, 0] - r[0, 1]) / s
        else:
            idx = int(np.argmax([r[0, 0], r[1, 1], r[2, 2]]))
            if idx == 0:
                s = np.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2.0
                w = (r[2, 1] - r[1, 2]) / s
                x = 0.25 * s
                y = (r[0, 1] + r[1, 0]) / s
                z = (r[0, 2] + r[2, 0]) / s
            elif idx == 1:
                s = np.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2.0
                w = (r[0, 2] - r[2, 0]) / s
                x = (r[0, 1] + r[1, 0]) / s
                y = 0.25 * s
                z = (r[1, 2] + r[2, 1]) / s
            else:
                s = np.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2.0
                w = (r[1, 0] - r[0, 1]) / s
                x = (r[0, 2] + r[2, 0]) / s
                y = (r[1, 2] + r[2, 1]) / s
                z = 0.25 * s
        return w, x, y, z

    @staticmethod
    def _normalize_pair(a: str, b: str) -> tuple[str, str]:
        return tuple(sorted((a, b)))

    def _is_excluded(self, a: str, b: str) -> bool:
        return self._normalize_pair(a, b) in self.exclude_pairs

    def _matrix_to_fcl_transform(self, mat: np.ndarray):
        # python-fcl accepts either (R, t) or (quat, t); fall back to quaternion if needed.
        r = np.asarray(mat[:3, :3], dtype=float)
        t = np.asarray(mat[:3, 3], dtype=float)
        try:
            return fcl.Transform(r, t)
        except Exception:
            w, x, y, z = self._rotation_matrix_to_quaternion(r)
            if hasattr(fcl, "Quaternion"):
                quat = fcl.Quaternion(w, x, y, z)
            else:
                quat = [w, x, y, z]
            return fcl.Transform(quat, t)

    @staticmethod
    def _z_rotation_matrix(angle_deg: float) -> np.ndarray:
        # Additional Z-rotation is applied to match the export pipeline exactly.
        angle = np.radians(angle_deg)
        return np.array([
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle),  np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ])

    @staticmethod
    def _get_joint_angle(joint_angles: list[float] | None, joint_index: int) -> float:
        # Matches fk_inference.py: use the next joint (index + 1) if available.
        if not joint_angles:
            return 0.0
        idx = joint_index + 1
        if 0 <= idx < len(joint_angles):
            return joint_angles[idx]
        return 0.0

    def _get_joint_angles_from_fk(self) -> list[float]:
        # fk_model.joints are stored in radians; convert to degrees to match export API.
        return [float(np.degrees(angle)) for angle in getattr(self.fk_model, "joints", [])]

    def _build_transform_matrix(self, joint_index: int, joint_angles: list[float] | None) -> np.ndarray:
        # Replicates export_fk_meshes_to_obj transform logic (FK rotation/translation + Z tweak).
        z_angle = self._get_joint_angle(joint_angles, joint_index)
        rotation_z = self._z_rotation_matrix(z_angle)

        if joint_index >= 0:
            rotation = self.fk_model.get_joint_rotation_matrix(joint_index) @ rotation_z
            translation = self.fk_model.get_joint_translation_vector(joint_index)
        else:
            rotation = rotation_z
            translation = np.array([[0.0], [0.0], [0.0]])

        transform = np.eye(4)
        transform[:3, :3] = rotation
        transform[:3, 3] = np.asarray(translation).flatten()
        return transform

    def _build_robot_objects(self, joint_angles: list[float] | None = None):
        # Rebuild transforms each call because joint angles can change per query.
        if joint_angles is None:
            joint_angles = self._get_joint_angles_from_fk()

        objects = {}
        for mesh_name, joint_index in self.mesh_map.items():
            model = self._robot_models.get(mesh_name)
            if model is None:
                continue
            if joint_index is None:
                transform = fcl.Transform()
            else:
                mat = self._build_transform_matrix(joint_index, joint_angles)
                transform = self._matrix_to_fcl_transform(mat)
            objects[mesh_name] = fcl.CollisionObject(model, transform)
        return objects

    def _build_env_objects(self):
        # Environment objects are static, so they always use identity transforms.
        objects = {}
        identity = fcl.Transform()
        for mesh_name, model in self._env_models.items():
            objects[mesh_name] = fcl.CollisionObject(model, identity)
        return objects

    def export_debug_meshes(
        self,
        joint_angles: list[float] | None = None,
        output_dir: Path | str = Path(__file__).parent / "debug_collision_meshes",
        cube_position: list[float] | None = None,
        cube_z_rotation: float | None = None,
    ):
        """
        Export transformed robot meshes as OBJ files using the same transform logic as collisions.
        """
        # This mirrors export_fk_meshes_to_obj but keeps collision and visual debug in sync.
        if joint_angles is None:
            joint_angles = self._get_joint_angles_from_fk()
        self.fk_model.set_joint_angles(*joint_angles)

        # Adjust cube position to match collision reference frame.
        if cube_position is not None:
            cube_position = [-cube_position[1], -cube_position[0], cube_position[2]]

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for mesh_name, joint_index in self.mesh_map.items():
            mesh_path = self.robot_files_dir / mesh_name
            if not mesh_path.exists():
                print(f"Missing mesh: {mesh_path}")
                continue

            mesh = self._load_trimesh(mesh_path)
            if joint_index is not None:
                transform = self._build_transform_matrix(joint_index, joint_angles)
                mesh.apply_transform(transform)

            out_path = output_dir / f"{mesh_path.stem}.obj"
            mesh.export(out_path)

        # Export environment meshes as well (static, identity transform).
        for mesh_name in self.env_meshes:
            if mesh_name == "Target Cube.stl" and cube_position is None:
                continue
            mesh_path = self.robot_files_dir / mesh_name
            if not mesh_path.exists():
                print(f"Missing env mesh: {mesh_path}")
                continue
            mesh = self._load_trimesh(mesh_path)
            if mesh_name == "Target Cube.stl":
                z_rotation = cube_z_rotation or 0.0
                transform = np.eye(4)
                transform[:3, :3] = self._z_rotation_matrix(z_rotation)
                transform[:3, 3] = np.asarray(cube_position, dtype=float)
                mesh.apply_transform(transform)
            out_path = output_dir / f"{mesh_path.stem}.obj"
            mesh.export(out_path)

    def check_collisions(
        self,
        include_self: bool = True,
        return_contacts: bool = False,
        joint_angles: list[float] | None = None,
        cube_position: list[float] | None = None,
        cube_z_rotation: float | None = None,
    ) -> list[CollisionHit]:
        """
        Check robot collisions against environment (and optionally self-collisions).
        """
        # Build fresh collision objects per call to reflect current joint state.
        robot_objects = self._build_robot_objects(joint_angles)
        env_objects = self._build_env_objects()

        # exclude cube if position not provided
        if cube_position is None:
            env_objects.pop("Target Cube.stl", None)
        # if position provided, add transformed cube
        else:
            z_rotation = cube_z_rotation or 0.0
            cube_model = self._env_models.get("Target Cube.stl")
            if cube_model is not None:
                rotation = self._z_rotation_matrix(z_rotation)
                translation = np.asarray(cube_position, dtype=float)
                try:
                    transform = fcl.Transform(rotation, translation)
                except Exception:
                    w, x, y, z = self._rotation_matrix_to_quaternion(rotation)
                    if hasattr(fcl, "Quaternion"):
                        quat = fcl.Quaternion(w, x, y, z)
                    else:
                        quat = [w, x, y, z]
                    transform = fcl.Transform(quat, translation)
                env_objects["Target Cube.stl"] = fcl.CollisionObject(cube_model, transform)

        hits: list[CollisionHit] = []
        request = fcl.CollisionRequest(num_max_contacts=1, enable_contact=return_contacts)

        for r_name, r_obj in robot_objects.items():
            for e_name, e_obj in env_objects.items():
                if self._is_excluded(r_name, e_name):
                    continue
                result = fcl.CollisionResult()
                if fcl.collide(r_obj, e_obj, request, result) > 0:
                    contacts = getattr(result, "contacts", None) if return_contacts else None
                    hits.append(CollisionHit(r_name, e_name, contacts))
        
        if include_self:
            names = list(robot_objects.keys())
            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    a_name = names[i]
                    b_name = names[j]
                    if self._is_excluded(a_name, b_name):
                        continue
                    result = fcl.CollisionResult()
                    if fcl.collide(robot_objects[a_name], robot_objects[b_name], request, result) > 0:
                        contacts = getattr(result, "contacts", None) if return_contacts else None
                        hits.append(CollisionHit(a_name, b_name, contacts))

        return hits

    # RRT-Connect
    def is_state_valid(
        self, 
        joint_angles: list[float], 
        cube_position: list[float] | None = None, 
        cube_z_rotation: float | None = None
    ) -> bool:
        self.fk_model.set_joint_angles(*joint_angles)
        hits = self.check_collisions(
            include_self=True, 
            joint_angles=joint_angles,
            cube_position=[-cube_position[1], -cube_position[0], cube_position[2]] if cube_position else None,
            cube_z_rotation=cube_z_rotation
        )
        return len(hits) == 0
    
    def is_edge_valid(self,
            start_angles: list[float],
            end_angles: list[float],
            cube_position: list[float] | None = None,
            cube_z_rotation: float | None = None,
            step_size: float = 1.0
        ) -> bool:
        # Linear interpolation between start and end joint angles, checking collisions along the way.
        start = np.array(start_angles)
        end   = np.array(end_angles)
        diff  = end - start
        max_diff = np.max(np.abs(diff))
        
        # checking if each step is valid
        for step in range(0, int(max_diff / step_size) + 1):
            current_angles = start + (diff * (step * step_size / max_diff))
            if not self.is_state_valid(current_angles.tolist(), cube_position, cube_z_rotation):
                return False
        
        # no invalid states found along the edge
        return True
