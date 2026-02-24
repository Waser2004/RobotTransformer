import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import sys

# Ensure imports work from repository root without additional packaging setup.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from expert_data_generation.optimal_generator import (  # noqa: E402
    GeneratorConfig,
    HOME_ACTUATOR_DEG,
    IKAdapter,
    JSONLWriter,
    OptimalEpisodeFailure,
    OptimalExpertGenerator,
    PHYSICAL_MAX_JOINT_VEL_DEG_S,
)


class FakeIK:
    """Predictable IK solver used for deterministic tests."""

    def solve_pregrasp_actuator_degrees(self, cube_location_m, cube_rotation_rad):
        # Return a valid actuator target near home.
        return [-85.0, 0.0, -50.0, 0.0, 140.0, 0.0]


class FakeEnv:
    """In-memory environment double with minimal RobotEnv behavior."""

    def __init__(self):
        self.actuator_deg = list(HOME_ACTUATOR_DEG)
        self.gripper_closed = False
        self.step_count = 0
        self.visible_after_step = 1
        self.force_collision = False
        self._target_location = [-0.20, -0.30, 0.025]
        self._target_rotation = [0.0, 0.0, 0.5]

    def reset(self, cube_position="home", robot_pose="home", actuator_rotations=None):
        self.actuator_deg = list(HOME_ACTUATOR_DEG)
        self.gripper_closed = False
        self.step_count = 0

    def get_state(self, image=True):
        return {
            "actuator_rotations": [v * 3.141592653589793 / 180.0 for v in self.actuator_deg],
            "actuator_velocities": [0.0] * 6,
            "target_cube_location": list(self._target_location),
            "target_cube_rotation": list(self._target_rotation),
            "graper": self.gripper_closed,
            "collisions": self.force_collision,
            "workplate_coverage": [],
            "distance_to_target": 0.2,
            "relative_rotation": [0.0, 0.0, 0.0],
            "image": [[0.0]] if image else None,
        }

    def step(self, actuator_velocities, grapper_state):
        signs = [-1.0, 1.0, -1.0, -1.0, 1.0, -1.0]
        fps = 20.0
        for i, vel in enumerate(actuator_velocities):
            self.actuator_deg[i] += signs[i] * float(vel) / fps
        self.gripper_closed = bool(grapper_state)
        self.step_count += 1
        return 0.0

    def target_cube_in_view(self, padding=0.0):
        return self.step_count >= self.visible_after_step


class TestOptimalGenerator(unittest.TestCase):
    def _records_from_file(self, path):
        with open(path, "r", encoding="utf-8") as fp:
            return [json.loads(line) for line in fp if line.strip()]

    def test_trigger_episode_logs_images_and_final(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            output = str(Path(tmp_dir) / "dataset.jsonl")
            config = GeneratorConfig(num_trigger_episodes=0, num_task_episodes=0, output_jsonl=output)
            env = FakeEnv()
            with JSONLWriter(output) as writer:
                with patch("expert_data_generation.optimal_generator.IKAdapter", return_value=FakeIK()):
                    generator = OptimalExpertGenerator(env=env, config=config, writer=writer)
                    generator.run_trigger_episode(0)

            records = self._records_from_file(output)
            step_records = [r for r in records if r["record_type"] == "step"]
            self.assertGreater(len(step_records), 0)
            self.assertIn("image", step_records[0])
            final = [r for r in records if r["record_type"] == "episode_final"][0]
            self.assertTrue(final["success"])

    def test_task_episode_search_not_found_fails(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            output = str(Path(tmp_dir) / "dataset.jsonl")
            config = GeneratorConfig(num_trigger_episodes=0, num_task_episodes=0, output_jsonl=output)
            env = FakeEnv()
            env.visible_after_step = 10_000
            with JSONLWriter(output) as writer:
                with patch("expert_data_generation.optimal_generator.IKAdapter", return_value=FakeIK()):
                    generator = OptimalExpertGenerator(env=env, config=config, writer=writer)
                    generator.run_task_episode(0, checkpoints=[[-90.0, 0.0, -47.0, 0.0, 137.0, 0.0]])

            records = self._records_from_file(output)
            final = [r for r in records if r["record_type"] == "episode_final"][0]
            self.assertEqual(final["outcome_label"], "failure_unrecoverable")
            self.assertEqual(final["termination_reason"], "search_target_not_found")

    def test_move_to_target_respects_velocity_cap(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            output = str(Path(tmp_dir) / "dataset.jsonl")
            config = GeneratorConfig(
                num_trigger_episodes=0,
                num_task_episodes=0,
                output_jsonl=output,
                max_joint_vel_deg_s=5.0,
                interp_dt=0.05,
                max_steps_per_motion=10,
            )
            env = FakeEnv()
            with JSONLWriter(output) as writer:
                with patch("expert_data_generation.optimal_generator.IKAdapter", return_value=FakeIK()):
                    generator = OptimalExpertGenerator(env=env, config=config, writer=writer)
                    with self.assertRaises(OptimalEpisodeFailure):
                        generator._move_to_target(
                            episode_id="ep",
                            start_t=0,
                            phase="search",
                            target_actuator_deg=[-40.0, 30.0, -120.0, 0.0, 160.0, 20.0],
                            gripper_closed=False,
                            require_visibility=False,
                        )

            records = self._records_from_file(output)
            step_records = [r for r in records if r["record_type"] == "step"]
            self.assertGreater(len(step_records), 0)
            for record in step_records:
                vel_cmd = record["action"]["actuator_velocities"]
                expected_caps = [min(5.0, p) for p in PHYSICAL_MAX_JOINT_VEL_DEG_S]
                self.assertTrue(all(abs(v) <= cap for v, cap in zip(vel_cmd, expected_caps)))

    def test_move_to_target_respects_physical_velocity_caps(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            output = str(Path(tmp_dir) / "dataset.jsonl")
            config = GeneratorConfig(
                num_trigger_episodes=0,
                num_task_episodes=0,
                output_jsonl=output,
                max_joint_vel_deg_s=100.0,
                interp_dt=0.05,
                max_steps_per_motion=10,
            )
            env = FakeEnv()
            with JSONLWriter(output) as writer:
                with patch("expert_data_generation.optimal_generator.IKAdapter", return_value=FakeIK()):
                    generator = OptimalExpertGenerator(env=env, config=config, writer=writer)
                    with self.assertRaises(OptimalEpisodeFailure):
                        generator._move_to_target(
                            episode_id="ep",
                            start_t=0,
                            phase="search",
                            target_actuator_deg=[-40.0, 30.0, -120.0, 45.0, 160.0, 40.0],
                            gripper_closed=False,
                            require_visibility=False,
                        )

            records = self._records_from_file(output)
            step_records = [r for r in records if r["record_type"] == "step"]
            self.assertGreater(len(step_records), 0)
            for record in step_records:
                vel_cmd = record["action"]["actuator_velocities"]
                self.assertTrue(
                    all(abs(v) <= cap for v, cap in zip(vel_cmd, PHYSICAL_MAX_JOINT_VEL_DEG_S))
                )

    def test_collision_fail_fast(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            output = str(Path(tmp_dir) / "dataset.jsonl")
            config = GeneratorConfig(num_trigger_episodes=0, num_task_episodes=0, output_jsonl=output)
            env = FakeEnv()
            env.force_collision = True
            with JSONLWriter(output) as writer:
                with patch("expert_data_generation.optimal_generator.IKAdapter", return_value=FakeIK()):
                    generator = OptimalExpertGenerator(env=env, config=config, writer=writer)
                    generator.run_task_episode(0, checkpoints=[[-90.0, 0.0, -47.0, 0.0, 137.0, 0.0]])

            records = self._records_from_file(output)
            final = [r for r in records if r["record_type"] == "episode_final"][0]
            self.assertEqual(final["termination_reason"], "collision_abort")

    def test_ik_adapter_unit_conversion_and_mapping(self):
        class DummySolver:
            def __init__(self):
                self.location = None
                self.rotation = None

            def set_end_effector(self, location, rotation):
                self.location = location
                self.rotation = rotation

            def calc_inverse_kinematics(self):
                return [100.0, 80.0, 140.0, 60.0, 170.0, 95.0]

        dummy_solver = DummySolver()

        with patch.object(IKAdapter, "_load_inverse_kinematics_class", return_value=lambda: dummy_solver):
            adapter = IKAdapter()
            result = adapter.solve_pregrasp_actuator_degrees(
                cube_location_m=[0.2, -0.3, 0.025],
                cube_rotation_rad=[0.0, 0.0, 0.5],
            )

        # Verify meter->millimeter conversion and prescribed pre-grasp z.
        self.assertEqual(dummy_solver.location[0], 200.0)
        self.assertEqual(dummy_solver.location[1], -300.0)
        self.assertEqual(dummy_solver.location[2], 25.0)

        # Verify IK->actuator mapping.
        self.assertEqual(result, [-10.0, 10.0, 40.0, 30.0, 10.0, -5.0])

    def test_task_success_logs_grasp_and_return_release(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            output = str(Path(tmp_dir) / "dataset.jsonl")
            config = GeneratorConfig(num_trigger_episodes=0, num_task_episodes=0, output_jsonl=output)
            env = FakeEnv()
            env.visible_after_step = 0
            with JSONLWriter(output) as writer:
                with patch("expert_data_generation.optimal_generator.IKAdapter", return_value=FakeIK()):
                    generator = OptimalExpertGenerator(env=env, config=config, writer=writer)
                    generator.run_task_episode(0, checkpoints=[[-90.0, 0.0, -47.0, 0.0, 137.0, 0.0]])

            records = self._records_from_file(output)
            final = [r for r in records if r["record_type"] == "episode_final"][0]
            self.assertTrue(final["success"])

            grasp_steps = [r for r in records if r.get("phase") == "grasp"]
            return_steps = [r for r in records if r.get("phase") == "return"]
            self.assertGreater(len(grasp_steps), 0)
            self.assertGreater(len(return_steps), 0)

            # Ensure we closed in grasp and eventually opened in return release.
            self.assertTrue(all(r["action"]["grapper_state"] for r in grasp_steps))
            self.assertTrue(any(not r["action"]["grapper_state"] for r in return_steps))


if __name__ == "__main__":
    unittest.main()
