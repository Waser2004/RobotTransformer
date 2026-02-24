import importlib
import json
import math
import sys
import tempfile
import types
import unittest
from pathlib import Path


def _install_data_generator_import_stubs():
    """Install minimal stubs so data_generator can be imported without heavy runtime deps."""
    if "numpy" not in sys.modules:
        numpy_stub = types.ModuleType("numpy")
        numpy_stub.ndarray = object
        numpy_stub.float32 = float
        numpy_stub.matrix = object
        numpy_stub.asarray = lambda x, dtype=None: x
        numpy_stub.max = lambda arr: 0.0
        numpy_stub.clip = lambda arr, a, b: arr
        numpy_stub.pad = lambda arr, pads, mode=None: arr
        numpy_stub.repeat = lambda arr, repeats, axis=None: arr
        sys.modules["numpy"] = numpy_stub

    if "scipy" not in sys.modules:
        scipy_mod = types.ModuleType("scipy")
        spatial_mod = types.ModuleType("scipy.spatial")
        transform_mod = types.ModuleType("scipy.spatial.transform")

        class _DummyRotation:
            @staticmethod
            def from_matrix(*args, **kwargs):
                return _DummyRotation()

            @staticmethod
            def from_euler(*args, **kwargs):
                return _DummyRotation()

            def as_euler(self, *args, **kwargs):
                return [0.0, 0.0, 0.0]

            def __mul__(self, other):
                return self

        class _DummySlerp:
            def __init__(self, *args, **kwargs):
                pass

            def __call__(self, values):
                return [_DummyRotation() for _ in values]

        transform_mod.Rotation = _DummyRotation
        transform_mod.Slerp = _DummySlerp
        spatial_mod.transform = transform_mod
        scipy_mod.spatial = spatial_mod
        sys.modules["scipy"] = scipy_mod
        sys.modules["scipy.spatial"] = spatial_mod
        sys.modules["scipy.spatial.transform"] = transform_mod

    if "env_client" not in sys.modules:
        env_client_mod = types.ModuleType("env_client")

        class EnvClient:  # pragma: no cover - type placeholder only
            pass

        env_client_mod.EnvClient = EnvClient
        sys.modules["env_client"] = env_client_mod

    if "robot_kinematics" not in sys.modules:
        rk_pkg = types.ModuleType("robot_kinematics")
        rk_pkg.__path__ = []
        sys.modules["robot_kinematics"] = rk_pkg

    inverse_mod = types.ModuleType("robot_kinematics.inverse_kinematics")
    forward_mod = types.ModuleType("robot_kinematics.forward_kinematics")
    collision_mod = types.ModuleType("robot_kinematics.collision_detection")
    avoidance_mod = types.ModuleType("robot_kinematics.collision_avoidance")

    class _DummyIK:
        def set_end_effector(self, *args, **kwargs):
            pass

        def calc_inverse_kinematics(self):
            return [0.0] * 6

    class _DummyFK:
        def set_joint_angles(self, *args, **kwargs):
            pass

        def get_joint_translation_vector(self, *args, **kwargs):
            return [0.0, 0.0, 0.0]

    class _DummyCollision:
        def __init__(self, *args, **kwargs):
            pass

        def is_edge_valid(self, *args, **kwargs):
            return True

    inverse_mod.InverseKinematics = _DummyIK
    forward_mod.RobotFKModel = _DummyFK
    collision_mod.RobotCollisionModel = _DummyCollision
    avoidance_mod.plan_and_smooth = lambda *args, **kwargs: None

    sys.modules["robot_kinematics.inverse_kinematics"] = inverse_mod
    sys.modules["robot_kinematics.forward_kinematics"] = forward_mod
    sys.modules["robot_kinematics.collision_detection"] = collision_mod
    sys.modules["robot_kinematics.collision_avoidance"] = avoidance_mod


def _load_data_generator_module():
    """Import the target module from src/expert_data_generation with local stubs installed."""
    _install_data_generator_import_stubs()
    repo_root = Path(__file__).resolve().parents[1]
    module_dir = str(repo_root / "src" / "expert_data_generation")
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)
    return importlib.import_module("data_generator")


class FakeEnv:
    """Small in-memory env stub for generator control-flow tests."""

    def __init__(self, home_deg):
        self.home_deg = list(home_deg)
        self.actuator_deg = [v + 10.0 for v in home_deg]
        self.reset_calls = []
        self.visible = False
        self.cube_gone_calls = 0
        self.cube_move_calls = 0

    def reset(self, cube_position="home", robot_pose="home", actuator_rotations=None):
        # Start away from home so _return_home has real movement to log when invoked.
        self.reset_calls.append(
            {
                "cube_position": cube_position,
                "robot_pose": robot_pose,
                "actuator_rotations": actuator_rotations,
            }
        )
        self.actuator_deg = [v + 10.0 for v in self.home_deg]

    def get_state(self, image=False):
        return {
            "actuator_rotations": [math.radians(v) for v in self.actuator_deg],
            "actuator_velocities": [0.0] * 6,
            "target_cube_location": [0.0, 0.0, 0.025],
            "target_cube_rotation": [0.0, 0.0, 0.0],
            "distance_to_target": 1.0,
            "image": None,
        }

    def step(self, actuator_velocities, grapper_state):
        # Mirror EnvControl signs so _move_to_target converges with the existing controller logic.
        signs = [-1.0, 1.0, -1.0, -1.0, 1.0, -1.0]
        fps = 20.0
        for idx, vel in enumerate(actuator_velocities):
            self.actuator_deg[idx] += signs[idx] * float(vel) / fps
        return 0.0

    def target_cube_in_view(self, padding=0.0):
        return bool(self.visible)

    def set_cube_gone(self):
        self.cube_gone_calls += 1

    def move_cube_random_on_workplate(self):
        self.cube_move_calls += 1


class TestDataGeneratorNonOptimal(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dg = _load_data_generator_module()

    def _read_records(self, path):
        with open(path, "r", encoding="utf-8") as fp:
            return [json.loads(line) for line in fp if line.strip()]

    def _make_generator(
        self,
        output_path,
        invisible_probability=0.01,
        grab_disappear_probability=0.10,
        grab_move_probability=0.10,
    ):
        config = self.dg.Config(
            invisible_cube_probability=invisible_probability,
            grab_cube_disappears_probability=grab_disappear_probability,
            grab_cube_moves_probability=grab_move_probability,
        )
        env = FakeEnv(self.dg.HOME_ACTUATOR_DEG)
        writer = self.dg.JSONLWriter(output_path)
        generator = self.dg.OptimalExpertGenerator(env=env, config=config, writer=writer, video_writer=None)
        return generator, env, writer

    def test_search_fail_episode_is_committed_and_marked_nonoptimal(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            output = str(Path(tmp_dir) / "dataset.jsonl")
            generator, env, writer = self._make_generator(output, invisible_probability=1.0)

            # Force the search path to fail and confirm the new return-home branch is used.
            generator.search = lambda visibility_padding=0.1: False
            return_home_calls = []

            def _fake_return_home(reason="episode_end"):
                return_home_calls.append(reason)
                return True

            generator._return_home = _fake_return_home
            try:
                generator.generate(num_episodes=1)
            finally:
                writer.close()

            records = self._read_records(output)
            self.assertEqual(len(records), 1)
            record = records[0]
            self.assertFalse(record["success"])
            self.assertEqual(return_home_calls, ["search_target_not_found"])
            self.assertEqual(env.reset_calls[0]["cube_position"], "invisible")
            self.assertEqual(record["meta"]["reset"]["cube_position"], "invisible")
            self.assertEqual(record["meta"]["invisible_cube_probability"], 1.0)
            self.assertFalse(record["meta"]["search_success"])
            self.assertFalse(record["meta"]["grab_success"])
            self.assertEqual(record["meta"]["termination_reason"], "search_target_not_found")
            self.assertEqual(record["meta"]["outcome_label"], "failure_unrecoverable")
            self.assertTrue(record["meta"]["return_home_attempted"])
            self.assertTrue(record["meta"]["return_home_success"])

    def test_return_home_logs_return_home_phase(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            output = str(Path(tmp_dir) / "dataset.jsonl")
            generator, env, writer = self._make_generator(output)
            try:
                writer.begin_sequence({"episode_index": 0, "reset": {"cube_position": "random_on_workplate"}})
                success = generator._return_home(reason="search_target_not_found")
                writer.commit_sequence({"search_success": False}, success=False)
            finally:
                writer.close()

            self.assertTrue(success)
            records = self._read_records(output)
            self.assertEqual(len(records), 1)
            phases = [step.get("phase") for step in records[0]["steps"]]
            self.assertIn("return_home", phases)
            self.assertIn("return_home", records[0]["phases"])

    def test_probability_zero_keeps_visible_cube_path_and_success_commit(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            output = str(Path(tmp_dir) / "dataset.jsonl")
            generator, env, writer = self._make_generator(output, invisible_probability=0.0)

            # Stub high-level phases so this test only verifies reset selection + commit behavior.
            generator.search = lambda visibility_padding=0.1: True
            generator.grab = lambda **kwargs: True
            try:
                generator.generate(num_episodes=1)
            finally:
                writer.close()

            records = self._read_records(output)
            self.assertEqual(len(records), 1)
            record = records[0]
            self.assertTrue(record["success"])
            self.assertEqual(env.reset_calls[0]["cube_position"], "random_on_workplate")
            self.assertEqual(record["meta"]["reset"]["cube_position"], "random_on_workplate")
            self.assertEqual(record["meta"]["invisible_cube_probability"], 0.0)
            self.assertTrue(record["meta"]["search_success"])
            self.assertTrue(record["meta"]["grab_success"])

    def test_grab_disappearance_falls_back_to_search_and_commits_failure(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            output = str(Path(tmp_dir) / "dataset.jsonl")
            generator, env, writer = self._make_generator(
                output,
                invisible_probability=0.0,
                grab_disappear_probability=1.0,
            )

            search_calls = []

            def _search_stub(visibility_padding=0.1):
                search_calls.append(float(visibility_padding))
                # Initial search succeeds, fallback search fails because cube is gone.
                return len(search_calls) == 1

            def _grab_stub(**kwargs):
                generator._last_grab_failure_reason = "cube_disappeared_during_grab"
                return False

            return_home_calls = []

            def _return_home_stub(reason="episode_end"):
                return_home_calls.append(reason)
                return True

            generator.search = _search_stub
            generator.grab = _grab_stub
            generator._return_home = _return_home_stub

            try:
                generator.generate(num_episodes=1)
            finally:
                writer.close()

            records = self._read_records(output)
            self.assertEqual(len(records), 1)
            record = records[0]
            self.assertFalse(record["success"])
            self.assertEqual(len(search_calls), 2)
            self.assertEqual(return_home_calls, ["cube_disappeared_during_grab"])
            self.assertEqual(env.reset_calls[0]["cube_position"], "random_on_workplate")
            self.assertTrue(record["meta"]["search_success"])
            self.assertFalse(record["meta"]["grab_success"])
            self.assertEqual(record["meta"]["termination_reason"], "cube_disappeared_during_grab")
            self.assertTrue(record["meta"]["fallback_search_attempted"])
            self.assertFalse(record["meta"]["fallback_search_success"])
            self.assertTrue(record["meta"]["cube_disappeared_during_grab"])
            self.assertEqual(record["meta"]["grab_failure_reason"], "cube_disappeared_during_grab")
            self.assertTrue(record["meta"]["return_home_attempted"])
            self.assertTrue(record["meta"]["return_home_success"])
            self.assertEqual(record["meta"]["grab_cube_disappears_probability"], 1.0)
            self.assertTrue(record["meta"]["planned_grab_cube_disappearance"])

    def test_grab_move_falls_back_to_search_and_retries_grab_successfully(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            output = str(Path(tmp_dir) / "dataset.jsonl")
            generator, env, writer = self._make_generator(
                output,
                invisible_probability=0.0,
                grab_disappear_probability=0.0,
                grab_move_probability=1.0,
            )

            search_calls = []
            grab_calls = []

            def _search_stub(visibility_padding=0.1):
                search_calls.append(float(visibility_padding))
                return True

            def _grab_stub(**kwargs):
                grab_calls.append(dict(kwargs))
                if len(grab_calls) == 1:
                    generator._last_grab_failure_reason = "cube_moved_during_grab"
                    return False
                return True

            generator.search = _search_stub
            generator.grab = _grab_stub

            try:
                generator.generate(num_episodes=1)
            finally:
                writer.close()

            records = self._read_records(output)
            self.assertEqual(len(records), 1)
            record = records[0]
            self.assertTrue(record["success"])
            self.assertEqual(len(search_calls), 2)  # initial search + fallback search
            self.assertEqual(len(grab_calls), 2)    # initial grab + retry grab
            self.assertTrue(grab_calls[0]["allow_cube_move"])
            self.assertFalse(grab_calls[1]["allow_cube_move"])
            self.assertFalse(grab_calls[1]["allow_cube_disappear"])
            self.assertTrue(record["meta"]["search_success"])
            self.assertTrue(record["meta"]["grab_success"])
            self.assertTrue(record["meta"]["cube_moved_during_grab"])
            self.assertTrue(record["meta"]["fallback_search_attempted"])
            self.assertTrue(record["meta"]["fallback_search_success"])
            self.assertTrue(record["meta"]["retry_grab_attempted"])
            self.assertTrue(record["meta"]["retry_grab_success"])
            self.assertEqual(record["meta"]["grab_cube_moves_probability"], 1.0)
            self.assertTrue(record["meta"]["planned_grab_cube_move"])


if __name__ == "__main__":
    unittest.main()
