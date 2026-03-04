import json
import importlib.util
import math
import os
import random
import sys
import types
import uuid
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
try:
    import imageio.v2 as imageio
except ImportError:  # pragma: no cover - runtime dependency may be absent in some environments.
    imageio = None
try:
    import cv2
except ImportError:  # pragma: no cover - runtime dependency may be absent in some environments.
    cv2 = None
from env_client import EnvClient, EnvRPCError
from robot_kinematics.inverse_kinematics import InverseKinematics
from robot_kinematics.forward_kinematics import RobotFKModel
from robot_kinematics.collision_detection import RobotCollisionModel
from robot_kinematics.collision_avoidance import plan_and_smooth
from scipy.spatial.transform import Rotation, Slerp

HOME_ACTUATOR_DEG   = [-90.0, 0.0, -46.90145, 0.0, 136.90145, 0.0] # Home pose from EnvControl.reset(robot_pose="home") converted to degrees.
STEP_VELOCITY_SIGN  = [-1.0, 1.0, -1.0, -1.0, 1.0, -1.0]           # EnvControl.step applies signs per joint when converting velocity to Euler updates.
MAX_JOINT_VELOCITY  = [6.696, 6.696, 6.696, 9.534, 6.696, 9.534]   # Per-joint physical actuator limits in deg/s.
ACTUATOR_BOUNDS_DEG = [
    (-120.0, 120.0),
    (-90.0, 90.0),
    (-180.0, 0.0),
    (-90.0, 90.0),
    (0.0, 180.0),
    (-90.0, 90.0),
]

@dataclass
class Config:
    host: str = "localhost"
    port: int = 5055
    invisible_cube_probability: float = 0.01      # default 1%
    grab_cube_disappears_probability: float = 0.1 # default 10%
    grab_cube_moves_probability: float = 0.1      # default 10%

class JSONLWriter:
    """Simple incremental JSONL writer."""

    def __init__(self, output_path: str):
        self.output_path = output_path
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        self._fp = open(output_path, "w", encoding="utf-8")
        # Sequence buffering state for transformer-ready one-line trajectory records.
        self._active_sequence: Optional[Dict[str, Any]] = None
        self._active_steps: List[Dict[str, Any]] = []

    def write(self, record: Dict[str, Any]) -> None:
        """Write one record per line for streaming-friendly storage."""
        self._fp.write(json.dumps(record) + "\n")
        self._fp.flush()

    def begin_sequence(self, meta: Dict[str, Any]) -> str:
        """Start buffering a new sequence record until commit/abort."""
        if self._active_sequence is not None:
            raise RuntimeError("A sequence is already active; commit or abort it before starting a new one.")

        sequence_id = str(meta.get("sequence_id") or f"episode-{meta.get('episode_index', 'unknown')}-{uuid.uuid4().hex[:8]}")
        episode_index = meta.get("episode_index")

        # Store episode-level metadata separately so per-step data stays compact and regular.
        sequence_meta = dict(meta)
        sequence_meta.pop("sequence_id", None)
        sequence_meta.pop("episode_index", None)

        self._active_sequence = {
            "record_type": "sequence",
            "schema_version": "v1",
            "sequence_id": sequence_id,
            "episode_index": episode_index,
            "phases": [],
            "success": True,
            "steps": [],
            "meta": sequence_meta,
        }
        self._active_steps = []
        return sequence_id

    def append_step(self, step: Dict[str, Any]) -> None:
        """Append a single transition to the active buffered sequence."""
        if self._active_sequence is None:
            raise RuntimeError("No active sequence. Call begin_sequence() before append_step().")

        step_record = dict(step)
        step_record.setdefault("t", len(self._active_steps))
        self._active_steps.append(step_record)

    def commit_sequence(
        self,
        final_meta: Optional[Dict[str, Any]] = None,
        success: Optional[bool] = None,
    ) -> None:
        """Finalize and write the buffered sequence as one JSONL line."""
        if self._active_sequence is None:
            raise RuntimeError("No active sequence. Call begin_sequence() before commit_sequence().")

        sequence_record = dict(self._active_sequence)
        sequence_record["steps"] = list(self._active_steps)
        if success is not None:
            # Allow callers to persist non-optimal/failed episodes without changing the schema shape.
            sequence_record["success"] = bool(success)

        # Preserve phase order of first appearance for downstream phase-aware training.
        seen_phases = set()
        phases: List[str] = []
        for step in self._active_steps:
            phase = step.get("phase")
            if isinstance(phase, str) and phase not in seen_phases:
                seen_phases.add(phase)
                phases.append(phase)
        sequence_record["phases"] = phases

        merged_meta = dict(sequence_record.get("meta", {}))
        if final_meta:
            merged_meta.update(final_meta)
        merged_meta.setdefault("num_steps", len(self._active_steps))
        merged_meta.setdefault("phases_present", phases)
        sequence_record["meta"] = merged_meta

        self._fp.write(json.dumps(sequence_record) + "\n")
        self._fp.flush()

        self._active_sequence = None
        self._active_steps = []

    def abort_sequence(self, reason: str, meta: Optional[Dict[str, Any]] = None) -> None:
        """Discard the active sequence buffer without writing a JSONL record."""
        if self._active_sequence is None:
            raise RuntimeError("No active sequence. Call begin_sequence() before abort_sequence().")

        # Intentionally discard buffered data for success-only datasets.
        _ = reason, meta
        self._active_sequence = None
        self._active_steps = []

    def close(self) -> None:
        """Close file handle safely."""
        # Fail-safe cleanup to avoid carrying partially buffered episodes across runs.
        self._active_sequence = None
        self._active_steps = []
        self._fp.close()

    def __enter__(self) -> "JSONLWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


class EpisodeVideoWriter:
    """Stores one image sequence per episode as a video file and returns lightweight references."""

    def __init__(self, output_dir: str, fps: int = 10):
        if imageio is None and cv2 is None:
            raise RuntimeError(
                "EpisodeVideoWriter requires either OpenCV (`cv2`) or imageio to be installed."
            )

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fps = fps

        self._active_episode_id: Optional[str] = None
        self._active_video_path: Optional[Path] = None
        self._writer = None
        self._backend: Optional[str] = None
        self._frame_count = 0
        self._frame_shape_hw: Optional[Tuple[int, int]] = None
        self._encoded_frame_shape_hw: Optional[Tuple[int, int]] = None

    def begin_episode(self, episode_id: str) -> None:
        """Begin a new episode video. Frames are appended with append_frame()."""
        if self._active_episode_id is not None:
            raise RuntimeError("A video episode is already active; commit or abort it before starting a new one.")

        self._active_episode_id = str(episode_id)
        self._active_video_path = self.output_dir / f"{self._active_episode_id}.mp4"
        self._writer = None
        self._backend = None
        self._frame_count = 0
        self._frame_shape_hw = None
        self._encoded_frame_shape_hw = None

    def append_frame(self, image: Any) -> int:
        """Append a grayscale frame and return its zero-based frame index."""
        if self._active_episode_id is None or self._active_video_path is None:
            raise RuntimeError("No active video episode. Call begin_episode() before append_frame().")
        if image is None:
            raise RuntimeError("Cannot append a missing image frame to the episode video.")

        frame_arr = np.asarray(image)
        frame_gray = np.asarray(frame_arr, dtype=np.float32)
        if frame_gray.ndim != 2:
            raise RuntimeError(f"Expected grayscale image with 2 dims, got shape {frame_gray.shape}.")

        frame_u8 = self._convert_grayscale_frame_to_u8(frame_arr, frame_gray)

        original_shape_hw = (int(frame_u8.shape[0]), int(frame_u8.shape[1]))
        if self._frame_shape_hw is None:
            self._frame_shape_hw = original_shape_hw
        elif self._frame_shape_hw != original_shape_hw:
            raise RuntimeError(
                f"Inconsistent frame size in episode video: expected {self._frame_shape_hw}, got {frame_u8.shape}."
            )

        # Many MP4 codecs/backends require even frame sizes; odd dimensions often produce black/corrupt frames.
        frame_u8 = self._pad_frame_to_even(frame_u8)
        encoded_shape_hw = (int(frame_u8.shape[0]), int(frame_u8.shape[1]))
        if self._encoded_frame_shape_hw is None:
            self._encoded_frame_shape_hw = encoded_shape_hw
        elif self._encoded_frame_shape_hw != encoded_shape_hw:
            raise RuntimeError(
                f"Inconsistent encoded frame size in episode video: expected {self._encoded_frame_shape_hw}, got {encoded_shape_hw}."
            )

        if self._writer is None:
            self._open_video_writer(frame_u8.shape[1], frame_u8.shape[0])

        # Encode grayscale image as RGB/BGR for broad MP4 codec compatibility.
        frame_rgb = np.repeat(frame_u8[:, :, None], 3, axis=2)

        frame_idx = self._frame_count
        if self._backend == "cv2":
            # OpenCV expects BGR channel order.
            self._writer.write(frame_rgb[:, :, ::-1])
        else:
            self._writer.append_data(frame_rgb)
        self._frame_count += 1
        return frame_idx

    def commit_episode(self) -> Dict[str, Any]:
        """Finalize the active video and return metadata to store in JSONL."""
        if self._active_episode_id is None or self._active_video_path is None:
            raise RuntimeError("No active video episode. Call begin_episode() before commit_episode().")

        self._close_active_writer()

        meta = {
            "episode_id": self._active_episode_id,
            "video_path": str(self._active_video_path.as_posix()),
            "video_format": "mp4",
            "fps": self.fps,
            "num_frames": self._frame_count,
        }
        if self._frame_shape_hw is not None:
            meta["frame_height"] = self._frame_shape_hw[0]
            meta["frame_width"] = self._frame_shape_hw[1]
        if self._encoded_frame_shape_hw is not None:
            meta["encoded_frame_height"] = self._encoded_frame_shape_hw[0]
            meta["encoded_frame_width"] = self._encoded_frame_shape_hw[1]

        self._reset_active_state()
        return meta

    def abort_episode(self) -> None:
        """Discard the active episode video and delete any partially written file."""
        if self._active_episode_id is None:
            raise RuntimeError("No active video episode. Call begin_episode() before abort_episode().")

        video_path = self._active_video_path
        self._close_active_writer()
        self._reset_active_state()

        if video_path is not None and video_path.exists():
            os.remove(video_path)

    def close(self) -> None:
        """Close resources without writing metadata. Safe for shutdown."""
        self._close_active_writer()
        self._reset_active_state()

    def _close_active_writer(self) -> None:
        if self._writer is not None:
            if self._backend == "cv2":
                self._writer.release()
            else:
                self._writer.close()
            self._writer = None
        self._backend = None

    def _reset_active_state(self) -> None:
        self._active_episode_id = None
        self._active_video_path = None
        self._backend = None
        self._frame_count = 0
        self._frame_shape_hw = None
        self._encoded_frame_shape_hw = None

    def _open_video_writer(self, width: int, height: int) -> None:
        """Open a backend-specific MP4 writer, trying OpenCV first, then imageio."""
        if self._active_video_path is None:
            raise RuntimeError("No active video path. Call begin_episode() before writing frames.")

        last_error: Optional[Exception] = None

        if cv2 is not None:
            try:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(self._active_video_path), fourcc, float(self.fps), (width, height))
                if writer is not None and writer.isOpened():
                    self._writer = writer
                    self._backend = "cv2"
                    return
                if writer is not None:
                    writer.release()
                last_error = RuntimeError("OpenCV VideoWriter failed to open MP4 output.")
            except Exception as exc:  # pragma: no cover - backend-specific runtime failure.
                last_error = exc

        if imageio is not None:
            try:
                self._writer = imageio.get_writer(str(self._active_video_path), fps=self.fps)
                self._backend = "imageio"
                return
            except Exception as exc:  # pragma: no cover - backend-specific runtime failure.
                last_error = exc

        raise RuntimeError(
            "Could not open MP4 writer backend. Install `opencv-python` or `imageio[ffmpeg]`/`imageio[pyav]`."
        ) from last_error

    @staticmethod
    def _pad_frame_to_even(frame_u8: np.ndarray) -> np.ndarray:
        """Pad grayscale frame to even H/W for MP4 codec compatibility."""
        height, width = frame_u8.shape[:2]
        pad_bottom = height % 2
        pad_right = width % 2
        if pad_bottom == 0 and pad_right == 0:
            return frame_u8
        return np.pad(frame_u8, ((0, pad_bottom), (0, pad_right)), mode="edge")

    @staticmethod
    def _convert_grayscale_frame_to_u8(frame_arr: np.ndarray, frame_gray_f32: np.ndarray) -> np.ndarray:
        """Convert env grayscale image to uint8 while handling Blender float images correctly.

        Blender/compositor outputs are often float images in ~[0, 1], but highlights may exceed 1.0.
        Treating such frames as already-8bit causes near-black videos (values 0..1 become 0/1 uint8).
        """
        frame_max = float(np.max(frame_gray_f32)) if frame_gray_f32.size else 0.0

        if np.issubdtype(frame_arr.dtype, np.floating):
            # Most env frames are normalized floats; allow slight overshoot and clip before scaling.
            if frame_max <= 4.0:
                scaled = np.clip(frame_gray_f32, 0.0, 1.0) * 255.0
                return np.clip(scaled, 0, 255).astype(np.uint8)
            return np.clip(frame_gray_f32, 0, 255).astype(np.uint8)

        return np.clip(frame_gray_f32, 0, 255).astype(np.uint8)

    def __enter__(self) -> "EpisodeVideoWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

class OptimalExpertGenerator:
    def __init__(
        self,
        env: EnvClient,
        config: Config,
        writer: JSONLWriter,
        video_writer: Optional[EpisodeVideoWriter] = None,
    ):
        self.env = env
        self.config = config
        self.writer = writer
        self.video_writer = video_writer

        # load kinematics modules
        self.fk_model = RobotFKModel()
        self.ik_solver = InverseKinematics()
        self._collision_model = RobotCollisionModel(self.fk_model)
        # Tracks the most recent grab failure mode so generate() can route non-optimal handling.
        self._last_grab_failure_reason: Optional[str] = None

    def _disable_video_capture(self, reason: str) -> None:
        """Drop optional preview-video capture so headless image RPC failures do not abort generation."""
        if self.video_writer is None:
            return
        try:
            self.video_writer.abort_episode()
        except RuntimeError:
            # The active episode may already be closed; best-effort cleanup is enough here.
            self.video_writer.close()
        self.video_writer = None
        print(f"[generator] Disabled episode video capture: {reason}")

    def _get_logging_state(
        self,
        state_without_image: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Fetch state with images and retry transient Blender render stalls before failing."""
        if self.video_writer is None:
            return state_without_image if state_without_image is not None else self.env.get_state(image=False)

        failure_reasons: List[str] = []
        for attempt in range(3):
            try:
                state_record = self.env.get_state(image=True)
            except (EnvRPCError, TimeoutError) as exc:
                # EnvControl may still be rendering or may have dropped the socket on a handler failure.
                self.env.close()
                self.env.connect()
                failure_reasons.append(f"attempt {attempt + 1}: RPC error ({exc})")
                time.sleep(1.0)
                continue

            if state_record.get("image") is not None:
                return state_record

            image_error = state_record.get("image_error") or "Blender image capture returned no frame"
            failure_reasons.append(f"attempt {attempt + 1}: {image_error}")
            time.sleep(1.0)

        raise RuntimeError(
            "Blender image capture failed after 3 attempts: " + "; ".join(failure_reasons)
        )

    def generate(self, num_episodes: int = 10):
        """Generate expert data for a number of episodes and write to JSONL file."""
        for episode in range(num_episodes):
            # Clamp to a safe range so accidental config values do not break sampling behavior.
            invisible_cube_probability       = max(0.0, min(1.0, float(self.config.invisible_cube_probability)))
            grab_cube_disappears_probability = max(0.0, min(1.0, float(self.config.grab_cube_disappears_probability)))
            grab_cube_moves_probability      = max(0.0, min(1.0, float(self.config.grab_cube_moves_probability)))
            selected_cube_position = (
                "invisible" if random.random() < invisible_cube_probability else "random_on_workplate"
            )
            # Only schedule mid-grab disappearance episodes when a cube exists to be found initially.
            should_disappear_during_grab = (
                selected_cube_position != "invisible"
                and random.random() < grab_cube_disappears_probability
            )
            # Keep move/disappear mutually exclusive so one episode has one perturbation label.
            should_move_during_grab = (
                selected_cube_position != "invisible"
                and not should_disappear_during_grab
                and random.random() < grab_cube_moves_probability
            )

            # reset env
            self.env.reset(cube_position=selected_cube_position, robot_pose="home")
            search_visibility_padding = 0.1
            grab_waypoint_step_mm = 20.0
            cube_disappear_trigger_distance_mm = 50.0

            sequence_id = self.writer.begin_sequence(
                {
                    "episode_index": episode,
                    "reset": {
                        "cube_position": selected_cube_position,
                        "robot_pose": "home",
                    },
                    "visibility_padding": search_visibility_padding,
                    "waypoint_step_mm": grab_waypoint_step_mm,
                    "invisible_cube_probability": invisible_cube_probability,
                    "grab_cube_disappears_probability": grab_cube_disappears_probability,
                    "grab_cube_moves_probability": grab_cube_moves_probability,
                    "grab_cube_disappears_trigger_distance_mm": cube_disappear_trigger_distance_mm,
                    "planned_grab_cube_disappearance": should_disappear_during_grab,
                    "planned_grab_cube_move": should_move_during_grab,
                }
            )
            if self.video_writer is not None:
                self.video_writer.begin_episode(sequence_id)

            try:
                # search and grab are stored as a single transformer training sequence.
                search_success = self.search(visibility_padding=search_visibility_padding)
                if not search_success:
                    # Search miss is a valid non-optimal episode: return home and persist the trajectory.
                    return_home_success = self._return_home(reason="search_target_not_found")
                    video_meta: Dict[str, Any] = {}
                    if self.video_writer is not None:
                        video_meta = self.video_writer.commit_episode()
                    self.writer.commit_sequence(
                        {
                            "search_success": False,
                            "grab_success": False,
                            "termination_reason": "search_target_not_found",
                            "outcome_label": "failure_unrecoverable",
                            "episode_id": sequence_id,
                            "video": video_meta if video_meta else None,
                            "return_home_attempted": True,
                            "return_home_success": return_home_success,
                        },
                        success=False,
                    )
                    continue

                grab_success = self.grab(
                    waypoint_step_mm=grab_waypoint_step_mm,
                    allow_cube_disappear=should_disappear_during_grab,
                    allow_cube_move=should_move_during_grab,
                    cube_disappear_distance_mm=cube_disappear_trigger_distance_mm,
                    cube_move_distance_mm=cube_disappear_trigger_distance_mm,
                )

                if not grab_success:
                    if self._last_grab_failure_reason == "cube_disappeared_during_grab":
                        # Resume searching from the current pose after the target disappears mid-approach.
                        fallback_search_success = self.search(visibility_padding=search_visibility_padding)
                        return_home_success = self._return_home(reason="cube_disappeared_during_grab")
                        video_meta: Dict[str, Any] = {}
                        if self.video_writer is not None:
                            video_meta = self.video_writer.commit_episode()
                        self.writer.commit_sequence(
                            {
                                "search_success": True,
                                "grab_success": False,
                                "termination_reason": "cube_disappeared_during_grab",
                                "outcome_label": "failure_unrecoverable",
                                "episode_id": sequence_id,
                                "video": video_meta if video_meta else None,
                                "fallback_search_attempted": True,
                                "fallback_search_success": fallback_search_success,
                                "cube_disappeared_during_grab": True,
                                "grab_failure_reason": self._last_grab_failure_reason,
                                "return_home_attempted": True,
                                "return_home_success": return_home_success,
                            },
                            success=False,
                        )
                        continue
                    if self._last_grab_failure_reason == "cube_moved_during_grab":
                        # Resume searching from the current pose, then retry the grab once without further perturbations.
                        fallback_search_success = self.search(visibility_padding=search_visibility_padding)
                        if not fallback_search_success:
                            return_home_success = self._return_home(reason="cube_moved_during_grab_search_failed")
                            video_meta: Dict[str, Any] = {}
                            if self.video_writer is not None:
                                video_meta = self.video_writer.commit_episode()
                            self.writer.commit_sequence(
                                {
                                    "search_success": True,
                                    "grab_success": False,
                                    "termination_reason": "cube_moved_during_grab_search_failed",
                                    "outcome_label": "failure_unrecoverable",
                                    "episode_id": sequence_id,
                                    "video": video_meta if video_meta else None,
                                    "fallback_search_attempted": True,
                                    "fallback_search_success": False,
                                    "cube_moved_during_grab": True,
                                    "grab_failure_reason": self._last_grab_failure_reason,
                                    "return_home_attempted": True,
                                    "return_home_success": return_home_success,
                                },
                                success=False,
                            )
                            continue

                        retry_grab_success = self.grab(
                            waypoint_step_mm=grab_waypoint_step_mm,
                            allow_cube_disappear=False,
                            allow_cube_move=False,
                            cube_disappear_distance_mm=cube_disappear_trigger_distance_mm,
                            cube_move_distance_mm=cube_disappear_trigger_distance_mm,
                        )
                        if not retry_grab_success:
                            return_home_success = self._return_home(reason="cube_moved_during_grab_regrab_failed")
                            video_meta: Dict[str, Any] = {}
                            if self.video_writer is not None:
                                video_meta = self.video_writer.commit_episode()
                            self.writer.commit_sequence(
                                {
                                    "search_success": True,
                                    "grab_success": False,
                                    "termination_reason": "cube_moved_during_grab_regrab_failed",
                                    "outcome_label": "failure_unrecoverable",
                                    "episode_id": sequence_id,
                                    "video": video_meta if video_meta else None,
                                    "fallback_search_attempted": True,
                                    "fallback_search_success": True,
                                    "retry_grab_attempted": True,
                                    "retry_grab_success": False,
                                    "cube_moved_during_grab": True,
                                    "grab_failure_reason": self._last_grab_failure_reason,
                                    "return_home_attempted": True,
                                    "return_home_success": return_home_success,
                                },
                                success=False,
                            )
                            continue

                        video_meta: Dict[str, Any] = {}
                        if self.video_writer is not None:
                            video_meta = self.video_writer.commit_episode()
                        self.writer.commit_sequence(
                            {
                                "search_success": True,
                                "grab_success": True,
                                "episode_id": sequence_id,
                                "video": video_meta if video_meta else None,
                                "fallback_search_attempted": True,
                                "fallback_search_success": True,
                                "retry_grab_attempted": True,
                                "retry_grab_success": True,
                                "cube_moved_during_grab": True,
                            }
                        )
                        continue
                    if self.video_writer is not None:
                        self.video_writer.abort_episode()
                    self.writer.abort_sequence("grab_failed")
                    continue

                video_meta: Dict[str, Any] = {}
                if self.video_writer is not None:
                    video_meta = self.video_writer.commit_episode()

                self.writer.commit_sequence(
                    {
                        "search_success": True,
                        "grab_success": True,
                        "episode_id": sequence_id,
                        "video": video_meta if video_meta else None,
                    }
                )
            
            except Exception:
                # Ensure a failed episode cannot leave the writer in an active-buffer state.
                if self.video_writer is not None:
                    try:
                        self.video_writer.abort_episode()
                    except RuntimeError:
                        pass
                try:
                    self.writer.abort_sequence("exception")
                except RuntimeError:
                    pass
                raise

    def search(self, visibility_padding: float = 0.1):
        """Search movement, following path from file to search eintier workplate."""
        search_path = self._load_search_path()

        for checkpoint_idx, target_actuator_deg in enumerate(search_path):
            while True:
                # read current joint rotations
                state = self.env.get_state(image=False)
                current_actuator_rad = state["actuator_rotations"]
                current_actuator_deg = [math.degrees(rad) for rad in current_actuator_rad]

                # check if target cube is visible at current pose with padding
                is_target_visible = self.env.target_cube_in_view(padding=visibility_padding)
                if is_target_visible:
                    return True

                # target reached
                if all(abs(current - target) < 0.5 for current, target in zip(current_actuator_deg, target_actuator_deg)):
                    break
                
                # move towards target
                self._move_to_target(
                    current_actuator_deg,
                    target_actuator_deg,
                    phase="search",
                    state_before=state,
                    step_meta={
                        "search_checkpoint_index": checkpoint_idx,
                        "visibility_padding": visibility_padding,
                    },
                )
        
        return False

    def _return_home(self, reason: str = "episode_end") -> bool:
        """Move the robot back to the configured home pose and log the motion as its own phase."""
        try:
            while True:
                state = self.env.get_state(image=False)
                current_actuator_rad = state["actuator_rotations"]
                current_actuator_deg = [math.degrees(rad) for rad in current_actuator_rad]

                # No-op return is allowed when search already ended at home.
                if all(
                    abs(current - target) < 0.5
                    for current, target in zip(current_actuator_deg, HOME_ACTUATOR_DEG)
                ):
                    return True

                self._move_to_target(
                    current_actuator_deg,
                    HOME_ACTUATOR_DEG,
                    phase="return_home",
                    state_before=state,
                    step_meta={
                        "return_reason": reason,
                    },
                )
        except Exception:
            # Best-effort cleanup path for non-optimal episodes should not discard the episode on return failure.
            return False

    def grab(
        self,
        waypoint_step_mm: float = 20,
        allow_cube_disappear: bool = False,
        allow_cube_move: bool = False,
        cube_disappear_distance_mm: float = 50.0,
        cube_move_distance_mm: float = 50.0,
    ) -> bool:
        """Approach cube with look-at IK waypoints and close gripper at final pre-grasp pose."""
        self._last_grab_failure_reason = None

        def _normalize(vec: List[float], eps: float = 1e-8) -> Optional[List[float]]:
            norm = math.sqrt(sum(v * v for v in vec))
            if norm < eps:
                return None
            return [v / norm for v in vec]

        def _cross(a: List[float], b: List[float]) -> List[float]:
            return [
                a[1] * b[2] - a[2] * b[1],
                a[2] * b[0] - a[0] * b[2],
                a[0] * b[1] - a[1] * b[0],
            ]

        def _dot(a: List[float], b: List[float]) -> float:
            return sum(x * y for x, y in zip(a, b))

        def _project_orthogonal(vec: List[float], normal: List[float]) -> Optional[List[float]]:
            """Project vec onto plane orthogonal to normal and normalize."""
            dot_vn = _dot(vec, normal)
            projected = [vec[i] - dot_vn * normal[i] for i in range(3)]
            return _normalize(projected)

        def _lookat_rotation_deg(waypoint_mm: List[float], target_mm: List[float]) -> List[float]:
            """Build a stable look-at rotation for a waypoint that points the tool Z axis to the cube."""
            forward = _normalize([target_mm[i] - waypoint_mm[i] for i in range(3)])
            if forward is None:
                forward = [0.0, 1.0, 0.0]

            z_axis = forward
            x_axis = _project_orthogonal([0.0, 0.0, 1.0], z_axis)
            if x_axis is None:
                x_axis = _project_orthogonal([1.0, 0.0, 0.0], z_axis)
            if x_axis is None:
                x_axis = [1.0, 0.0, 0.0]

            y_axis = _normalize(_cross(z_axis, x_axis))
            if y_axis is None:
                y_axis = [0.0, 1.0, 0.0]
            x_axis = _normalize(_cross(y_axis, z_axis)) or x_axis

            lookat_rot = Rotation.from_matrix([
                [x_axis[0], y_axis[0], z_axis[0]],
                [x_axis[1], y_axis[1], z_axis[1]],
                [x_axis[2], y_axis[2], z_axis[2]],
            ])
            # Apply the fixed frame remap expected by the IK solver / environment convention.
            return (
                lookat_rot * Rotation.from_euler("xyz", [90, 0, 0], degrees=True)
            ).as_euler("xyz", degrees=True).tolist()

        initial_state = self.env.get_state(image=False)
        if initial_state.get("collisions", False):
            self._last_grab_failure_reason = "grab_initial_collision"
            return False

        # calculate target cube position
        current_actuator_deg = [math.degrees(rad) for rad in initial_state["actuator_rotations"]]
        cube_mm = [1000.0 * v for v in initial_state["target_cube_location"]]
        cube_z_rotation_deg = math.degrees(initial_state["target_cube_rotation"][2])

        # calculate current end-effector position with FK
        fk_joints_deg = self._venv_to_kinematics(current_actuator_deg)
        self.fk_model.set_joint_angles(*fk_joints_deg)
        # Flatten the FK output because slicing a numpy.matrix returns a column matrix (3x1).
        ee_mm = np.asarray(self.fk_model.get_joint_translation_vector(5), dtype=float).reshape(-1).tolist()
        
        # move closer to cube on x, y axis by an amount relative to distance_to_target
        dx = cube_mm[0] - ee_mm[0]
        dy = cube_mm[1] - ee_mm[1]
        distance_xy = math.sqrt(dx * dx + dy * dy)
        distance_to_target = math.sqrt(cube_mm[0] ** 2 + cube_mm[1] ** 2)
        
        approach_distance = 100 if distance_to_target < 500 else (1000 - distance_to_target) / 5
        
        if distance_xy > approach_distance:
            # move closer towards cube
            scale = (distance_xy - approach_distance) / distance_xy
            ee_mm[0] = ee_mm[0] + dx * (1 - scale)
            ee_mm[1] = ee_mm[1] + dy * (1 - scale)
        else:
            # move to cube position if closer than approach_distance
            ee_mm[0] = cube_mm[0]
            ee_mm[1] = cube_mm[1]

        # flip y axis to match robot coordinate convention
        ee_mm[1]   = -ee_mm[1]
        cube_mm[1] = -cube_mm[1]
        
        # calculate target grasp pose
        grasp_pos = [cube_mm[i] for i in range(3)]
        grasp_rot = [
            -25 if distance_to_target > 440 else -90, 
            0, 
            self._calculate_optimal_grasp_rotation(cube_mm, cube_z_rotation_deg),
        ]

        # calculate final grasp pose
        self.ik_solver.set_end_effector(grasp_pos, grasp_rot)
        ik_joint_deg = self.ik_solver.calc_inverse_kinematics()
        final_target_actuator_deg = self._kinematics_to_venv(ik_joint_deg)

        total_distance_mm = math.sqrt(sum((cube_mm[i] - ee_mm[i]) ** 2 for i in range(3)))
        num_waypoints = max(2, int(math.ceil(total_distance_mm / max(1e-6, waypoint_step_mm))))

        # calculate waypoint positions and rotations
        waypoint_pose = []
        first_waypoint_alpha = 1.0 / num_waypoints
        first_waypoint_mm = [
            ee_mm[i] + first_waypoint_alpha * (grasp_pos[i] - ee_mm[i]) for i in range(3)
        ]
        first_waypoint_rot_deg = _lookat_rotation_deg(first_waypoint_mm, grasp_pos)

        # First waypoint is a look-at pose so the robot initially faces the cube during approach.
        self.ik_solver.set_end_effector(first_waypoint_mm, first_waypoint_rot_deg)
        ik_joint_deg = self.ik_solver.calc_inverse_kinematics()
        target_actuator_deg = self._kinematics_to_venv(ik_joint_deg)
        target_actuator_deg[5] = final_target_actuator_deg[5]
        waypoint_pose.append(target_actuator_deg)

        if num_waypoints > 2:
            # Interpolate orientation smoothly from the first look-at waypoint to the final grasp pose.
            rotation_slerp = Slerp(
                [0.0, 1.0],
                Rotation.from_euler("xyz", [first_waypoint_rot_deg, grasp_rot], degrees=True),
            )

            for wp_idx in range(2, num_waypoints):
                t = (wp_idx - 1) / (num_waypoints - 1)

                # Interpolate from the first waypoint to the final grasp pose.
                waypoint_mm = [
                    first_waypoint_mm[i] + t * (grasp_pos[i] - first_waypoint_mm[i]) for i in range(3)
                ]
                waypoint_rot_deg = rotation_slerp([t])[0].as_euler("xyz", degrees=True).tolist()

                self.ik_solver.set_end_effector(waypoint_mm, waypoint_rot_deg)
                ik_joint_deg = self.ik_solver.calc_inverse_kinematics()
                target_actuator_deg = self._kinematics_to_venv(ik_joint_deg)
                target_actuator_deg[5] = final_target_actuator_deg[5]
                waypoint_pose.append(target_actuator_deg)

        # calculate inverse kinematics for final grasp pose and add to waypoints
        waypoint_pose.append(final_target_actuator_deg)

        # move through waypoints
        cube_disappearance_triggered = False
        cube_move_triggered = False
        cube_disappear_distance_m = max(0.0, float(cube_disappear_distance_mm)) / 1000.0
        cube_move_distance_m = max(0.0, float(cube_move_distance_mm)) / 1000.0
        for waypoint_idx, target_actuator_deg in enumerate(waypoint_pose):
            state = self.env.get_state(image=False)

            current_actuator_deg = [math.degrees(rad) for rad in state["actuator_rotations"]]
            planned_segment = self._plan_collision_free_segment(
                current_actuator_deg=current_actuator_deg,
                target_actuator_deg=target_actuator_deg,
                cube_mm=cube_mm,
                cube_z_rotation_deg=cube_z_rotation_deg,
            )
            if planned_segment is None:
                print("No collision-free path found for segment, aborting grab.")
                self._last_grab_failure_reason = "grab_path_planning_failed"
                return False

            # Execute each planned path waypoint; abort immediately on any observed collision.
            for segment_step_idx, planned_target_actuator_deg in enumerate(planned_segment[1:], start=1):
                while True:
                    state = self.env.get_state(image=False)
                    
                    current_actuator_rad = state["actuator_rotations"]
                    current_actuator_deg = [math.degrees(rad) for rad in current_actuator_rad]

                    # Trigger the disappearance scenario only once, and only near the target.
                    if allow_cube_disappear and not cube_disappearance_triggered:
                        distance_to_target_m = state.get("distance_to_target")
                        if isinstance(distance_to_target_m, (int, float)) and distance_to_target_m <= cube_disappear_distance_m:
                            self.env.set_cube_gone()
                            cube_disappearance_triggered = True
                            self._last_grab_failure_reason = "cube_disappeared_during_grab"
                            return False
                    if allow_cube_move and not cube_move_triggered:
                        distance_to_target_m = state.get("distance_to_target")
                        if isinstance(distance_to_target_m, (int, float)) and distance_to_target_m <= cube_move_distance_m:
                            self.env.move_cube_random_on_workplate()
                            cube_move_triggered = True
                            self._last_grab_failure_reason = "cube_moved_during_grab"
                            return False

                    if all(
                        abs(current - target) < 0.5
                        for current, target in zip(current_actuator_deg, planned_target_actuator_deg)
                    ):
                        break

                    self._move_to_target(
                        current_actuator_deg,
                        planned_target_actuator_deg,
                        phase="grab",
                        state_before=state,
                        step_meta={
                            "waypoint_index": waypoint_idx,
                            "planned_segment_step_index": segment_step_idx,
                        },
                    )

        self._last_grab_failure_reason = None
        return True
    
    def _load_search_path(self) -> List[List[float]]:
        """Load search path from JSON file, which is a list of waypoints with 6-DOF actuator targets in degrees."""
        path = Path(__file__).resolve().parent / "search_path.json"

        with open(path, "r", encoding="utf-8") as f:
            search_path = json.load(f)

        return search_path["search_path"]
    
    def _move_to_target(
        self,
        current_actuator_deg: List[float],
        target_actuator_deg: List[float],
        phase: str,
        state_before: Optional[Dict[str, Any]] = None,
        step_meta: Optional[Dict[str, Any]] = None,
    ):
        """Move robot to target actuator positions using a simple velocity-based controller that respects max joint velocities and actuator bounds."""   
        # compute the delta rotations
        delta_rotations = []
        for i, (current, target) in enumerate(zip(current_actuator_deg, target_actuator_deg)):
            current = (current + 360) % 360
            target = (target + 360) % 360
            delta = target - current
            delta_rotations.append(delta if abs(delta) <= 180 else delta - 360 * math.copysign(1, delta))
        
        # compute time for each joint to reach target at max velocity and determine max time needed
        times_to_target = []
        max_time = 0.0
        for i, delta in enumerate(delta_rotations):
            time = abs(delta) / MAX_JOINT_VELOCITY[i]
            times_to_target.append(time)
            max_time = max(max_time, time)

        # calculate relative velocity for each joint to reach target at the same time
        velocity_commands = []
        for i, delta in enumerate(delta_rotations):
            velocity = (delta / max_time)
            velocity_commands.append(velocity * STEP_VELOCITY_SIGN[i])

        # Logged sequence state is intentionally minimal; images are stored in episode video files.
        if state_before is None:
            state_before_record = self._get_logging_state()
        elif self.video_writer is not None and state_before.get("image") is None:
            state_before_record = self._get_logging_state(state_without_image=state_before)
        else:
            state_before_record = state_before
        self.env.step(velocity_commands, grapper_state=False)
        state_after_record = self._get_logging_state()

        state_before_min = self._extract_logged_state(state_before_record)
        state_after_min = self._extract_logged_state(state_after_record)

        step_record: Dict[str, Any] = {
            "phase": phase,
            "state_before": state_before_min,
            "action": {
                "actuator_velocities": [float(v) for v in velocity_commands],
            },
            "state_after": state_after_min,
        }

        meta_payload: Dict[str, Any] = dict(step_meta or {})
        if self.video_writer is not None:
            frame_before_idx = self.video_writer.append_frame(state_before_record.get("image"))
            frame_after_idx = self.video_writer.append_frame(state_after_record.get("image"))
            meta_payload.update(
                {
                    "image_frame_before_idx": frame_before_idx,
                    "image_frame_after_idx": frame_after_idx,
                }
            )
        if meta_payload:
            step_record["meta"] = meta_payload

        self.writer.append_step(
            step_record
        )

    def _plan_collision_free_segment(
        self,
        current_actuator_deg: List[float],
        target_actuator_deg: List[float],
        cube_mm: List[float],
        cube_z_rotation_deg: float,
    ) -> Optional[List[List[float]]]:
        """Plan a collision-free joint path for one grab segment in environment actuator convention."""
        start_kin_deg = self._venv_to_kinematics(current_actuator_deg)
        goal_kin_deg = self._venv_to_kinematics(target_actuator_deg)

        if self._collision_model.is_edge_valid(start_kin_deg, goal_kin_deg, cube_mm, cube_z_rotation_deg):
            return [current_actuator_deg, target_actuator_deg]

        planned_path = plan_and_smooth(
            start=start_kin_deg,
            goal=goal_kin_deg,
            collision_model=self._collision_model,
            joint_limits=None,
            cube_position=cube_mm,
            cube_z_rotation=cube_z_rotation_deg,
        )

        if planned_path is None:
            return None

        return [[float(v) for v in waypoint] for waypoint in planned_path]

    @staticmethod
    def _extract_logged_state(state: Dict[str, Any]) -> Dict[str, Any]:
        """Keep only the transformer-relevant state fields in the dataset."""
        return {
            "actuator_rotations": state.get("actuator_rotations"),
            "actuator_velocities": state.get("actuator_velocities"),
        }

    @staticmethod
    def _calculate_optimal_grasp_rotation(cube_mm: List[float], cube_z_rotation: float) -> float:
        """Calculate a stable grasp yaw in degrees.

        Expects:
        - ``cube_mm`` in the robot/IK position frame (the caller already flips ``y``)
        - ``cube_z_rotation`` in degrees from the environment cube pose
        """
        def _normalize_deg(angle_deg: float) -> float:
            """Normalize any angle to [-180, 180)."""
            return ((angle_deg + 180.0) % 360.0) - 180.0

        x_mm, y_mm = cube_mm[0], cube_mm[1]
        distance_xy = math.hypot(x_mm, y_mm)
        if distance_xy < 1e-6:
            snapped_candidates = [_normalize_deg(cube_z_rotation + 90.0 * i) for i in range(4)]
            return min(snapped_candidates, key=abs)
        
        cube_yaw_deg_in_ik_frame = -cube_z_rotation
        radial_offset_deg = math.degrees(math.atan2(x_mm, y_mm))
        cube_rel_deg = _normalize_deg(cube_yaw_deg_in_ik_frame + radial_offset_deg)
        rel_candidates_deg = [_normalize_deg(cube_rel_deg + 90.0 * i) for i in range(4)]

        best_rel_deg = min(rel_candidates_deg, key=abs)

        return _normalize_deg(best_rel_deg - radial_offset_deg)

    @staticmethod
    def _venv_to_kinematics(joint_angles_deg: List[float]) -> List[float]:
        """Convert environment actuator angles in degrees to the FK joint convention."""
        return [
            90 + joint_angles_deg[0],
            joint_angles_deg[1],
            joint_angles_deg[2],
            joint_angles_deg[3],
            joint_angles_deg[4],
            - joint_angles_deg[5],
        ]
    
    @staticmethod
    def _kinematics_to_venv(joint_angles_deg: List[float]) -> List[float]:
        """Convert FK joint angles in degrees to the environment actuator convention."""
        return [
            joint_angles_deg[0],
            joint_angles_deg[1],
            - joint_angles_deg[2],
            - joint_angles_deg[3],
            joint_angles_deg[4],
            - joint_angles_deg[5],
        ]

if __name__ == "__main__":
    config = Config()
    env = EnvClient(config.host, config.port)
    env.connect()
    # Store videos at 10 Hz for easier playback review and alignment with the desired robot-speed preview.
    with JSONLWriter("docs/data/expert_data.jsonl") as writer, EpisodeVideoWriter("docs/data/expert_videos", fps=10) as video_writer:
        generator = OptimalExpertGenerator(env, config, writer, video_writer=video_writer)
        generator.generate(num_episodes=10000)
    
    env.close()
