"""Automation runner for expert data generation using the Blender RPC environment.

This script does not modify any generation logic. It only orchestrates:
1) launching Blender with the project .blend scene,
2) auto-starting the EnvControl RPC server inside Blender,
3) running the existing expert data generator script, and
4) shutting Blender down after generation completes.

It is designed to run locally and on AWS (Linux) by optionally wrapping Blender
with ``xvfb-run`` when no display is available.
"""

from __future__ import annotations

import argparse
import os
import queue
import shlex
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import IO, List, Optional


STARTUP_READY_MARKERS = (
    "RL Env Server started (modal timer)",
    "AUTOSTART: server operator started",
)


@dataclass
class ProcessStreams:
    """Container for Blender stdout reader state."""

    thread: threading.Thread
    stop_event: threading.Event
    ready_event: threading.Event
    line_queue: "queue.Queue[str]"


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for local/AWS orchestration."""
    repo_root = Path(__file__).resolve().parents[2]
    default_blend = repo_root / "src" / "virtual_robot_environment" / "Robot_V2_environment - Kopie.blend"
    default_env_control = repo_root / "src" / "virtual_robot_environment" / "EnvControl.py"
    default_generator = repo_root / "src" / "expert_data_generation" / "data_generator.py"

    parser = argparse.ArgumentParser(
        description="Launch Blender EnvControl automatically and run expert data generation.",
    )
    parser.add_argument(
        "--blender-exec",
        default=os.environ.get("BLENDER_BIN", "blender"),
        help="Blender executable path (default: BLENDER_BIN env var or 'blender').",
    )
    parser.add_argument(
        "--blend-file",
        type=Path,
        default=default_blend,
        help="Path to the Blender .blend scene file.",
    )
    parser.add_argument(
        "--env-control-script",
        type=Path,
        default=default_env_control,
        help="Path to EnvControl.py inside the repository.",
    )
    parser.add_argument(
        "--generator-script",
        type=Path,
        default=default_generator,
        help="Path to the existing expert data generator script.",
    )
    parser.add_argument(
        "--python-exec",
        default=sys.executable,
        help="Python executable used to run the generator script.",
    )
    parser.add_argument(
        "--host",
        default="localhost",
        help="Host for the Blender RPC server bind/connect (default: localhost).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5055,
        help="Port for the Blender RPC server (default: 5055).",
    )
    parser.add_argument(
        "--startup-timeout-s",
        type=float,
        default=120.0,
        help="Seconds to wait for Blender EnvControl server startup.",
    )
    parser.add_argument(
        "--shutdown-timeout-s",
        type=float,
        default=10.0,
        help="Seconds to wait for Blender shutdown before force kill.",
    )
    parser.add_argument(
        "--use-xvfb",
        choices=("auto", "always", "never"),
        default="auto",
        help="Linux/AWS display strategy. 'auto' uses xvfb-run when DISPLAY is missing.",
    )
    parser.add_argument(
        "--xvfb-screen",
        default="1280x720x24",
        help="Virtual screen config passed to xvfb-run (default: 1280x720x24).",
    )
    parser.add_argument(
        "--blender-log-file",
        type=Path,
        default=None,
        help="Optional file path to store Blender stdout/stderr logs.",
    )
    parser.add_argument(
        "--keep-blender-open",
        action="store_true",
        help="Leave Blender running after the generator exits (debugging).",
    )
    parser.add_argument(
        "--generator-args",
        nargs=argparse.REMAINDER,
        default=[],
        help="Extra args forwarded to the generator script after '--generator-args'.",
    )
    return parser.parse_args()


def ensure_inputs_exist(args: argparse.Namespace) -> None:
    """Validate local repository paths before starting subprocesses."""
    for label, path in (
        ("blend file", args.blend_file),
        ("EnvControl script", args.env_control_script),
        ("generator script", args.generator_script),
    ):
        if not Path(path).exists():
            raise FileNotFoundError(f"Missing {label}: {path}")


def build_blender_bootstrap_script(
    env_control_path: Path,
    host: str,
    port: int,
    ready_sentinel_path: Path,
) -> str:
    """Create a temporary Blender Python bootstrap script that autostarts EnvControl."""
    env_control_path_posix = env_control_path.resolve().as_posix()
    host_literal = host.replace("\\", "\\\\").replace('"', '\\"')
    ready_sentinel_literal = ready_sentinel_path.resolve().as_posix()
    # Keep the bootstrap script minimal and explicit because it runs inside Blender's Python.
    return f"""import importlib.util
import traceback
import bpy

ENV_CONTROL_PATH = r"{env_control_path_posix}"
SERVER_HOST = "{host_literal}"
SERVER_PORT = {int(port)}
READY_SENTINEL_PATH = r"{ready_sentinel_literal}"

def _load_envcontrol_module():
    spec = importlib.util.spec_from_file_location("robot_envcontrol_autostart", ENV_CONTROL_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load EnvControl module from {{ENV_CONTROL_PATH}}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

module = _load_envcontrol_module()
module.HOST = SERVER_HOST
module.PORT = SERVER_PORT
module.env = module.RobotEnv(fps=20)
module.register()
print(f"AUTOSTART: EnvControl loaded host={{SERVER_HOST}} port={{SERVER_PORT}}")

def _start_server_when_ui_ready():
    # Under Xvfb, bpy.context.window can remain None even after the window manager is ready.
    window_manager = getattr(bpy.context, "window_manager", None)
    windows = list(getattr(window_manager, "windows", [])) if window_manager is not None else []
    if not windows:
        print("AUTOSTART: waiting for Blender window manager windows...")
        return 0.25
    window = windows[0]
    try:
        # Start the modal operator with an explicit window override so the timer can attach to a real UI window.
        if hasattr(bpy.context, "temp_override"):
            with bpy.context.temp_override(window=window, screen=window.screen):
                result = bpy.ops.wm.rl_env_server_modal()
        else:
            result = bpy.ops.wm.rl_env_server_modal({{"window": window, "screen": window.screen}})
        # The wrapper waits on this file so headless runs do not depend on stdout flushing.
        with open(READY_SENTINEL_PATH, "w", encoding="utf-8") as ready_handle:
            ready_handle.write("ready\\n")
        print(f"AUTOSTART: server operator started -> {{result}}")
    except Exception as exc:
        print(f"AUTOSTART: failed to start server operator: {{exc}}")
        traceback.print_exc()
    return None

bpy.app.timers.register(_start_server_when_ui_ready, first_interval=0.1)
"""


def write_temp_bootstrap(contents: str) -> Path:
    """Write the generated Blender bootstrap code to a temporary file."""
    temp_dir = Path(tempfile.gettempdir())
    temp_file = temp_dir / f"robot_env_autostart_{int(time.time() * 1000)}.py"
    temp_file.write_text(contents, encoding="utf-8")
    return temp_file


def should_wrap_with_xvfb(mode: str) -> bool:
    """Decide whether to run Blender under xvfb-run (useful on AWS Linux instances)."""
    if mode == "never":
        return False
    if mode == "always":
        return True
    # Auto mode: use Xvfb when running on Linux without a display.
    return sys.platform.startswith("linux") and not os.environ.get("DISPLAY")


def spawn_blender_log_reader(
    stdout_pipe: Optional[IO[str]],
    log_file: Optional[Path],
) -> ProcessStreams:
    """Start a background thread that mirrors Blender logs and detects startup readiness."""
    stop_event = threading.Event()
    ready_event = threading.Event()
    line_queue: "queue.Queue[str]" = queue.Queue()

    def _reader() -> None:
        log_handle: Optional[IO[str]] = None
        try:
            if log_file is not None:
                log_file.parent.mkdir(parents=True, exist_ok=True)
                log_handle = log_file.open("w", encoding="utf-8")

            if stdout_pipe is None:
                return

            for raw_line in stdout_pipe:
                line = raw_line.rstrip("\n")
                line_queue.put(line)
                print(f"[blender] {line}")
                if log_handle is not None:
                    log_handle.write(line + "\n")
                    log_handle.flush()
                if any(marker in line for marker in STARTUP_READY_MARKERS):
                    ready_event.set()
                if stop_event.is_set():
                    break
        finally:
            if log_handle is not None:
                log_handle.close()

    thread = threading.Thread(target=_reader, name="blender-log-reader", daemon=True)
    thread.start()
    return ProcessStreams(thread=thread, stop_event=stop_event, ready_event=ready_event, line_queue=line_queue)


def wait_for_blender_server_start(
    blender_proc: subprocess.Popen[str],
    streams: ProcessStreams,
    timeout_s: float,
    ready_sentinel_path: Path,
) -> None:
    """Wait for Blender to report readiness or fail fast if it exits."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if ready_sentinel_path.exists():
            return
        if streams.ready_event.wait(timeout=0.25):
            return
        exit_code = blender_proc.poll()
        if exit_code is not None:
            raise RuntimeError(f"Blender exited before EnvControl server startup (exit code {exit_code}).")
    raise TimeoutError(
        f"Timed out after {timeout_s:.1f}s waiting for Blender EnvControl startup. "
        "Check Blender logs for errors."
    )


def run_generator_process(
    python_exec: str,
    generator_script: Path,
    generator_args: List[str],
    repo_root: Path,
) -> int:
    """Run the existing generator script as-is to preserve generation logic."""
    cmd = [python_exec, str(generator_script.resolve()), *generator_args]
    print(f"[runner] Starting generator: {shlex.join(cmd)}")
    completed = subprocess.run(cmd, cwd=str(repo_root), check=False)
    return int(completed.returncode)


def terminate_process_tree(proc: subprocess.Popen[str], timeout_s: float) -> None:
    """Terminate Blender wrapper process and force kill if needed."""
    if proc.poll() is not None:
        return

    proc.terminate()
    try:
        proc.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=timeout_s)


def main() -> int:
    """Entrypoint for Blender + generator orchestration."""
    args = parse_args()
    ensure_inputs_exist(args)
    if str(args.host) != "localhost" or int(args.port) != 5055:
        # The existing data_generator.py entrypoint hardcodes localhost:5055.
        # This wrapper keeps generation logic untouched, so custom host/port would mismatch.
        raise ValueError(
            "Custom --host/--port are not supported with the current unchanged "
            "data_generator.py entrypoint. Use localhost:5055 or update the wrapper "
            "to import and call the generator classes directly."
        )

    repo_root = Path(__file__).resolve().parents[2]
    ready_sentinel_path = Path(tempfile.gettempdir()) / f"robot_env_ready_{int(time.time() * 1000)}.flag"
    bootstrap_contents = build_blender_bootstrap_script(
        env_control_path=Path(args.env_control_script),
        host=str(args.host),
        port=int(args.port),
        ready_sentinel_path=ready_sentinel_path,
    )
    bootstrap_file = write_temp_bootstrap(bootstrap_contents)

    blender_cmd: List[str] = [
        str(args.blender_exec),
        str(Path(args.blend_file).resolve()),
        "--python",
        str(bootstrap_file),
    ]

    if should_wrap_with_xvfb(args.use_xvfb):
        # AWS/Linux often has no DISPLAY, so Xvfb provides a virtual screen for Blender's UI modal operator.
        blender_cmd = [
            "xvfb-run",
            "-a",
            "-s",
            f"-screen 0 {args.xvfb_screen}",
            *blender_cmd,
        ]

    print(f"[runner] Starting Blender: {shlex.join(blender_cmd)}")

    blender_proc: Optional[subprocess.Popen[str]] = None
    streams: Optional[ProcessStreams] = None

    try:
        blender_proc = subprocess.Popen(
            blender_cmd,
            cwd=str(repo_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        streams = spawn_blender_log_reader(blender_proc.stdout, args.blender_log_file)
        wait_for_blender_server_start(
            blender_proc,
            streams,
            args.startup_timeout_s,
            ready_sentinel_path,
        )
        print("[runner] Blender EnvControl server is ready.")

        generator_exit_code = run_generator_process(
            python_exec=str(args.python_exec),
            generator_script=Path(args.generator_script),
            generator_args=list(args.generator_args),
            repo_root=repo_root,
        )
        print(f"[runner] Generator exited with code {generator_exit_code}.")
        return generator_exit_code
    finally:
        if streams is not None:
            streams.stop_event.set()

        if blender_proc is not None and not args.keep_blender_open:
            print("[runner] Stopping Blender.")
            terminate_process_tree(blender_proc, args.shutdown_timeout_s)
        elif blender_proc is not None and args.keep_blender_open:
            print("[runner] Leaving Blender running (--keep-blender-open set).")

        # Remove the temp bootstrap file even if generation failed.
        try:
            if bootstrap_file.exists():
                bootstrap_file.unlink()
        except OSError:
            pass
        try:
            if ready_sentinel_path.exists():
                ready_sentinel_path.unlink()
        except OSError:
            pass

        if streams is not None:
            streams.thread.join(timeout=1.0)


if __name__ == "__main__":
    raise SystemExit(main())
