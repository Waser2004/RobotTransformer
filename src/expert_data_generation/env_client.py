import json
import socket
import struct
from typing import Any, Dict, Optional


class EnvRPCError(RuntimeError):
    """Raised when the environment server responds with malformed data."""


class EnvClient:
    """Small TCP JSON-RPC client for the Blender RobotEnv server."""

    def __init__(self, host: str = "localhost", port: int = 5055, timeout_s: float = 10.0):
        self.host = host
        self.port = port
        self.timeout_s = timeout_s
        self._sock: Optional[socket.socket] = None

    def connect(self) -> None:
        """Connect once and reuse a single socket for all requests."""
        if self._sock is not None:
            return
        self._sock = socket.create_connection((self.host, self.port), timeout=self.timeout_s)

    def close(self) -> None:
        """Close the socket if connected."""
        if self._sock is not None:
            self._sock.close()
            self._sock = None

    def __enter__(self) -> "EnvClient":
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _send(self, function: str, args: Dict[str, Any]) -> Optional[Any]:
        """Send a framed request and decode an optional framed response."""
        if self._sock is None:
            raise EnvRPCError("Socket is not connected")

        payload = json.dumps({"function": function, "args": args}).encode("utf-8")
        header = struct.pack(">I", len(payload))
        self._sock.sendall(header + payload)

        # reset and set_robot_pose currently do not send a response in EnvControl.
        if function in {"reset", "set_robot_pose"}:
            return None

        response_header = self._recv_exact(4)
        if len(response_header) != 4:
            raise EnvRPCError("Did not receive full response header")
        response_len = struct.unpack(">I", response_header)[0]
        response_payload = self._recv_exact(response_len)

        try:
            response = json.loads(response_payload.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise EnvRPCError("Invalid JSON response") from exc

        if "result" not in response:
            raise EnvRPCError("Response missing 'result' field")
        return response["result"]

    def _recv_exact(self, num_bytes: int) -> bytes:
        """Read an exact byte count or fail loudly."""
        if self._sock is None:
            raise EnvRPCError("Socket is not connected")
        data = b""
        while len(data) < num_bytes:
            chunk = self._sock.recv(num_bytes - len(data))
            if not chunk:
                raise EnvRPCError("Connection closed while receiving data")
            data += chunk
        return data

    def reset(self, cube_position: str = "home", robot_pose: str = "home", actuator_rotations=None) -> None:
        """Reset environment to a known start state."""
        self._send(
            "reset",
            {
                "cube_position": cube_position,
                "robot_pose": robot_pose,
                "actuator_rotations": actuator_rotations,
            },
        )

    def get_state(self, image: bool = True) -> Dict[str, Any]:
        """Read full state payload needed by expert data generation."""
        result = self._send(
            "get_state",
            {
                "actuator_rotations": True,
                "actuator_velocities": True,
                "target_cube_state": True,
                "graper": True,
                "collisions": True,
                "workplate_coverage": False,
                "distance_to_target": False,
                "image": image,
            },
        )
        if not isinstance(result, dict):
            raise EnvRPCError("get_state returned non-dict payload")
        return result

    def step(self, actuator_velocities, grapper_state: bool) -> float:
        """Advance one env step with velocity + gripper command."""
        result = self._send(
            "step",
            {
                "actuator_velocities": list(actuator_velocities),
                "grapper_state": bool(grapper_state),
            },
        )
        return float(result)

    def target_cube_in_view(self, padding: float = 0.0) -> bool:
        """Visibility check exposed by EnvControl server with optional image-boundary padding."""
        return bool(self._send("target_cube_in_view", {"padding": float(padding)}))

    def set_robot_pose(self, actuator_rotations) -> None:
        """Directly set robot pose in radians (mostly useful for debugging)."""
        self._send("set_robot_pose", {"actuator_rotations": list(actuator_rotations)})
