"""Differentiable forward kinematics for the Dobot Magician E6, for use inside the loss.

The training objective needs to know what a joint-space prediction error does to the tool,
so the kinematic chain has to be differentiable and live where the loss is computed. This
mirrors ``scripts/e6_fk.py`` exactly — same URDF, same convention, same joint order — but in
JAX, and it also exposes the task Jacobian.

What the task map returns is deliberately narrow. For an axisymmetric suction cup the task
is defined by where the tool is and which way it points; rotation about the tool axis does
not change either, and the URDF agrees (j6 moves neither the tool origin nor its z-axis).
Returning a full SO(3) term would make the objective penalise a rotation the task cannot
feel. This choice is end-effector specific and is the part that has to be redefined for a
gripper, where roll about the tool axis does matter.

    from openpi.models.e6_fk_jax import E6Chain
    chain = E6Chain.from_urdf()
    phi = chain.task_map(q_deg)            # (..., 6) = [x, y, z (mm), tool_z unit vector]
    J   = chain.task_jacobian(q_deg)       # (..., 6, 6) d phi / d q, per degree

The Jacobian is evaluated on ground-truth poses, so it enters the loss as a constant and
nothing has to be back-propagated through the chain itself.
"""

from __future__ import annotations

import dataclasses
import pathlib
import xml.etree.ElementTree as ET

import jax
import jax.numpy as jnp
import numpy as np

DEFAULT_URDF = pathlib.Path("/home/billy/26kp/ydg/DOBOT_6Axis_ROS2_V4/dobot_rviz/urdf/me6_robot.urdf")
MM_PER_M = 1000.0


def _rpy_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr, cp, sp, cy, sy = np.cos(roll), np.sin(roll), np.cos(pitch), np.sin(pitch), np.cos(yaw), np.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


def _read_urdf(path: pathlib.Path) -> tuple[np.ndarray, np.ndarray]:
    """Fixed 4x4 transform preceding each revolute joint, and its rotation axis."""
    root = ET.parse(path).getroot()
    origins, axes = [], []
    for joint in root.findall("joint"):
        if joint.get("type") != "revolute":
            continue
        origin = joint.find("origin")
        xyz = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ") if origin is not None else np.zeros(3)
        rpy = np.fromstring(origin.get("rpy", "0 0 0"), sep=" ") if origin is not None else np.zeros(3)
        axis = joint.find("axis")
        t = np.eye(4)
        t[:3, :3] = _rpy_matrix(*rpy)
        t[:3, 3] = xyz
        origins.append(t)
        axes.append(np.fromstring(axis.get("xyz", "0 0 1"), sep=" ") if axis is not None else np.array([0.0, 0.0, 1.0]))
    return np.stack(origins), np.stack(axes)


# eq=False: nnx compares module structure for equality, and a generated __eq__ would
# compare numpy arrays element-wise, which raises rather than returning a bool. These
# are fixed constants loaded from a file, so identity is the comparison that fits.
@dataclasses.dataclass(frozen=True, eq=False)
class E6Chain:
    """The chain geometry, held as numpy rather than jax arrays.

    This matters and is easy to get wrong. The model is constructed inside `jax.jit`, so
    anything built with `jnp.asarray` during construction is a tracer, and storing it on the
    module lets it escape the trace — training dies with a leaked-tracer error while every
    test that builds the model eagerly passes. Numpy arrays are inlined as constants by
    whatever traced function uses them, which is what these are.

    For the same reason this is not registered as a pytree node: the geometry is fixed by the
    robot, not something to differentiate through or map over.
    """

    origins: np.ndarray  # (n, 4, 4) fixed transform preceding each joint
    axes: np.ndarray  # (n, 3), unit
    source: str = ""
    """The URDF this came from. Two chains read from the same file are the same chain, which
    is what module-structure comparison needs to see."""

    def __eq__(self, other) -> bool:
        return isinstance(other, E6Chain) and self.source == other.source

    def __hash__(self) -> int:
        return hash(self.source)

    @classmethod
    def from_urdf(cls, path: pathlib.Path = DEFAULT_URDF) -> "E6Chain":
        origins, axes = _read_urdf(path)
        axes = axes / np.linalg.norm(axes, axis=-1, keepdims=True)
        return cls(origins.astype(np.float32), axes.astype(np.float32), source=str(path))

    @property
    def n_joints(self) -> int:
        return self.axes.shape[0]

    def _rot(self, axis: np.ndarray, angle: jnp.ndarray) -> jnp.ndarray:
        """Rodrigues about an arbitrary axis, so a non-z URDF axis still works."""
        kx = np.array(
            [
                [0.0, -axis[2], axis[1]],
                [axis[2], 0.0, -axis[0]],
                [-axis[1], axis[0], 0.0],
            ],
            dtype=np.float32,
        )
        return jnp.eye(3) + jnp.sin(angle) * kx + (1.0 - jnp.cos(angle)) * (kx @ kx)

    def pose(self, q_deg: jnp.ndarray) -> jnp.ndarray:
        """Tool pose as 4x4 in the base frame. `q_deg` is (n_joints,) in degrees."""
        q = jnp.deg2rad(q_deg)
        t = jnp.eye(4)
        for i in range(self.n_joints):
            r = jnp.eye(4).at[:3, :3].set(self._rot(self.axes[i], q[i]))
            t = t @ self.origins[i] @ r
        return t

    def task_map(self, q_deg: jnp.ndarray) -> jnp.ndarray:
        """[tool position (mm), tool z-axis (unit)] — what an axisymmetric suction task sees.

        Position in millimetres rather than metres so the two blocks are not separated by a
        factor of a thousand before any weighting is applied.
        """
        t = self.pose(q_deg)
        return jnp.concatenate([t[:3, 3] * MM_PER_M, t[:3, 2]])

    def task_jacobian(self, q_deg: jnp.ndarray) -> jnp.ndarray:
        """d task_map / d q, per **degree** — matching the unit the actions are stored in."""
        return jax.jacfwd(self.task_map)(q_deg)

    # Batched forms. The chain is a short Python loop, so vmap is the cheap way to map it.
    def task_map_batched(self, q_deg: jnp.ndarray) -> jnp.ndarray:
        flat = q_deg.reshape(-1, q_deg.shape[-1])
        out = jax.vmap(self.task_map)(flat)
        return out.reshape(*q_deg.shape[:-1], out.shape[-1])

    def task_jacobian_batched(self, q_deg: jnp.ndarray) -> jnp.ndarray:
        flat = q_deg.reshape(-1, q_deg.shape[-1])
        out = jax.vmap(self.task_jacobian)(flat)
        return out.reshape(*q_deg.shape[:-1], out.shape[-2], out.shape[-1])
