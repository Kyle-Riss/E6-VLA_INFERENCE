"""Score a flow-matching residual by what it does to the tool, not by its size in coordinates.

The standard objective measures the residual `r = v_t - u_t` with a Euclidean norm in
normalized action space, which treats every dimension as equally consequential. On this arm
they are not: the same normalized error on different joints moves the tool by amounts that
differ by orders of magnitude, and the dimension the normalizer weighs most heavily is close
to the least consequential one for a suction tool.

The term added here converts the residual into degrees, propagates it along the chunk the
way the action contract actually accumulates, and asks the arm's Jacobian what that does to
the tool:

    L_task = ‖ W · J(q_gt) · A · S · r ‖²

  S  normalized residual -> degrees, from the dataset's own statistics
  A  the action contract: on this corpus a step's delta moves every later pose, verified by
     reconstructing recorded states from recorded actions to 0.000000 degrees
  J  the task Jacobian, evaluated on the ground-truth poses so it enters as a constant —
     nothing back-propagates through the kinematics
  W  what the task can feel: tool position in millimetres, and the direction the suction cup
     points, converted to a millimetre-equivalent by a characteristic tool length

Two properties this leans on. Evaluating J on ground truth makes the whole term a quadratic
form in r with a constant matrix, which is cheap and avoids reconstructing a clean action
estimate — that reconstruction carries a factor of t and would need an arbitrary time
weighting to undo. And because the term is *added* to the ordinary objective rather than
replacing it, the smallest eigenvalue of the combined form stays at one, so no direction
loses its gradient — including rotation about the tool axis, which this task genuinely
cannot feel and which the metric alone scores at exactly zero.

No joint is named anywhere here. The weighting comes out of the robot's URDF and the
dataset's statistics, so pointing it at another robot produces that robot's weighting. What
does not transfer is the task descriptor: position plus tool-axis direction suits an
axisymmetric suction cup, and a gripper whose grasp depends on roll about that axis needs a
different one.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import pathlib

import jax
import jax.numpy as jnp
import numpy as np

from openpi.models.e6_fk_jax import E6Chain


# eq=False: nnx compares module structure for equality, and a generated __eq__ would
# compare numpy arrays element-wise, which raises rather than returning a bool. These
# are fixed constants loaded from a file, so identity is the comparison that fits.
@dataclasses.dataclass(frozen=True, eq=False)
class TaskMetric:
    chain: E6Chain
    n_arm: int
    """Continuous joints the kinematics covers. Dimensions past this — the gripper and the
    padding — are not represented in the task map and keep the ordinary objective."""
    action_scale: np.ndarray  # (n_arm,) degrees per normalized unit
    action_offset: np.ndarray  # (n_arm,) degrees
    state_scale: np.ndarray  # (n_arm,)
    state_offset: np.ndarray  # (n_arm,)
    weight: np.ndarray  # (6,) task-space weighting, diagonal of W
    lam: float
    spec: dict
    fingerprint: str
    """Digest of the spec file. Two metrics loaded from the same file are the same metric, and
    nnx compares module structure across the abstract and concrete construction of the model —
    identity would make those two look different and value comparison would compare arrays."""

    def __eq__(self, other) -> bool:
        return isinstance(other, TaskMetric) and self.fingerprint == other.fingerprint

    def __hash__(self) -> int:
        return hash(self.fingerprint)

    @classmethod
    def load(cls, path: str | None) -> "TaskMetric | None":
        if path is None:
            return None
        raw = pathlib.Path(path).read_text()
        spec = json.loads(raw)
        if spec.get("reconstruction_operator") != "lower-triangular ones (cumsum)":
            raise ValueError(
                f"task metric spec declares reconstruction operator "
                f"{spec.get('reconstruction_operator')!r}; this implementation accumulates. "
                "Regenerate the spec or extend the operator rather than assuming."
            )
        ell = float(spec["tool_length_mm"])
        return cls(
            chain=E6Chain.from_urdf(pathlib.Path(spec["urdf"])),
            n_arm=int(spec["n_arm_joints"]),
            action_scale=np.asarray(spec["action_scale_deg_per_unit"], dtype=np.float32),
            action_offset=np.asarray(spec["action_offset_deg"], dtype=np.float32),
            state_scale=np.asarray(spec["state_scale_deg_per_unit"], dtype=np.float32),
            state_offset=np.asarray(spec["state_offset_deg"], dtype=np.float32),
            weight=np.asarray([1.0, 1.0, 1.0, ell, ell, ell], dtype=np.float32),
            lam=float(spec["lambda"]),
            spec=spec,
            fingerprint=hashlib.sha256(raw.encode()).hexdigest()[:16],
        )

    def _denorm(self, x: jnp.ndarray, offset: jnp.ndarray, scale: jnp.ndarray) -> jnp.ndarray:
        """Inverse of the [-1, 1] mapping the transform stage applies."""
        return (x + 1.0) * scale + offset

    def loss(
        self,
        residual: jnp.ndarray,  # (b, ah, action_dim) v_t - u_t
        actions: jnp.ndarray,  # (b, ah, action_dim) normalized ground truth
        state: jnp.ndarray,  # (b, state_dim) normalized
    ) -> jnp.ndarray:
        """Task-space penalty per (batch, step). Returns (b, ah)."""
        n = self.n_arm
        # Ground-truth poses along the chunk. Only the offsets differ between the residual
        # and the actions here: a residual has no offset, an absolute action does.
        dq_gt = self._denorm(actions[..., :n], self.action_offset, self.action_scale)
        q0 = self._denorm(state[..., :n], self.state_offset, self.state_scale)
        q_gt = q0[:, None, :] + jnp.cumsum(dq_gt, axis=1)  # (b, ah, n)

        # The Jacobian is a property of the demonstrated trajectory, not of the prediction.
        # Detaching it says so, and keeps the kinematics out of the backward pass.
        jac = jax.lax.stop_gradient(self.chain.task_jacobian_batched(q_gt))  # (b, ah, 6, n)

        # A step's residual displaces every pose from that step onward, so the joint error
        # seen at step h is the running sum up to h — the same accumulation the actions use.
        dq_err = jnp.cumsum(residual[..., :n] * self.action_scale, axis=1)  # (b, ah, n)
        task_err = jnp.einsum("bhij,bhj->bhi", jac, dq_err) * self.weight
        return jnp.square(task_err).sum(axis=-1)



def calibration_report(metric: TaskMetric) -> dict:
    """What the spec committed to, for logging next to the first training step."""
    return {
        "lambda": metric.lam,
        "lambda_rule": metric.spec.get("lambda_rule"),
        "n_arm_joints": metric.n_arm,
        "tool_length_mm": metric.spec.get("tool_length_mm"),
        "scale_source": metric.spec.get("scale_source"),
        "split": metric.spec.get("split"),
        "action_scale_deg_per_unit": np.asarray(metric.action_scale).tolist(),
    }
