# SPDX-License-Identifier: MIT
"""J02: kinematic parity. Every URDF link's pose in MuJoCo equals the URDF's forward kinematics.

Configurations: L1000 / L1175 / L1350 x bp000 / bp050 with the arms at zero, plus 50 random joint samples (uniform
within the URDF limits, fixed seed) per base plate. The reference is the FK in conftest.py (numpy on the expanded URDF),
cross-checked against the description package's own urdf_utils.zero_pose_poses.

Tolerance 1e-9 m / 1e-9 rad, fixed before the first run: floating-point composition over ~20 joints is ~1e-14; anything
near 1e-9 is a modelling difference, not rounding.
"""

import sys

import numpy as np
import pytest
from conftest import (AXES, DESCRIPTION, STOPS, generate, mujoco, random_q, rotation_angle, set_qpos, urdf_fk,
                      urdf_root)

POS_TOL, ROT_TOL = 1e-9, 1e-9
N_RANDOM = 50


def _worst(model, data, root, q):
    ref = urdf_fk(root, q)
    set_qpos(model, data, root, q)
    worst_p, worst_r = (0.0, ""), (0.0, "")
    for link, t in ref.items():
        b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, link)
        assert b >= 0, f"URDF link {link} has no MuJoCo body"
        dp = float(np.linalg.norm(data.xpos[b] - t[:3, 3]))
        dr = rotation_angle(data.xmat[b].reshape(3, 3), t[:3, :3])
        worst_p = max(worst_p, (dp, link))
        worst_r = max(worst_r, (dr, link))
    return worst_p, worst_r


@pytest.mark.parametrize("bp", AXES["base_plate"])
@pytest.mark.parametrize("stop", list(STOPS))
def test_lift_stops(bp, stop, model_for):
    model, _ = model_for(generate.Config(base_plate=bp))
    data = mujoco.MjData(model)
    (dp, lp), (dr, lr) = _worst(model, data, urdf_root(bp), {"lift_joint": STOPS[stop]})
    print(f"{bp} {stop}: worst position error {dp:.3e} m ({lp}), worst rotation error {dr:.3e} rad ({lr})")
    assert dp <= POS_TOL and dr <= ROT_TOL


@pytest.mark.parametrize("bp", AXES["base_plate"])
def test_random_joint_samples(bp, model_for):
    model, _ = model_for(generate.Config(base_plate=bp))
    data = mujoco.MjData(model)
    root = urdf_root(bp)
    rng = np.random.default_rng(20261112)
    worst_p, worst_r = (0.0, ""), (0.0, "")
    for _ in range(N_RANDOM):
        p, r = _worst(model, data, root, random_q(root, rng))
        worst_p, worst_r = max(worst_p, p), max(worst_r, r)
    print(f"{bp}, {N_RANDOM} samples: worst position error {worst_p[0]:.3e} m ({worst_p[1]}), "
          f"worst rotation error {worst_r[0]:.3e} rad ({worst_r[1]})")
    assert worst_p[0] <= POS_TOL and worst_r[0] <= ROT_TOL


def test_reference_fk_matches_the_description_package():
    """The reference FK here and the description package's urdf_utils agree, so J02 does not rest on one FK alone."""
    sys.path.insert(0, str(DESCRIPTION / "test"))
    import urdf_utils  # noqa: PLC0415

    root = urdf_root()
    rng = np.random.default_rng(7)
    for _ in range(10):
        q = random_q(root, rng)
        ours = urdf_fk(root, q)
        # urdf_utils does not apply <mimic>; give it the mimic joints' values explicitly.
        full = dict(q)
        for j in root.findall("joint"):
            m = j.find("mimic")
            if m is not None:
                full[j.get("name")] = float(m.get("multiplier", "1")) * q.get(m.get("joint"), 0.0) + float(m.get("offset", "0"))
        theirs = urdf_utils.zero_pose_poses(root, full)
        assert set(ours) == set(theirs)
        assert max(np.abs(ours[k] - theirs[k]).max() for k in ours) <= 1e-12


def test_negative_control_detects_a_1mm_lift_offset(model_for):
    """The check is not vacuous: MuJoCo 1 mm off the URDF on lift_joint must show up as a ~1e-3 m error."""
    model, _ = model_for(generate.Config())
    data = mujoco.MjData(model)
    root = urdf_root()
    set_qpos(model, data, root, {"lift_joint": 0.175 + 0.001})
    ref = urdf_fk(root, {"lift_joint": 0.175})
    link = "openarm_left_ee_base_link"  # the preset collapses link7 into it
    b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, link)
    assert b >= 0
    err = float(np.linalg.norm(data.xpos[b] - ref[link][:3, 3]))
    assert abs(err - 0.001) < 1e-12, err
    worst_error, _link = _worst_offset(model, data, root)
    assert worst_error > POS_TOL  # and the same comparison J02 runs flags it


def _worst_offset(model, data, root):
    """(worst position error, link) with MuJoCo's lift 1 mm off the URDF's: the parity comparison must flag it."""
    ref_q = {"lift_joint": 0.175}
    set_qpos(model, data, root, {"lift_joint": 0.176})
    ref = urdf_fk(root, ref_q)
    return max((float(np.linalg.norm(data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, k)] - t[:3, 3])), k)
               for k, t in ref.items())
