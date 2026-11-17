# SPDX-License-Identifier: MIT
"""J06: the physics does what the data says. Every bound below was written before the first run, from the number it
checks, and is stated next to it.

* Lift: never faster than lift_joint's URDF velocity (0.018 m/s, +1 % for integration), so the 0.35 m stroke takes at
  least 19.4 s. The sag at L1350 is by design (the servo stiffness is sized from the same load for 0.5 mm), so that test
  only shows the design holds in the scene and the 800 N limit is not reached; the DOLD datasheet gives no stiffness.
* Arms under gravity, SRDF hands_up at L1175: the joint's equation of motion near rest,
  kp e - (kd + d) qdot + f_friction = tau_g + (M qddot)_i, bounds the servo error e:
  |kp e - tau_g| <= frictionloss + (kd + d) |qdot| + |(M qddot)_i| (+1e-9), with qdot and qddot as measured. With gravity
  compensation the actuators feed tau_g forward, and the same bound holds with tau_g = 0.
* Wheeled base (on the empty floor_*.xml): equal wheel speeds w = 1 rad/s drive straight at w R = 0.1 m/s (+-5 %);
  opposite speeds turn the base the right way; the base stays within 2 degrees of level. Turning at the thin-wheel rate
  2 w R / track = 0.5 rad/s (+-10 %) is a KNOWN GAP (strict xfail below, with the measurement).
"""

import numpy as np
import pytest
import yaml
from conftest import generate, mujoco, urdf_root

SIDES = ("left", "right")


def _reset(model, key):
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, key))
    mujoco.mj_forward(model, data)
    return data


def _id(model, kind, name):
    i = mujoco.mj_name2id(model, kind, name)
    assert i >= 0, f"no {kind} {name}"
    return i


def _limits(name):
    return {k: float(v) for k, v in next(j for j in urdf_root().findall("joint") if j.get("name") == name).find("limit").attrib.items()}


def test_lift_speed_and_stroke(scene_for):
    model = scene_for(generate.Config(base_mode="fixed"))
    data = _reset(model, "L1000")
    lim = _limits("lift_joint")
    act, jnt = _id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "lift_joint"), _id(model, mujoco.mjtObj.mjOBJ_JOINT, "lift_joint")
    qa, va = model.jnt_qposadr[jnt], model.jnt_dofadr[jnt]
    data.ctrl[act] = 10.0  # far beyond the range: the actuator clamps it to the URDF velocity
    vmax, t_top = 0.0, None
    for k in range(int(25.0 / model.opt.timestep)):
        mujoco.mj_step(model, data)
        vmax = max(vmax, float(data.qvel[va]))
        if t_top is None and data.qpos[qa] >= lim["upper"] - 1e-3:
            t_top = (k + 1) * model.opt.timestep
    print(f"lift: peak speed {vmax:.5f} m/s (limit {lim['velocity']}), 0 -> {lim['upper'] - 1e-3:.3f} m in {t_top} s, "
          f"final {data.qpos[qa]:.5f} m")
    assert data.ctrl[act] == 10.0 and vmax <= lim["velocity"] * 1.01
    assert t_top is not None and t_top >= (lim["upper"] - 1e-3) / lim["velocity"]
    assert data.act[model.actuator_actadr[act]] <= lim["upper"]  # the target never leaves the joint range


def test_lift_holds_the_arms_at_l1350(scene_for):
    model = scene_for(generate.Config(base_mode="fixed"))
    data = _reset(model, "L1350")
    for _ in range(int(3.0 / model.opt.timestep)):
        mujoco.mj_step(model, data)
    twin = yaml.safe_load(open(generate.config_dir() / "mujoco_twin.yaml"))
    designed = twin["lift_static_error"]["value"]
    sag = 0.35 - data.qpos[model.jnt_qposadr[_id(model, mujoco.mjtObj.mjOBJ_JOINT, "lift_joint")]]
    force = abs(float(data.actuator_force[_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "lift_joint")]))
    effort = _limits("lift_joint")["effort"]
    print(f"lift sag at L1350: {sag * 1e3:.4f} mm (designed {designed * 1e3} mm), actuator force {force:.1f} N (limit {effort})")
    assert 0.0 <= sag <= designed * 1.10 and force < effort


@pytest.mark.parametrize("gravcomp", [False, True], ids=["no_gravcomp", "gravcomp"])
def test_arms_hold_hands_up_under_gravity(gravcomp, scene_for):
    model = scene_for(generate.Config(base_mode="fixed", arm_gravity_compensation=gravcomp))
    data = _reset(model, "L1175")
    target = {f"openarm_{s}_joint4": 1.5 for s in SIDES}  # SRDF hands_up: joint4 1.5, the rest 0
    for name, v in target.items():
        data.qpos[model.jnt_qposadr[_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)]] = v
        data.ctrl[_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)] = v
    for _ in range(int(5.0 / model.opt.timestep)):
        mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)
    m_qacc = np.zeros(model.nv)
    mujoco.mj_mulM(model, data, m_qacc, data.qacc)
    rows = []
    for s in SIDES:
        for k in range(1, 8):
            name = f"openarm_{s}_joint{k}"
            a, j = _id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name), _id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            dof = model.jnt_dofadr[j]
            kp = model.actuator_gainprm[a, 0]
            e = data.ctrl[a] - data.qpos[model.jnt_qposadr[j]]
            tau_g = 0.0 if gravcomp else data.qfrc_bias[dof]
            residual = abs(kp * e - tau_g)
            kd = -model.actuator_biasprm[a, 2]
            bound = (model.dof_frictionloss[dof] + (kd + model.dof_damping[dof]) * abs(data.qvel[dof])
                     + abs(m_qacc[dof]) + 1e-9)
            rows.append((residual / bound, name, e, tau_g, residual, bound))
            assert abs(data.qvel[dof]) < 1e-3, f"{name} not at rest: {data.qvel[dof]}"
    rows.sort(reverse=True)
    print(f"hands_up @ L1175, gravcomp={gravcomp}: largest error {max(abs(r[2]) for r in rows):.4f} rad; "
          f"worst |kp e - tau_g| {rows[0][4]:.4f} N m on {rows[0][1]} (bound {rows[0][5]:.3f})")
    for _ratio, name, e, tau_g, residual, bound in rows:
        assert residual <= bound, f"{name}: e {e:.5f} rad, tau_g {tau_g:.4f}, |kp e - tau_g| {residual:.4f} > {bound:.4f}"


def _drive(model, w_left, w_right, seconds):
    data = _reset(model, "L1000")
    data.ctrl[_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "left_wheel_joint")] = w_left
    data.ctrl[_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "right_wheel_joint")] = w_right
    base = _id(model, mujoco.mjtObj.mjOBJ_BODY, "base_footprint")
    samples, max_tilt = [], 0.0
    for k in range(int(seconds / model.opt.timestep)):
        mujoco.mj_step(model, data)
        max_tilt = max(max_tilt, float(np.degrees(np.arccos(np.clip(data.xmat[base][8], -1, 1)))))
        samples.append((k * model.opt.timestep, data.xpos[base].copy(), data.xmat[base].reshape(3, 3).copy()))
    return samples, max_tilt


def _wheel_geometry():
    root = urdf_root()
    wheel = next(l for l in root.findall("link") if l.get("name") == "left_wheel_link")
    radius = float(wheel.find("collision/geometry/cylinder").get("radius"))
    ys = [float(j.find("origin").get("xyz").split()[1]) for j in root.findall("joint") if j.get("name") in ("left_wheel_joint", "right_wheel_joint")]
    return radius, abs(ys[0] - ys[1])


def test_wheeled_base_drives_straight(scene_for):
    model = scene_for(generate.Config(), "floor")
    radius, _ = _wheel_geometry()
    samples, tilt = _drive(model, 1.0, 1.0, 4.0)
    (t0, p0, _), (t1, p1, r1) = samples[int(2.0 / model.opt.timestep)], samples[-1]
    v = (p1[0] - p0[0]) / (t1 - t0)
    yaw = np.arctan2(r1[1, 0], r1[0, 0])
    print(f"straight: v {v:.4f} m/s (expected {radius:.3f}), lateral {p1[1] * 1e3:.2f} mm, yaw {yaw:.5f} rad, max tilt {tilt:.3f} deg")
    assert abs(v - radius) <= 0.05 * radius and abs(p1[1]) < 5e-3 and abs(yaw) < 0.01 and tilt < 2.0


def _spin(scene_for):
    model = scene_for(generate.Config(), "floor")
    radius, track = _wheel_geometry()
    samples, tilt = _drive(model, -1.0, 1.0, 4.0)
    (t0, _, r0), (t1, p1, r1) = samples[int(2.0 / model.opt.timestep)], samples[-1]
    dyaw = np.arctan2(r1[1, 0], r1[0, 0]) - np.arctan2(r0[1, 0], r0[0, 0])
    rate = ((dyaw + np.pi) % (2 * np.pi) - np.pi) / (t1 - t0)
    expected = 2.0 * radius / track
    print(f"spin: yaw rate {rate:.4f} rad/s (point-contact kinematics {expected:.3f}, track {track} m), "
          f"drift {np.linalg.norm(p1[:2]) * 1e3:.1f} mm, max tilt {tilt:.3f} deg")
    return rate, expected, tilt


def test_wheeled_base_turns_on_the_spot(scene_for):
    """Opposite wheel speeds turn the base the right way (left wheel back, right forward: counter-clockwise), level."""
    rate, _, tilt = _spin(scene_for)
    assert rate > 0.0 and tilt < 2.0


@pytest.mark.xfail(strict=True, reason=(
    "KNOWN GAP, measured 2026-11-17 on the 87.1 kg model: the URDF's drive wheel is a rigid 46 mm wide cylinder, and "
    "MuJoCo contacts it at both rim edges, which scrub at full friction when turning on the spot (each |F_t| = 1.5 x "
    "normal, e.g. 331 N on 220.7 N). With the wheel servo at its 16 N m peak torque (actuator ledger, #65) the base turns "
    "at 0.32 rad/s against 0.50 for a thin wheel, so it is the contact model, not the motor, that limits it. Gazebo "
    "(DART) on the same wheels at +-1 rad/s: 0.48 rad/s, but 0.081 m/s straight where MuJoCo gives 0.0999 "
    "(review/gazebo_spin/RESULTS.md in the work notes). The wheel-ground model is for the maintainers to decide. "
    "strict: if a model change makes this pass, remove the mark."))
def test_spin_rate_matches_thin_wheel_kinematics(scene_for):
    rate, expected, _ = _spin(scene_for)
    assert abs(rate - expected) <= 0.10 * expected
