# SPDX-License-Identifier: MIT
"""J04: every limit, gain and joint dynamic in the MJCF is the number its source gives.

Sources: the URDF <limit> and <mimic>; the pinned openarm_description control_gains.yaml (the gains the real ros2_control
profile sends to the motors); the description's config/sim_dynamics/openarm_v2.yaml (simulator joint dynamics, #66);
config/mujoco_twin.yaml. Exact equality: these numbers are copied, not computed.
"""


import pytest
import yaml
from conftest import generate, mujoco, urdf_root
from ament_index_python.packages import get_package_share_directory

SIDES = ("left", "right")


@pytest.fixture(scope="module")
def model(model_for):
    return model_for(generate.Config())[0]


def _limits():
    return {j.get("name"): {k: float(v) for k, v in j.find("limit").attrib.items()}
            for j in urdf_root().findall("joint") if j.find("limit") is not None}


def _act(model, name):
    i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
    assert i >= 0, f"no actuator {name}"
    return i


def _jnt(model, name):
    i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    assert i >= 0, f"no joint {name}"
    return i


def test_joint_ranges(model):
    for name, lim in _limits().items():
        if "lower" not in lim:
            continue
        j = _jnt(model, name)
        assert model.jnt_limited[j], name
        assert list(model.jnt_range[j]) == [lim["lower"], lim["upper"]], name
    for side in SIDES:  # continuous drive wheels stay unlimited
        assert not model.jnt_limited[_jnt(model, f"{side}_wheel_joint")]


def test_arm_actuators_use_the_real_gains_and_urdf_limits(model):
    gains = yaml.safe_load(open(f"{get_package_share_directory('openarm_description')}/{generate.OPENARM_GAINS_REL}"))
    lim = _limits()
    for side in SIDES:
        for k in range(1, 8):
            name = f"openarm_{side}_joint{k}"
            a = _act(model, name)
            assert model.actuator_gainprm[a, 0] == gains[f"joint{k}"]["kp"], name
            assert model.actuator_biasprm[a, 1] == -gains[f"joint{k}"]["kp"], name
            assert model.actuator_biasprm[a, 2] == -gains[f"joint{k}"]["kd"], name
            assert list(model.actuator_ctrlrange[a]) == [lim[name]["lower"], lim[name]["upper"]], name
            assert model.actuator_forcelimited[a] and list(model.actuator_forcerange[a]) == [-lim[name]["effort"], lim[name]["effort"]], name


def test_finger_actuators_and_mimic(model):
    twin = yaml.safe_load(open(generate.config_dir() / "mujoco_twin.yaml"))["finger_gains"]["value"]
    lim = _limits()
    for side in SIDES:
        lead, follow = f"openarm_{side}_finger_joint1", f"openarm_{side}_finger_joint2"
        a = _act(model, lead)
        assert model.actuator_gainprm[a, 0] == twin["kp"] and model.actuator_biasprm[a, 2] == -twin["kv"]
        assert list(model.actuator_forcerange[a]) == [-lim[lead]["effort"], lim[lead]["effort"]]
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, follow) == -1, "the mimic joint has no actuator"
        e = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, f"{follow}_mimic")
        assert e >= 0
        mimic = next(j.find("mimic") for j in urdf_root().findall("joint") if j.get("name") == follow)
        assert model.eq_obj1id[e] == _jnt(model, follow) and model.eq_obj2id[e] == _jnt(model, lead)
        assert list(model.eq_data[e, :2]) == [float(mimic.get("offset", "0")), float(mimic.get("multiplier", "1"))]


def test_lift_servo(model):
    lim = _limits()["lift_joint"]
    a = _act(model, "lift_joint")
    assert model.actuator_dyntype[a] == mujoco.mjtDyn.mjDYN_INTEGRATOR, "intvelocity: ctrl is a speed"
    assert list(model.actuator_ctrlrange[a]) == [-lim["velocity"], lim["velocity"]]
    assert model.actuator_actlimited[a] and list(model.actuator_actrange[a]) == [lim["lower"], lim["upper"]]
    assert model.actuator_forcelimited[a] and list(model.actuator_forcerange[a]) == [-lim["effort"], lim["effort"]]


def test_joint_dynamics_are_the_sim_dynamics_profile(model):
    """Damping and friction (through the URDF's <dynamics>) and armature (set by the generator) of every OpenArm joint =
    the description's sim_dynamics profile, the file Gazebo reads too. Exact: copied numbers. Every profile entry is
    checked, so a joint the generator misses (or a by-name edit that lands on the wrong joint) fails here."""
    profile = yaml.safe_load(open(f"{get_package_share_directory('openamrobot_upperbody_description')}/{generate.SIM_DYNAMICS_REL}"))
    joints = profile["joints"]
    assert len(joints) == 18
    for name, entry in joints.items():
        dof = model.jnt_dofadr[_jnt(model, name)]
        want = (entry["damping"]["value"], entry["friction"]["value"], entry["armature"]["value"])
        got = (model.dof_damping[dof], model.dof_frictionloss[dof], model.dof_armature[dof])
        assert got == want, f"{name}: (damping, frictionloss, armature) {got} != profile {want}"


def test_solver_options(model):
    s = yaml.safe_load(open(generate.config_dir() / "mujoco_twin.yaml"))["solver"]["value"]
    assert model.opt.timestep == s["timestep"] and model.opt.impratio == s["impratio"]
    assert model.opt.integrator == mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    assert model.opt.cone == mujoco.mjtCone.mjCONE_ELLIPTIC


def test_gravity_compensation_variant(model_for):
    off, _ = model_for(generate.Config())
    on, _ = model_for(generate.Config(arm_gravity_compensation=True))
    arm_bodies = [b for b in range(on.nbody) if (mujoco.mj_id2name(on, mujoco.mjtObj.mjOBJ_BODY, b) or "").startswith(("openarm_left_link", "openarm_right_link"))]
    assert arm_bodies and all(on.body_gravcomp[b] == 1.0 for b in arm_bodies)
    assert not off.body_gravcomp.any() and not off.jnt_actgravcomp.any()
    for side in SIDES:
        assert on.jnt_actgravcomp[_jnt(on, f"openarm_{side}_joint1")]


def test_moveit_joint_limits_add_no_position_or_velocity_limit():
    """Issue #58 names MoveIt's joint_limits.yaml. It adds acceleration limits only (status placeholder); position and
    velocity come from the URDF, which is what the twin uses. If it ever overrides one, the twin must follow: fail here.
    MuJoCo has no joint acceleration limit; the lift reaches 0.018 m/s at once where MoveIt plans 0.01 m/s^2 (known gap)."""
    from conftest import MOVEIT  # noqa: PLC0415

    limits = yaml.safe_load(open(MOVEIT / "config" / "joint_limits.yaml"))["joint_limits"]
    overrides = {j: k for j, v in limits.items() for k in v
                 if k in ("has_position_limits", "min_position", "max_position", "has_velocity_limits", "max_velocity")}
    assert not overrides, f"joint_limits.yaml overrides URDF position/velocity limits: {overrides}"
    assert set(limits) <= {j.get("name") for j in urdf_root().findall("joint")}
