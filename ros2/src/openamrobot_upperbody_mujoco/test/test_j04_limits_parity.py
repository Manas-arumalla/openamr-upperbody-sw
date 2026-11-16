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


def test_gravity_compensation_goes_through_actuators_only(model_for):
    """With --arm-gravity-compensation every arm body is compensated, and ALL of it reaches the joints through actuators
    (arm joints and the lift), inside their force limits: no DOF gets any of it as a passive force (review H1: before,
    98 N went to the base and the lift). Without the option there is none at all."""
    off, _ = model_for(generate.Config(base_mode="fixed"))
    on, _ = model_for(generate.Config(base_mode="fixed", arm_gravity_compensation=True))
    name = lambda m, b: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or ""  # noqa: E731
    arm_bodies = [b for b in range(on.nbody) if name(on, b).startswith(("openarm_left_link", "openarm_right_link"))]
    assert arm_bodies and all(on.body_gravcomp[b] == 1.0 for b in arm_bodies)
    assert not off.body_gravcomp.any() and not off.jnt_actgravcomp.any()
    assert on.jnt_actgravcomp[_jnt(on, "lift_joint")]
    data = mujoco.MjData(on)
    mujoco.mj_forward(on, data)
    routed = [j for j in range(on.njnt) if on.jnt_actgravcomp[j]]
    passive = {mujoco.mj_id2name(on, mujoco.mjtObj.mjOBJ_JOINT, j): float(data.qfrc_gravcomp[on.jnt_dofadr[j]])
               for j in range(on.njnt) if j not in routed and abs(data.qfrc_gravcomp[on.jnt_dofadr[j]]) > 0}
    assert not passive, f"gravity compensation reaching joints as a passive force: {passive}"
    lift = on.jnt_dofadr[_jnt(on, "lift_joint")]
    assert data.qfrc_gravcomp[lift] > 0 and data.qfrc_passive[lift] == 0.0  # through the lift actuator, not passive
    print(f"lift carries {data.qfrc_gravcomp[lift]:.2f} N of arm weight through its actuator (limit 800 N)")


def test_contact_parameters(model_for):
    """Wheel friction, frictionless casters and the priorities that make them the contact's (mujoco_twin.yaml)."""
    twin = yaml.safe_load(open(generate.config_dir() / "mujoco_twin.yaml"))
    model, _ = model_for(generate.Config())
    def geoms(link):
        b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, link)
        assert b >= 0, link
        return [g for g in range(model.ngeom) if model.geom_bodyid[g] == b and model.geom_contype[g]]
    for link in twin["drive_wheel_links"]["value"]:
        assert geoms(link) and all(model.geom_friction[g][0] == twin["wheel_friction"]["value"] and model.geom_priority[g] == 1
                                   for g in geoms(link)), link
    for link in twin["caster_links"]["value"]:
        assert geoms(link) and all(model.geom_condim[g] == 1 and model.geom_priority[g] == 1 for g in geoms(link)), link


def test_left_arm_meshes_are_mirrored_not_shared(model_for):
    """The left arm draws from the right arm's STLs with scale 1 -1 1. MuJoCo reuses a mesh asset by file name and
    ignores scale on reuse, so without one asset per (file, scale) the left arm would get the right arm's geometry.
    Every mesh geom pair (left, right) must use different assets, and the left one must carry the URDF's scale."""
    model, _ = model_for(generate.Config())
    scale = {}
    for link in urdf_root().findall("link"):
        for el in link.findall("collision") + link.findall("visual"):
            mesh = el.find("geometry/mesh")
            if mesh is not None:
                scale.setdefault(link.get("name"), set()).add(mesh.get("scale", "1 1 1"))
    pairs = 0
    for g in range(model.ngeom):
        if model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        left = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[g])
        if not left.startswith("openarm_left_") or "1 -1 1" not in scale.get(left, set()):
            continue
        right = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, left.replace("_left_", "_right_"))
        rgeoms = [h for h in range(model.ngeom) if model.geom_bodyid[h] == right and model.geom_type[h] == mujoco.mjtGeom.mjGEOM_MESH]
        assert rgeoms, left
        assert model.geom_dataid[g] not in {model.geom_dataid[h] for h in rgeoms}, f"{left} shares a mesh asset with the right arm"
        assert list(model.mesh_scale[model.geom_dataid[g]]) == [1.0, -1.0, 1.0], left
        pairs += 1
    assert pairs > 0, "no mirrored left-arm mesh found: the check checked nothing"
    print(f"{pairs} mirrored left-arm mesh geoms, each on its own asset")


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
