# SPDX-License-Identifier: MIT
"""Generate the OpenAMRobot 2.0 MuJoCo model (MJCF) from the installed combined_robot.xacro.

The robot is the URDF the description package expands, imported by MuJoCo's own URDF compiler and then given what a
URDF cannot say to MuJoCo: actuators, the finger mimic, cameras, the collision exclusions MoveIt uses, and a base
mode. Every number added here comes from the URDF, from the pinned OpenArm description (the real arms' control
gains and the description package's sim_dynamics profile), or from config/mujoco_twin.yaml, each with its source. Masses and
inertias are the URDF's and nothing else: `inertiafromgeom="false"` stops MuJoCo from giving the massless URDF links
a mass from their geometry.

The mesh-alias step (one file per mesh file and scale) follows OpenRAL's OpenAMRobot 2.0 twin
(github.com/OpenRAL/openral, python/hal/src/openral_hal/_openamrobot_v2_assets.py, Apache-2.0), as do the
mimic-as-joint-equality and the optical-frame camera rotation.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import re
import shutil
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import yaml

PKG = "openamrobot_upperbody_mujoco"
DESCRIPTION_PKG = "openamrobot_upperbody_description"
MOVEIT_PKG = "openamrobot_upperbody_moveit"
OPENARM_GAINS_REL = "assets/robot/openarm_v2.0/config/arm/control/control_gains.yaml"
SIM_DYNAMICS_REL = "config/sim_dynamics/openarm_v2.yaml"
ACTUATOR_LEDGER_REL = "config/actuator_ledger.yaml"
BASE_MODES = ("wheeled", "fixed")
SIDES = ("left", "right")
GRAVITY = 9.81
# A REP-103 optical frame looks along +z with +y down; a MuJoCo camera looks along -z with +y up: a half-turn about x.
OPTICAL_TO_MUJOCO_CAMERA_QUAT = (0.0, 1.0, 0.0, 0.0)


def share(pkg: str) -> Path:
    from ament_index_python.packages import get_package_share_directory  # noqa: PLC0415

    return Path(get_package_share_directory(pkg))


def config_dir() -> Path:
    """config/ of the source tree when run from it (the build step, before install), else the installed share/."""
    src = Path(__file__).resolve().parent.parent / "config"
    return src if (src / "mujoco_twin.yaml").is_file() else share(PKG) / "config"


def load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def ledger_peaks() -> dict[str, float]:
    """{urdf_joint: peak torque (N m) or peak force (N)} from the description's actuator ledger (#65), for every
    actuator the ledger maps to a URDF joint and gives a peak for. Unknown (null) peaks are left out."""
    peaks = {}
    for act in load_yaml(share(DESCRIPTION_PKG) / ACTUATOR_LEDGER_REL)["actuators"]:
        peak = act.get("peak_torque") or act.get("peak_force")
        if act.get("urdf_joint") and peak and peak.get("value") is not None:
            peaks[act["urdf_joint"]] = float(peak["value"])
    return peaks


@dataclass(frozen=True)
class Config:
    """One buildable configuration: option B's configuration IDs plus the MuJoCo-only choices."""

    base_plate: str = "bp000"
    head_pitch: str = "hp30"
    base_tilt: str = "bt10"
    column: str = "dold_v4_350"
    base_mode: str = "wheeled"
    arm_gravity_compensation: bool = False

    @property
    def name(self) -> str:
        gc = "_gc" if self.arm_gravity_compensation else ""
        return f"openamrobot_{self.column}_{self.base_plate}_{self.head_pitch}_{self.base_tilt}_{self.base_mode}{gc}"


# --------------------------------------------------------------------------------------------------------------------
# URDF


def expand_urdf(cfg: Config) -> str:
    """The installed combined_robot.xacro, expanded with the configuration's args (as Gazebo and MoveIt load it)."""
    import xacro  # noqa: PLC0415

    xacro_file = share(DESCRIPTION_PKG) / "urdf" / "combined_robot.xacro"
    mappings = {
        "column": cfg.column,
        "base_plate": cfg.base_plate,
        "head_pitch": cfg.head_pitch,
        "base_tilt": cfg.base_tilt,
        "arm_package": "openarm",
        # mock is the profile that gives lift_joint its DOLD effort and velocity (the real profile gives it 0 / 0).
        "hardware_profile": "mock",
        "enable_planar_base": "false",
        # The description's simulator joint dynamics (config/sim_dynamics/openarm_v2.yaml, #66): <dynamics damping
        # friction> on every OpenArm joint, which MuJoCo's URDF importer reads as damping and frictionloss.
        "sim_dynamics": "on",
    }
    return xacro.process_file(str(xacro_file), mappings=mappings).toxml()


def strip_for_mujoco(urdf_xml: str) -> ET.Element:
    """The URDF without <gazebo>, <ros2_control> and <transmission> (MuJoCo would reject or ignore them)."""
    root = ET.fromstring(urdf_xml)
    for tag in ("gazebo", "ros2_control", "transmission"):
        for el in root.findall(tag):
            root.remove(el)
    return root


def resolve_meshes(robot: ET.Element, mesh_dir: Path) -> dict[str, str]:
    """Copy every mesh into mesh_dir under a name unique per (file, scale); rewrite the URDF to those names.

    MuJoCo names a mesh asset after its file and ignores `scale` when it reuses one, so the left arm (the right arm's
    STLs with scale 1 -1 1) would otherwise be drawn and collided with the right arm's mirror image. COLLADA visuals
    become their collision STL twins: MuJoCo has no COLLADA reader.
    Returns {new file name: source path} for the record.
    """
    from ament_index_python.packages import get_package_share_directory  # noqa: PLC0415

    mesh_dir.mkdir(parents=True, exist_ok=True)
    seen: dict[tuple[str, str], str] = {}
    sources: dict[str, str] = {}
    for mesh in robot.iter("mesh"):
        uri = mesh.get("filename")
        m = re.fullmatch(r"package://([^/]+)/(.+)", uri or "")
        if not m:
            raise ValueError(f"mesh {uri!r}: only package:// URIs are expected in the combined URDF")
        rel = m.group(2)
        if rel.endswith(".dae"):
            rel = rel.replace("/visual/", "/collision/")[: -len(".dae")] + ".stl"
        src = Path(get_package_share_directory(m.group(1))) / rel
        if not src.is_file():
            raise FileNotFoundError(f"mesh {uri!r} -> {src}: not found")
        scale = mesh.get("scale", "1 1 1")
        key = (str(src), scale)
        if key not in seen:
            # Named by content and scale, so configurations generated into one directory never overwrite each other's.
            digest = hashlib.sha256(src.read_bytes() + scale.encode()).hexdigest()[:12]
            name = f"{src.stem}_{digest}{src.suffix.lower()}"
            shutil.copyfile(src, mesh_dir / name)
            seen[key] = name
            sources[name] = (m.group(1), rel, scale)
        mesh.set("filename", seen[key])
    return sources


# --------------------------------------------------------------------------------------------------------------------
# MuJoCo


def srdf_disabled_pairs() -> list[tuple[str, str]]:
    srdf = ET.parse(share(MOVEIT_PKG) / "config" / "openamrobot.srdf").getroot()
    return [(d.get("link1"), d.get("link2")) for d in srdf.findall("disable_collisions")]


def urdf_mimics(robot: ET.Element) -> list[tuple[str, str, float, float]]:
    out = []
    for j in robot.findall("joint"):
        mim = j.find("mimic")
        if mim is not None:
            out.append((j.get("name"), mim.get("joint"), float(mim.get("multiplier", "1")), float(mim.get("offset", "0"))))
    return out


def urdf_limits(robot: ET.Element) -> dict[str, dict[str, float]]:
    out = {}
    for j in robot.findall("joint"):
        lim = j.find("limit")
        if lim is not None:
            out[j.get("name")] = {k: float(v) for k, v in lim.attrib.items()}
    return out


def build_spec(cfg: Config, out_dir: Path):
    """Return (MjSpec, provenance dict) for one configuration; meshes are copied into out_dir/meshes."""
    import mujoco  # noqa: PLC0415

    twin = load_yaml(config_dir() / "mujoco_twin.yaml")
    # Armature has no URDF element: the profile says a MuJoCo model built from the URDF must set it from the file.
    sim_dynamics = load_yaml(share(DESCRIPTION_PKG) / SIM_DYNAMICS_REL)["joints"]
    gains = load_yaml(share("openarm_description") / OPENARM_GAINS_REL)

    robot = strip_for_mujoco(expand_urdf(cfg))
    mesh_dir = out_dir / "meshes"
    mesh_sources = resolve_meshes(robot, mesh_dir)
    limits = urdf_limits(robot)
    mimics = urdf_mimics(robot)

    compiler = ET.SubElement(robot, "mujoco")
    ET.SubElement(compiler, "compiler", {
        "meshdir": "meshes", "inertiafromgeom": "false", "fusestatic": "false", "discardvisual": "false",
        "balanceinertia": "false", "angle": "radian", "strippath": "false",
    })
    robot.insert(0, robot[-1])  # <mujoco> first, so the compiler options are read before the meshes
    del robot[-1]

    # MuJoCo resolves meshdir against the directory of the file it reads; from a string, against the cwd.
    cwd = os.getcwd()
    try:
        os.chdir(out_dir)
        urdf_text = ET.tostring(robot, encoding="unicode")
        # The URDF import itself, before any edit: fail here first if it is the URDF. On a throwaway spec: in MuJoCo
        # 3.9.0, compiling a spec and then adding a joint (the free base) makes later by-name joint edits land on the
        # previous joint (scratch repro: compile, add_freejoint, set j1..j3 damping [1, 2, 3] -> [2, 3, 0]).
        # J04 checks every joint's dynamics against its source, so a recurrence fails the tests.
        mujoco.MjSpec.from_string(urdf_text).compile()
        spec = mujoco.MjSpec.from_string(urdf_text)
    finally:
        os.chdir(cwd)
    spec.modelname = cfg.name
    spec.meshdir = str(mesh_dir)  # absolute while building; generate() writes it relative to the MJCF
    spec.compiler.inertiafromgeom = mujoco.mjtInertiaFromGeom.mjINERTIAFROMGEOM_FALSE

    solver = twin["solver"]["value"]
    spec.option.timestep = solver["timestep"]
    spec.option.integrator = getattr(mujoco.mjtIntegrator, f"mjINT_{solver['integrator'].upper()}")
    spec.option.cone = getattr(mujoco.mjtCone, f"mjCONE_{solver['cone'].upper()}")
    spec.option.impratio = solver["impratio"]
    spec.visual.global_.offwidth, spec.visual.global_.offheight = 1920, 1080

    # ---- collision: visual geoms never collide; collision geoms collide with everything (self-collision on) -------
    for geom in spec.geoms:
        if geom.contype or geom.conaffinity:
            geom.contype, geom.conaffinity, geom.group = 1, 1, 3
        else:
            geom.group = 1
    for a, b in srdf_disabled_pairs():
        spec.add_exclude(bodyname1=a, bodyname2=b)

    # ---- base -----------------------------------------------------------------------------------------------------
    base = spec.body("base_footprint")
    if cfg.base_mode == "wheeled":
        base.add_freejoint(name="base_free")
        wheel_friction = twin["wheel_friction"]["value"]
        for link in twin["drive_wheel_links"]["value"]:
            for g in spec.body(link).geoms:
                if g.contype:
                    g.friction = [wheel_friction, 0.005, 0.0001]
                    g.priority = 1  # out-rank the floor so this friction is the contact's
        for link in twin["caster_links"]["value"]:
            for g in spec.body(link).geoms:
                if g.contype:
                    g.condim, g.priority = 1, 1  # frictionless, and it decides the contact's condim
        kv = twin["wheel_velocity_gain"]["value"]
        # The URDF wheel joints are continuous with no effort limit; the hub motor's peak torque is in the ledger.
        peaks = ledger_peaks()
        for side in SIDES:
            jname = f"{side}_wheel_joint"
            if jname not in peaks:
                raise ValueError(f"{ACTUATOR_LEDGER_REL} gives no peak torque for {jname}")
            act = spec.add_actuator(name=jname, target=jname, trntype=mujoco.mjtTrn.mjTRN_JOINT)
            act.set_to_velocity(kv=kv)
            act.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE
            act.forcerange = [-peaks[jname], peaks[jname]]
    elif cfg.base_mode != "fixed":
        raise ValueError(f"base_mode {cfg.base_mode!r}: one of {BASE_MODES}")

    # ---- arms -----------------------------------------------------------------------------------------------------
    for side in SIDES:
        for i in range(1, 8):
            jname = f"openarm_{side}_joint{i}"
            _set_armature(spec, jname, sim_dynamics)
            g = gains[f"joint{i}"]
            act = spec.add_actuator(name=jname, target=jname, trntype=mujoco.mjtTrn.mjTRN_JOINT)
            act.set_to_position(kp=float(g["kp"]), kv=float(g["kd"]))
            _limit_actuator(act, limits[jname])
        for fname in (f"openarm_{side}_finger_joint1", f"openarm_{side}_finger_joint2"):
            _set_armature(spec, fname, sim_dynamics)
        fname = f"openarm_{side}_finger_joint1"
        fg = twin["finger_gains"]["value"]
        act = spec.add_actuator(name=fname, target=fname, trntype=mujoco.mjtTrn.mjTRN_JOINT)
        act.set_to_position(kp=fg["kp"], kv=fg["kv"])
        _limit_actuator(act, limits[fname])
    for follower, leader, mult, offset in mimics:
        eq = spec.add_equality(name=f"{follower}_mimic", type=mujoco.mjtEq.mjEQ_JOINT, name1=follower, name2=leader)
        # follower = offset + mult * leader: the first two polynomial coefficients. The rest stay zero; the array's
        # length differs between MuJoCo versions (11 in 3.9, 12 in 4.0), so it is not written out here.
        data = list(eq.data)
        data[0], data[1] = offset, mult
        eq.data = data

    if cfg.arm_gravity_compensation:
        # MuJoCo spreads a body's gravcomp over every DOF above it. On joints with actgravcomp it goes through the
        # actuator (and its force limit); on any other DOF it becomes a passive force. With a free base that would make
        # the arms weigh nothing on the floor (98 N on base and lift, review H1), which no motor can do. So: base fixed,
        # and the lift's share goes through the lift actuator, which carries the arms anyway (800 N limit).
        if cfg.base_mode != "fixed":
            raise ValueError("arm gravity compensation needs --base-mode fixed: on a free base MuJoCo would apply part "
                             "of it to the base as a passive force")
        for side in SIDES:
            for body in _subtree(spec.body(f"openarm_{side}_link1")):
                body.gravcomp = 1.0
            for j in spec.joints:
                if j.name.startswith(f"openarm_{side}_"):
                    j.actgravcomp = True
        spec.joint("lift_joint").actgravcomp = True

    # ---- lift: a speed-limited servo (the DOLD is a screw drive: it cannot step) ---------------------------------
    lift = limits["lift_joint"]
    compiled = spec.compile()
    carriage = mujoco.mj_name2id(compiled, mujoco.mjtObj.mjOBJ_BODY, "lift_carriage_link")
    load = float(compiled.body_subtreemass[carriage])
    # Rounded to 6 significant digits so that the MJCF writer (6 digits) stores them exactly.
    kp = float(f"{load * GRAVITY / twin['lift_static_error']['value']:.6g}")
    kv = float(f"{2.0 * math.sqrt(kp * load):.6g}")
    act = spec.add_actuator(name="lift_joint", target="lift_joint", trntype=mujoco.mjtTrn.mjTRN_JOINT)
    act.set_to_intvelocity(kp=kp, kv=kv)
    act.ctrllimited = mujoco.mjtLimited.mjLIMITED_TRUE
    act.ctrlrange = [-lift["velocity"], lift["velocity"]]
    act.actlimited = mujoco.mjtLimited.mjLIMITED_TRUE
    act.actrange = [lift["lower"], lift["upper"]]
    act.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE
    act.forcerange = [-lift["effort"], lift["effort"]]

    # ---- cameras on the URDF optical frames ----------------------------------------------------------------------
    for cam_name, cam in twin["cameras"].items():
        v = cam["value"]
        spec.body(v["frame"]).add_camera(name=cam_name, quat=list(OPTICAL_TO_MUJOCO_CAMERA_QUAT),
                                         fovy=v["fovy_deg"], resolution=v["resolution"])

    provenance = {
        "config": cfg.__dict__,
        "lift_servo": {"supported_mass_kg": load, "kp": kp, "kv": kv},
        "meshes": mesh_sources,
    }
    return spec, provenance


def _subtree(body):
    yield body
    for child in body.bodies:
        yield from _subtree(child)


def _set_armature(spec, name: str, sim_dynamics: dict) -> None:
    """Joint armature from the sim_dynamics profile; damping and friction already came in through the URDF."""
    if name not in sim_dynamics:
        raise KeyError(f"{name}: not in {SIM_DYNAMICS_REL}; every movable OpenArm joint needs an entry")
    spec.joint(name).armature = float(sim_dynamics[name]["armature"]["value"])


def _limit_actuator(act, lim: dict) -> None:
    import mujoco  # noqa: PLC0415

    act.ctrllimited = mujoco.mjtLimited.mjLIMITED_TRUE
    act.ctrlrange = [lim["lower"], lim["upper"]]
    act.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE
    act.forcerange = [-lim["effort"], lim["effort"]]


def lift_stops() -> dict[str, float]:
    """Lift calibration stops as lift_joint positions: {L1000: 0.0, L1175: 0.175, L1350: 0.35}."""
    stops = load_yaml(share(DESCRIPTION_PKG) / "config" / "upperbody_configuration.yaml")["lift_calibration_stops"]
    lowest = min(stops.values())
    return {name: round(shoulder - lowest, 9) for name, shoulder in sorted(stops.items())}


def keyframes_xml(model) -> str:
    """<keyframe> for a complete model: each lift stop with the arms at SRDF 'home' (zero) and everything else at
    qpos0. Written into each top-level file, computed on that file's own model, so no DOF is left undefined.
    (Keyframes in an included file do not cover the including file's extra DOFs.)"""
    import mujoco  # noqa: PLC0415

    lift_jnt = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "lift_joint")
    lift_act = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "lift_joint")
    lines = ["  <keyframe>"]
    for stop, lift in lift_stops().items():
        qpos = model.qpos0.copy()
        qpos[model.jnt_qposadr[lift_jnt]] = lift
        ctrl = [0.0] * model.nu
        for i in range(model.nu):
            # Position servos hold where the key puts their joint; the wheels' velocity servos and the lift's speed
            # input are zero.
            if model.actuator_dyntype[i] == mujoco.mjtDyn.mjDYN_NONE and model.actuator_biasprm[i, 1] != 0:
                ctrl[i] = float(qpos[model.jnt_qposadr[model.actuator_trnid[i, 0]]])
        act = [0.0] * model.na
        act[model.actuator_actadr[lift_act]] = lift  # the lift servo's target is its activation, not its ctrl (a speed)
        lines.append(f'    <key name="{stop}" qpos="{_num(qpos)}" ctrl="{_num(ctrl)}" act="{_num(act)}"/>')
    lines.append("  </keyframe>")
    return "\n".join(lines) + "\n"


def _num(values) -> str:
    return " ".join(repr(float(v)) for v in values)


def restore_precision(xml: str, spec) -> str:
    """Rewrite each body's pose and inertial from the spec at full double precision.

    MjSpec.to_xml() (MuJoCo 3.9) writes masses, quaternions and inertias with 6 significant digits, and writes each
    inertial as MuJoCo's principal-axis diagonalisation. Here the body quat/pos and the inertial pos/mass come from the
    spec, and the inertia is the URDF's own tensor (fullinertia), so the file carries the URDF numbers, not a rounding.
    """
    root = ET.fromstring(xml)
    for body in root.iter("body"):
        sb = spec.body(body.get("name"))
        body.set("pos", _num(sb.pos))
        body.set("quat", _num(sb.quat))
        geoms = body.findall("geom")
        if len(geoms) != len(sb.geoms):
            raise RuntimeError(f"body {sb.name}: {len(geoms)} geoms written, {len(sb.geoms)} in the spec")
        for xg, sg in zip(geoms, sb.geoms, strict=True):
            xg.set("pos", _num(sg.pos))
            xg.set("quat", _num(sg.quat))
        inertial = body.find("inertial")
        if sb.explicitinertial:
            if inertial is None:
                raise RuntimeError(f"body {sb.name}: explicit inertial in the spec but none written")
            inertial.attrib.clear()
            inertial.set("pos", _num(sb.ipos))
            inertial.set("mass", repr(float(sb.mass)))
            inertial.set("fullinertia", _num(sb.fullinertia))
        elif inertial is not None:
            raise RuntimeError(f"body {sb.name}: no inertial in the URDF but one was written")
    return ET.tostring(root, encoding="unicode")


# Model arrays the written file must reproduce from the compiled spec (relative tolerance REL_TOL).
ROUND_TRIP_FIELDS = ("body_mass", "body_pos", "body_quat", "body_ipos", "body_inertia", "jnt_pos", "jnt_axis",
                     "jnt_range", "geom_pos", "geom_quat", "geom_size", "cam_pos", "cam_quat", "actuator_gainprm",
                     "actuator_biasprm", "actuator_ctrlrange", "actuator_forcerange", "dof_damping", "dof_armature",
                     "dof_frictionloss", "eq_data", "geom_friction", "geom_priority", "geom_condim", "body_gravcomp",
                     "jnt_actgravcomp", "mesh_scale")
REL_TOL = 1e-12


def round_trip_error(a, b) -> dict[str, float]:
    """Largest |a - b| / max(|a|, 1) per field. Inertia and its frame: compared as the world-axis tensor."""
    import numpy as np  # noqa: PLC0415

    out = {}
    for f in ROUND_TRIP_FIELDS:
        x, y = (np.asarray(getattr(m, f), dtype=float) for m in (a, b))  # some fields are bool or int
        if f == "body_inertia":
            continue
        if f in ("body_quat",):
            # q and -q are the same rotation
            d = np.minimum(np.abs(x - y).max(axis=1), np.abs(x + y).max(axis=1))
            out[f] = float(d.max(initial=0.0))
            continue
        out[f] = float((np.abs(x - y) / np.maximum(np.abs(x), 1.0)).max(initial=0.0))
    out["body_inertia_tensor"] = float(max((np.abs(_tensor(a, i) - _tensor(b, i)).max() for i in range(a.nbody)),
                                           default=0.0))
    return out


def _tensor(m, i):
    import mujoco  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    r = np.zeros(9)
    mujoco.mju_quat2Mat(r, m.body_iquat[i])
    r = r.reshape(3, 3)
    return r @ np.diag(m.body_inertia[i]) @ r.T


MESH_LICENCES = {
    "openarm_description": ("Apache-2.0", "enactic/openarm_description; licence text: LICENSE-Apache-2.0.txt"),
    "openamrobot_description": ("CERN-OHL-P-2.0", "openAMRobot/openamr-platform-sw; CAD-derived robot geometry is "
                                "CERN-OHL-P-2.0 by that repository's LICENSING.md (https://spdx.org/licenses/CERN-OHL-P-2.0.html)"),
}


def licences_dir() -> Path:
    src = Path(__file__).resolve().parent.parent / "licenses"
    return src if (src / "Apache-2.0.txt").is_file() else share(PKG) / "licenses"


def write_notice(out_dir: Path, meshes: dict) -> None:
    """NOTICE.md beside the MJCF: where every copied mesh and the model data come from, and under which licence."""
    lines = ["# NOTICE (generated by openamrobot_upperbody_mujoco; do not edit)", "",
             "The MJCF files here are generated from the OpenAMRobot combined description (MIT; CAD-derived geometry",
             "CERN-OHL-P-2.0) and contain kinematics, inertias, limits and meshes of the packages below.", "",
             "| File | Source | Licence |", "|---|---|---|"]
    rows = set()
    notice = out_dir / "NOTICE.md"
    if notice.is_file():  # several configurations share one mesh directory: keep the rows of the others
        rows = {line for line in notice.read_text().splitlines() if line.startswith("| meshes/")}
    for name, (pkg, rel, scale) in meshes.items():
        if pkg not in MESH_LICENCES:
            raise KeyError(f"mesh {name} from {pkg}: no licence recorded in MESH_LICENCES")
        rows.add(f"| meshes/{name} | package://{pkg}/{rel} (scale {scale}) | {MESH_LICENCES[pkg][0]} |")
    lines += sorted(rows)
    lines += ["", "Sources:"] + [f"- {pkg}: {lic} ({where})" for pkg, (lic, where) in MESH_LICENCES.items()]
    notice.write_text("\n".join(lines) + "\n")
    shutil.copyfile(licences_dir() / "Apache-2.0.txt", out_dir / "LICENSE-Apache-2.0.txt")


def generate(cfg: Config, out_dir: Path, file_name: str | None = None) -> Path:
    """Write <out_dir>/<file_name or cfg.name>.xml and its meshes; return the MJCF path."""
    import mujoco  # noqa: PLC0415

    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    spec, provenance = build_spec(cfg, out_dir)
    compiled = spec.compile()
    xml = spec.to_xml()
    # The spec carries the absolute mesh dir (to_xml compiles); the file names it relative to itself. MuJoCo 3.9 writes
    # it with a trailing slash, 4.0 without.
    absolute = re.escape(str(out_dir / "meshes"))
    xml, n = re.subn(rf'meshdir="{absolute}/?"', 'meshdir="meshes/"', xml)
    if n != 1:
        raise RuntimeError(f"expected one meshdir naming {out_dir / 'meshes'} in the MJCF, found {n}")
    # MjSpec.to_xml() in MuJoCo 3.9 does not write inertiafromgeom (4.0 does), and the MJCF default ("auto") gives every
    # URDF link without an <inertial> a mass from its geometry (+6.9 kg at platform 131e500, +3.4 kg at 8e7a454).
    # Make sure it is "false".
    compilers = re.findall(r"<compiler\b[^>]*>", xml)
    if len(compilers) != 1:
        raise RuntimeError(f"expected one <compiler> element in the MJCF, found {len(compilers)}")
    found = re.findall(r'inertiafromgeom="([^"]*)"', compilers[0])
    if not found:
        xml = xml.replace("<compiler ", '<compiler inertiafromgeom="false" ', 1)
    elif found != ["false"]:
        raise RuntimeError(f"<compiler> says inertiafromgeom={found}, expected false")
    xml = restore_precision(xml, spec)
    path = out_dir / f"{file_name or cfg.name}.xml"
    write_notice(out_dir, provenance["meshes"])
    header = (f"<!-- SPDX-License-Identifier: MIT AND Apache-2.0 AND CERN-OHL-P-2.0. Generated by {PKG} from "
              "combined_robot.xacro; contains material from the sources in NOTICE.md next to this file "
              f"({', '.join(f'{k}={v}' for k, v in cfg.__dict__.items())}), MuJoCo {mujoco.__version__}. "
              "Do not edit: regenerate. Lift servo: supported mass "
              f"{provenance['lift_servo']['supported_mass_kg']:.4f} kg, kp {provenance['lift_servo']['kp']:.1f}, "
              f"kv {provenance['lift_servo']['kv']:.1f}. -->\n")
    path.write_text(header + xml)
    # Load it back from disk: the file, not the spec, is what users get. It must be the compiled spec, field by field.
    reloaded = mujoco.MjModel.from_xml_path(str(path))
    sizes = ("nbody", "njnt", "ngeom", "nu", "na", "neq", "nexclude", "ncam")
    if any(getattr(reloaded, n) != getattr(compiled, n) for n in sizes):
        raise RuntimeError(f"{path}: sizes differ on reload: {[(n, getattr(compiled, n), getattr(reloaded, n)) for n in sizes]}")
    err = round_trip_error(compiled, reloaded)
    bad = {k: v for k, v in err.items() if v > REL_TOL}
    if bad:
        raise RuntimeError(f"{path}: the written file does not reproduce the compiled spec: {bad}")
    return path


SCENE_HEAD = """<!-- SPDX-License-Identifier: MIT. Generated by {pkg}: {what} around {robot}. Do not edit: regenerate. -->
<mujoco model="{name}">
  <include file="{robot}"/>
  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
    <global azimuth="150" elevation="-20"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
    <texture type="2d" name="floor_grid" builtin="checker" mark="edge" rgb1="0.42 0.44 0.47" rgb2="0.55 0.57 0.6"
             markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="floor_grid" texture="floor_grid" texuniform="true" texrepeat="5 5" reflectance="0.1"/>
  </asset>
  <worldbody>
    <light pos="0 0 3.5" dir="0 0 -1" directional="true"/>
    <geom name="floor" type="plane" size="0 0 0.05" material="floor_grid"/>
"""

TABLE_AND_CUBES = """    <!-- A table in front of the arms (top 0.75 m, front edge 0.45 m ahead of base_footprint, 35 mm clear of the
         chassis: drive the base in floor_*.xml, not here) and two 40 mm cubes at x 0.68 m, where the head camera
         (hp30, fovy 52) sees them at L1000; at x 0.55 m they were below its field of view (J07 prints this per stop). The robot's compiler sets
         inertiafromgeom="false" for the whole model, so every moving body here states its inertial: a 0.05 kg cube
         of side 0.04 m has I = m s^2 / 6 = 1.33333e-05 kg m^2 about each axis. -->
    <body name="table" pos="0.75 0 0.375">
      <geom name="table_top" type="box" size="0.30 0.40 0.375" rgba="0.62 0.5 0.38 1"/>
    </body>
    <body name="cube_red" pos="0.68 0.12 0.77">
      <freejoint/>
      <inertial pos="0 0 0" mass="0.05" diaginertia="1.33333e-05 1.33333e-05 1.33333e-05"/>
      <geom type="box" size="0.02 0.02 0.02" rgba="0.8 0.2 0.2 1"/>
    </body>
    <body name="cube_blue" pos="0.68 -0.12 0.77">
      <freejoint/>
      <inertial pos="0 0 0" mass="0.05" diaginertia="1.33333e-05 1.33333e-05 1.33333e-05"/>
      <geom type="box" size="0.02 0.02 0.02" rgba="0.2 0.3 0.8 1"/>
    </body>
"""


def write_scene(robot_path: Path, kind: str = "scene") -> Path:
    """A top-level file next to the robot MJCF that includes it (so it always loads the current robot):
    kind "scene" = floor, table and two cubes (scene_<robot>.xml); kind "floor" = an empty floor (floor_<robot>.xml)."""
    import mujoco  # noqa: PLC0415

    if kind not in ("scene", "floor"):
        raise ValueError(f"kind {kind!r}: scene or floor")
    path = robot_path.with_name(f"{kind}_{robot_path.name}")
    what = "a basic scene (floor, table, two cubes)" if kind == "scene" else "an empty floor"
    text = SCENE_HEAD.format(pkg=PKG, what=what, robot=robot_path.name, name=f"{kind}_{robot_path.stem}")
    text += (TABLE_AND_CUBES if kind == "scene" else "") + "  </worldbody>\n</mujoco>\n"
    path.write_text(text)
    keys = keyframes_xml(mujoco.MjModel.from_xml_path(str(path)))
    path.write_text(text.replace("</mujoco>", keys + "</mujoco>"))
    model = mujoco.MjModel.from_xml_path(str(path))
    if model.nkey != len(lift_stops()):
        raise RuntimeError(f"{path}: {model.nkey} keyframes, expected {len(lift_stops())}")
    return path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--file-name", default=None, help="MJCF file stem (default: the configuration name)")
    p.add_argument("--base-plate", default=Config.base_plate)
    p.add_argument("--head-pitch", default=Config.head_pitch)
    p.add_argument("--base-tilt", default=Config.base_tilt)
    p.add_argument("--column", default=Config.column)
    p.add_argument("--base-mode", default=Config.base_mode, choices=BASE_MODES)
    p.add_argument("--arm-gravity-compensation", action="store_true")
    a = p.parse_args(argv)
    cfg = Config(a.base_plate, a.head_pitch, a.base_tilt, a.column, a.base_mode, a.arm_gravity_compensation)
    robot = generate(cfg, a.out_dir, a.file_name)
    print(robot)
    print(write_scene(robot, "scene"))
    print(write_scene(robot, "floor"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
