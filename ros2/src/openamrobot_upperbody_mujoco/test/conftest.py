# SPDX-License-Identifier: MIT
"""Shared fixtures for the MuJoCo twin tests (J01-J07).

The reference everything is compared with is the URDF, expanded here with xacro directly (not through the generator
under test), from the INSTALLED description package, with the same arguments the generator uses.
"""

import functools
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest
import xacro
import yaml
from ament_index_python.packages import get_package_share_directory

mujoco = pytest.importorskip("mujoco", reason="the MuJoCo Python package (rosdep key python3-mujoco-pip) is not installed")

from openamrobot_upperbody_mujoco import generate  # noqa: E402

DESCRIPTION = Path(get_package_share_directory("openamrobot_upperbody_description"))
MOVEIT = Path(get_package_share_directory("openamrobot_upperbody_moveit"))
MUJOCO_SHARE = Path(get_package_share_directory("openamrobot_upperbody_mujoco"))
UPPERBODY = yaml.safe_load(open(DESCRIPTION / "config" / "upperbody_configuration.yaml"))
AXES = UPPERBODY["configuration_axes"]
BASELINE = UPPERBODY["baseline"]
STOPS = generate.lift_stops()  # {L1000: 0.0, L1175: 0.175, L1350: 0.35}


@functools.lru_cache(maxsize=None)
def urdf_root(base_plate="bp000", head_pitch="hp30", base_tilt="bt10", column="dold_v4_350"):
    """The expanded URDF (ElementTree root), as Gazebo and MoveIt get it with hardware_profile mock."""
    doc = xacro.process_file(str(DESCRIPTION / "urdf" / "combined_robot.xacro"), mappings={
        "column": column, "base_plate": base_plate, "head_pitch": head_pitch, "base_tilt": base_tilt,
        "arm_package": "openarm", "hardware_profile": "mock", "enable_planar_base": "false"})
    return ET.fromstring(doc.toxml())


@pytest.fixture(scope="session")
def build_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("mjcf")


@pytest.fixture(scope="session")
def model_for(build_dir):
    """model_for(Config) -> (MjModel loaded from the written file, its path). Each configuration is generated once."""
    cache = {}

    def get(cfg):
        if cfg not in cache:
            path = generate.generate(cfg, build_dir / cfg.name)
            cache[cfg] = (mujoco.MjModel.from_xml_path(str(path)), path)
        return cache[cfg]

    return get


@pytest.fixture(scope="session")
def scene_for(model_for):
    """scene_for(Config, kind="scene"|"floor") -> MjModel of that generated top-level file around the configuration."""
    cache = {}

    def get(cfg, kind="scene"):
        if (cfg, kind) not in cache:
            _, robot = model_for(cfg)
            cache[(cfg, kind)] = mujoco.MjModel.from_xml_path(str(generate.write_scene(robot, kind)))
        return cache[(cfg, kind)]

    return get


# ---- URDF forward kinematics, written here (independent of MuJoCo and of the generator) ---------------------------

def _rpy(r, p, y):
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return (np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]]) @ np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
            @ np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]]))


def _origin(el):
    t = np.eye(4)
    o = el.find("origin") if el is not None else None
    if o is not None:
        t[:3, :3] = _rpy(*[float(v) for v in o.get("rpy", "0 0 0").split()])
        t[:3, 3] = [float(v) for v in o.get("xyz", "0 0 0").split()]
    return t


def _motion(joint, q):
    t = np.eye(4)
    kind = joint.get("type")
    if kind == "fixed":
        return t
    ax = joint.find("axis")
    a = np.array([float(v) for v in (ax.get("xyz") if ax is not None else "1 0 0").split()])
    a /= np.linalg.norm(a)
    if kind == "prismatic":
        t[:3, 3] = a * q
    else:  # revolute, continuous: Rodrigues
        k = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
        t[:3, :3] = np.eye(3) + np.sin(q) * k + (1 - np.cos(q)) * k @ k
    return t


def urdf_fk(root, q):
    """{link: 4x4 pose in the URDF root frame} with joint positions q ({joint: value}; absent = 0; mimics applied)."""
    q = dict(q)
    for j in root.findall("joint"):
        m = j.find("mimic")
        if m is not None:
            q[j.get("name")] = float(m.get("multiplier", "1")) * q.get(m.get("joint"), 0.0) + float(m.get("offset", "0"))
    by_child = {j.find("child").get("link"): j for j in root.findall("joint")}
    poses = {}

    def pose(link):
        if link not in poses:
            j = by_child.get(link)
            poses[link] = np.eye(4) if j is None else pose(j.find("parent").get("link")) @ _origin(j) @ _motion(j, q.get(j.get("name"), 0.0))
        return poses[link]

    for link in root.findall("link"):
        pose(link.get("name"))
    return poses


def movable_joints(root):
    return [j for j in root.findall("joint") if j.get("type") in ("revolute", "prismatic", "continuous") and j.find("mimic") is None]


def random_q(root, rng):
    """Uniform within URDF limits; continuous joints in [-pi, pi]."""
    q = {}
    for j in movable_joints(root):
        lim = j.find("limit")
        lo, hi = (-np.pi, np.pi) if j.get("type") == "continuous" else (float(lim.get("lower")), float(lim.get("upper")))
        q[j.get("name")] = rng.uniform(lo, hi)
    return q


def set_qpos(model, data, root, q):
    """Write URDF joint positions (mimics applied) into MuJoCo qpos; the free base (if any) at identity."""
    data.qpos[:] = model.qpos0
    full = dict(q)
    for j in root.findall("joint"):
        m = j.find("mimic")
        if m is not None:
            full[j.get("name")] = float(m.get("multiplier", "1")) * full.get(m.get("joint"), 0.0) + float(m.get("offset", "0"))
    for name, v in full.items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        assert jid >= 0, f"URDF joint {name} has no MuJoCo joint"
        data.qpos[model.jnt_qposadr[jid]] = v
    mujoco.mj_kinematics(model, data)


def rotation_angle(r1, r2):
    """Angle (rad) of r1^T r2, robust near 0."""
    c = (np.trace(r1.T @ r2) - 1.0) / 2.0
    s = np.linalg.norm([r1.T @ r2 - (r1.T @ r2).T]) / (2.0 * np.sqrt(2.0))
    return float(np.arctan2(s, np.clip(c, -1.0, 1.0)))


def gl_available():
    """(ok, reason): can MuJoCo render offscreen here (MUJOCO_GL egl / osmesa / glfw)?"""
    try:
        m = mujoco.MjModel.from_xml_string("<mujoco><worldbody><geom size='1'/></worldbody></mujoco>")
        r = mujoco.Renderer(m, 8, 8)
        r.close()
        return True, ""
    except Exception as exc:  # noqa: BLE001
        return False, f"no offscreen OpenGL (MUJOCO_GL={os.environ.get('MUJOCO_GL')!r}): {exc}"
