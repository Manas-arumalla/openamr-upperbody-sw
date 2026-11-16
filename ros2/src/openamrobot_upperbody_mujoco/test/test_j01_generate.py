# SPDX-License-Identifier: MIT
"""J01: every buildable configuration generates an MJCF that MuJoCo loads, with one body per URDF link, and the MJCF
installed by the build is byte-identical to what the installed generator writes now (no stale file)."""

import itertools

import pytest
from conftest import AXES, BASELINE, MUJOCO_SHARE, generate, mujoco, urdf_root

CONFIGS = [generate.Config(base_plate=bp, head_pitch=hp) for bp, hp in itertools.product(AXES["base_plate"], AXES["head_pitch"])]
VARIANTS = [generate.Config(base_mode="fixed"), generate.Config(arm_gravity_compensation=True)]


def test_mujoco_version():
    # Polynomial joint damping ([linear, quadratic, cubic]) is what generate.py writes: MuJoCo 3.8 or later.
    major, minor = (int(x) for x in mujoco.__version__.split(".")[:2])
    assert (major, minor) >= (3, 8), mujoco.__version__


@pytest.mark.parametrize("cfg", CONFIGS + VARIANTS, ids=lambda c: c.name)
def test_generates_and_loads(cfg, model_for):
    model, _ = model_for(cfg)
    root = urdf_root(cfg.base_plate, cfg.head_pitch, cfg.base_tilt, cfg.column)
    links = {link.get("name") for link in root.findall("link")}
    bodies = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) for b in range(1, model.nbody)}
    assert bodies == links, f"bodies without a link: {bodies - links}; links without a body: {links - bodies}"
    urdf_joints = {j.get("name") for j in root.findall("joint") if j.get("type") != "fixed"}
    mj_joints = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(model.njnt)}
    base = {"base_free"} if cfg.base_mode == "wheeled" else set()
    assert mj_joints == urdf_joints | base


def test_base_modes(model_for):
    wheeled, _ = model_for(generate.Config())
    fixed, _ = model_for(generate.Config(base_mode="fixed"))
    assert wheeled.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE
    assert mujoco.mjtJoint.mjJNT_FREE not in list(fixed.jnt_type)
    assert wheeled.nu == fixed.nu + 2  # the two drive-wheel servos


@pytest.mark.parametrize("bp", AXES["base_plate"])
def test_installed_mjcf_is_current(bp, tmp_path):
    """I01-style: the MJCF the build installed equals what the installed generator writes from the installed URDF."""
    installed = MUJOCO_SHARE / "mjcf" / f"openamrobot_{bp}.xml"
    assert installed.is_file(), f"{installed} not installed: the build did not generate it"
    fresh = generate.generate(generate.Config(base_plate=bp, head_pitch=BASELINE["head_pitch"],
                                              base_tilt=BASELINE["base_tilt"], column=BASELINE["column"]),
                              tmp_path, f"openamrobot_{bp}")
    assert fresh.read_text() == installed.read_text(), "the installed MJCF is stale: rebuild the package"
