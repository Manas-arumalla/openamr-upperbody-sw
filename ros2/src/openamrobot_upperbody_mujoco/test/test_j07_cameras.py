# SPDX-License-Identifier: MIT
"""J07: cameras. Each camera sits exactly on its URDF optical frame (REP-103, z forward, y down) turned half a turn
about x into MuJoCo's convention (looks along -z, y up), tolerance 1e-9; each renders RGB and depth offscreen at its
configured resolution (skipped, with the reason, where no offscreen OpenGL exists); and the head camera sees the table
at L1000. What the wrist cameras see of the gripper is measured and printed, not asserted: their poses are upstream
placeholders (issue #24)."""

import time

import numpy as np
import pytest
import yaml
from conftest import generate, gl_available, mujoco, rotation_angle, urdf_fk, urdf_root

CAMERAS = yaml.safe_load(open(generate.config_dir() / "mujoco_twin.yaml"))["cameras"]
RX_PI = np.diag([1.0, -1.0, -1.0])


@pytest.mark.parametrize("cam", list(CAMERAS))
@pytest.mark.parametrize("lift", [0.0, 0.35])
def test_camera_on_urdf_optical_frame(cam, lift, model_for):
    model, _ = model_for(generate.Config())
    root = urdf_root()
    data = mujoco.MjData(model)
    from conftest import set_qpos  # noqa: PLC0415

    set_qpos(model, data, root, {"lift_joint": lift})
    mujoco.mj_camlight(model, data)  # mj_kinematics does not place cameras
    frame = CAMERAS[cam]["value"]["frame"]
    ref = urdf_fk(root, {"lift_joint": lift})[frame]
    c = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam)
    assert c >= 0
    assert np.linalg.norm(data.cam_xpos[c] - ref[:3, 3]) <= 1e-9
    assert rotation_angle(data.cam_xmat[c].reshape(3, 3), ref[:3, :3] @ RX_PI) <= 1e-9
    v = CAMERAS[cam]["value"]
    assert model.cam_fovy[c] == v["fovy_deg"] and list(model.cam_resolution[c]) == v["resolution"]


@pytest.fixture(scope="module")
def scene(scene_for):
    ok, why = gl_available()
    if not ok:
        pytest.skip(why)
    return scene_for(generate.Config())


@pytest.mark.parametrize("cam", list(CAMERAS))
def test_renders_rgb_and_depth(cam, scene):
    c = mujoco.mj_name2id(scene, mujoco.mjtObj.mjOBJ_CAMERA, cam)
    w, h = scene.cam_resolution[c]
    data = mujoco.MjData(scene)
    mujoco.mj_resetDataKeyframe(scene, data, mujoco.mj_name2id(scene, mujoco.mjtObj.mjOBJ_KEY, "L1000"))
    mujoco.mj_forward(scene, data)
    with mujoco.Renderer(scene, int(h), int(w)) as r:
        r.update_scene(data, camera=cam)
        rgb = r.render()
        t = time.perf_counter()
        for _ in range(20):
            r.update_scene(data, camera=cam)
            r.render()
        ms = (time.perf_counter() - t) / 20 * 1e3
        r.enable_depth_rendering()
        r.update_scene(data, camera=cam)
        depth = r.render()
    print(f"{cam}: {w}x{h} RGB {ms:.2f} ms/frame, depth {np.nanmin(depth):.3f}..{np.nanmax(depth):.3f} m")
    assert rgb.shape == (h, w, 3) and rgb.std() > 1.0, "blank image"
    assert depth.shape == (h, w) and np.isfinite(depth).all() and depth.min() > 0.0


def _segmentation(scene, data, cam):
    c = mujoco.mj_name2id(scene, mujoco.mjtObj.mjOBJ_CAMERA, cam)
    w, h = scene.cam_resolution[c]
    with mujoco.Renderer(scene, int(h), int(w)) as r:
        r.enable_segmentation_rendering()
        r.update_scene(data, camera=cam)
        seg = r.render()
    geoms = seg[..., 0][seg[..., 1] == mujoco.mjtObj.mjOBJ_GEOM]
    names = [mujoco.mj_id2name(scene, mujoco.mjtObj.mjOBJ_BODY, scene.geom_bodyid[g]) or "world" for g in geoms]
    return names, int(w) * int(h)


def test_head_camera_sees_the_scene_objects_at_l1000(scene):
    """The scene is laid out so the head camera sees the table and both cubes at L1000. At the higher stops the near
    table leaves the view (hp30, fovy 52): printed per stop, asserted only at L1000."""
    data = mujoco.MjData(scene)
    shares = {}
    for stop in ("L1000", "L1175", "L1350"):
        mujoco.mj_resetDataKeyframe(scene, data, mujoco.mj_name2id(scene, mujoco.mjtObj.mjOBJ_KEY, stop))
        mujoco.mj_forward(scene, data)
        names, n = _segmentation(scene, data, "head")
        shares[stop] = {k: names.count(k) / n for k in ("table", "cube_red", "cube_blue")}
        print(f"head camera @ {stop}: share of pixels {', '.join(f'{k} {v:.2%}' for k, v in shares[stop].items())}")
    assert all(v > 0.0 for v in shares["L1000"].values())


@pytest.mark.parametrize("side", ["left", "right"])
def test_wrist_camera_view_of_the_gripper_is_reported(side, scene):
    data = mujoco.MjData(scene)
    mujoco.mj_resetDataKeyframe(scene, data, mujoco.mj_name2id(scene, mujoco.mjtObj.mjOBJ_KEY, "L1000"))
    mujoco.mj_forward(scene, data)
    names, n = _segmentation(scene, data, f"{side}_wrist")
    fingers = sum(1 for k in names if k in (f"openarm_{side}_ee_link1", f"openarm_{side}_ee_link2"))
    hand = sum(1 for k in names if k == f"openarm_{side}_ee_base_link")
    print(f"{side}_wrist camera (upstream placeholder pose, #24): fingers on {fingers / n:.1%} of pixels, "
          f"gripper body (ee_base_link) on {hand / n:.1%}")
