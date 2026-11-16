# SPDX-License-Identifier: MIT
"""J05: the collision model. Self-collision is on; the pairs MuJoCo excludes are exactly the SRDF's disabled pairs
(the matrix MoveIt plans with); and at every SRDF named pose (arms home / hands_up / park x grippers open / closed x lift
L1000 / L1175 / L1350), which MoveIt treats as collision-free, MuJoCo finds no robot-robot contact either.

MuJoCo collides meshes as their convex hulls and MoveIt (FCL) uses the meshes, so the two can disagree on concave
shapes; this test is where such a disagreement would show up, and it is reported as a failure, not hidden.
"""

import itertools
import xml.etree.ElementTree as ET

import pytest
from conftest import MOVEIT, STOPS, generate, mujoco, urdf_root

SRDF = ET.parse(MOVEIT / "config" / "openamrobot.srdf").getroot()


def _group_state(name, group):
    gs = [g for g in SRDF.findall("group_state") if g.get("name") == name and g.get("group") == group]
    assert len(gs) == 1, f"SRDF group_state {name}/{group}: {len(gs)} found"
    return {j.get("name"): float(j.get("value")) for j in gs[0].findall("joint")}


ARM_POSES = ("home", "hands_up", "park")
GRIPPER = ("open", "closed")


@pytest.fixture(scope="module")
def model(model_for):
    return model_for(generate.Config())[0]


def test_self_collision_is_on(model):
    robot = [g for g in range(model.ngeom) if model.geom_bodyid[g] != 0 and model.geom_contype[g]]
    assert robot and all(model.geom_contype[g] & model.geom_conaffinity[g] for g in robot)


def test_exclusions_are_the_srdf_disabled_pairs(model):
    srdf = {frozenset((d.get("link1"), d.get("link2"))) for d in SRDF.findall("disable_collisions")}
    name = lambda b: mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b)  # noqa: E731
    excluded = {frozenset((name(model.exclude_signature[i] >> 16), name(model.exclude_signature[i] & 0xFFFF)))
                for i in range(model.nexclude)}
    assert excluded == srdf, f"only in MuJoCo: {excluded - srdf}; only in the SRDF: {srdf - excluded}"


@pytest.mark.parametrize("stop,arm_pose,gripper", list(itertools.product(STOPS, ARM_POSES, GRIPPER)))
def test_named_poses_have_no_self_contact(stop, arm_pose, gripper, model):
    from conftest import set_qpos  # noqa: PLC0415

    q = {"lift_joint": STOPS[stop]}
    for side in ("left", "right"):
        q.update(_group_state(arm_pose, f"{side}_arm"))
        q.update(_group_state(gripper, f"{side}_gripper"))
    data = mujoco.MjData(model)
    set_qpos(model, data, urdf_root(), q)
    mujoco.mj_collision(model, data)
    name = lambda g: mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[g])  # noqa: E731
    self_contacts = [(name(c.geom1), name(c.geom2), round(float(c.dist), 5)) for c in data.contact[:data.ncon]
                     if model.geom_bodyid[c.geom1] != 0 and model.geom_bodyid[c.geom2] != 0]
    assert not self_contacts, f"{stop} {arm_pose} {gripper}: robot-robot contacts (body, body, depth m): {self_contacts}"


def test_negative_control_without_exclusions_home_has_contacts(model_for):
    """The no-contact result is not vacuous: with the SRDF exclusions removed, the home pose at L1000 does have
    robot-robot contacts (the pairs MoveIt disables because they always touch), so collision detection is live."""
    _, path = model_for(generate.Config())
    spec = mujoco.MjSpec.from_file(str(path))
    for ex in list(spec.excludes):
        spec.delete(ex)
    bare = spec.compile()
    assert bare.nexclude == 0
    data = mujoco.MjData(bare)
    mujoco.mj_forward(bare, data)
    self_contacts = [c for c in data.contact[:data.ncon] if bare.geom_bodyid[c.geom1] and bare.geom_bodyid[c.geom2]]
    print(f"home @ L1000 without exclusions: {len(self_contacts)} robot-robot contacts")
    assert self_contacts
