# SPDX-License-Identifier: MIT
"""J03: mass parity. Per URDF link: mass, centre of mass and inertia tensor (both in the world frame) in MuJoCo equal
the URDF's; links without an <inertial> are massless in MuJoCo too; the total is the URDF total.

Mass 1e-12 relative and COM 1e-12 m, as fixed before the first run (measured: exactly 0 and 0).

Inertia. The first run used 1e-12 kg m^2, assuming only rounding was left, and measured 9.3e-10 kg m^2 (base_link).
MuJoCo compiling the URDF directly, with nothing of this package in between, gives the same number: it is MuJoCo's
compile-time diagonalisation of each inertia into principal axes, not the generator. So inertia is checked twice:
  1. The generated file equals MuJoCo's direct compile of the URDF to 1e-12 relative: what this package controls.
  2. Against the URDF tensor, within the precision of MuJoCo's eigensolver, derived from its source rather than from
     the measurement: mju_eig3 (engine_util_solve.c) stops its Jacobi rotations once the next rotation's cosine
     exceeds 1 - eigEPS, eigEPS = 1000 * mjMINVAL = 1e-12, so the principal frame can be off by up to
     theta = sqrt(2 * eigEPS) ~ 1.4e-6 rad. A frame error theta changes the tensor by about theta * (l_max - l_min);
     with a factor 2 for the residual off-diagonal it drops, the bound per link is 2 * sqrt(2 * eigEPS) * (l_max - l_min).
"""

import xml.etree.ElementTree as ET

import numpy as np
import pytest
from conftest import AXES, _rpy, generate, mujoco, urdf_fk, urdf_root

MASS_REL, COM_TOL = 1e-12, 1e-12
EIG_EPS = 1000.0 * mujoco.mjMINVAL  # mju_eig3's stopping tolerance
INERTIA_REL_VS_DIRECT_COMPILE = 1e-12


def eigensolver_bound(tensor):
    """Largest entry change MuJoCo's principal-axis diagonalisation may introduce in `tensor` (see module docstring)."""
    lam = np.linalg.eigvalsh(tensor)
    return 2.0 * np.sqrt(2.0 * EIG_EPS) * (lam[-1] - lam[0])


def _urdf_inertials(root):
    """{link: (mass, com in world 3, inertia in world 3x3)} at the zero pose; links without <inertial>: None."""
    fk = urdf_fk(root, {})
    out = {}
    for link in root.findall("link"):
        name = link.get("name")
        inertial = link.find("inertial")
        if inertial is None:
            out[name] = None
            continue
        o = inertial.find("origin")
        xyz = np.array([float(v) for v in (o.get("xyz", "0 0 0") if o is not None else "0 0 0").split()])
        rpy = [float(v) for v in (o.get("rpy", "0 0 0") if o is not None else "0 0 0").split()]
        r_in = fk[name][:3, :3] @ _rpy(*rpy)
        i = inertial.find("inertia")
        tensor = np.array([[float(i.get(a, "0")) for a in row] for row in
                           (("ixx", "ixy", "ixz"), ("ixy", "iyy", "iyz"), ("ixz", "iyz", "izz"))])
        out[name] = (float(inertial.find("mass").get("value")), fk[name][:3, :3] @ xyz + fk[name][:3, 3],
                     r_in @ tensor @ r_in.T)
    return out


def _mujoco_inertials(model):
    """{body: (mass, COM in world, inertia tensor in world)} at qpos0."""
    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)
    out = {}
    for b in range(1, model.nbody):
        r = data.ximat[b].reshape(3, 3)
        out[mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b)] = (
            float(model.body_mass[b]), data.xipos[b].copy(), r @ np.diag(model.body_inertia[b]) @ r.T)
    return out


@pytest.mark.parametrize("bp", AXES["base_plate"])
def test_per_link(bp, model_for):
    model, _ = model_for(generate.Config(base_plate=bp))
    urdf = _urdf_inertials(urdf_root(bp))
    mj = _mujoco_inertials(model)
    worst_mass = worst_com = worst_ratio = 0.0
    for link, ref in urdf.items():
        mass, com, inertia = mj[link]
        if ref is None:
            assert mass == 0.0 and not inertia.any(), f"{link}: no <inertial> in the URDF but MuJoCo gives it {mass} kg"
            continue
        worst_mass = max(worst_mass, abs(mass - ref[0]) / ref[0])
        worst_com = max(worst_com, float(np.abs(com - ref[1]).max()))
        err, bound = float(np.abs(inertia - ref[2]).max()), eigensolver_bound(ref[2])
        assert err <= bound, f"{link}: inertia differs by {err:.3e} kg m^2, beyond MuJoCo's eigensolver bound {bound:.3e}"
        worst_ratio = max(worst_ratio, err / bound if bound else 0.0)
    print(f"{bp}: worst relative mass error {worst_mass:.2e}, COM {worst_com:.2e} m, "
          f"inertia at {worst_ratio:.1%} of the eigensolver bound")
    assert worst_mass <= MASS_REL and worst_com <= COM_TOL


@pytest.mark.parametrize("bp", AXES["base_plate"])
def test_inertia_equals_mujocos_direct_urdf_compile(bp, model_for, tmp_path):
    """The file's inertias are exactly what MuJoCo makes of the URDF itself: the generator adds no error of its own."""
    model, _ = model_for(generate.Config(base_plate=bp))
    robot = generate.strip_for_mujoco(ET.tostring(urdf_root(bp), encoding="unicode"))
    generate.resolve_meshes(robot, tmp_path / "meshes")
    robot.insert(0, ET.fromstring(f'<mujoco><compiler meshdir="{tmp_path / "meshes"}" inertiafromgeom="false" '
                                  'fusestatic="false"/></mujoco>'))
    direct = _mujoco_inertials(mujoco.MjSpec.from_string(ET.tostring(robot, encoding="unicode")).compile())
    ours = _mujoco_inertials(model)
    worst = max(float(np.abs(ours[k][2] - direct[k][2]).max() / np.abs(direct[k][2]).max())
                for k in direct if direct[k][0] > 0)
    print(f"{bp}: generated file vs MuJoCo's direct URDF compile, worst relative inertia difference {worst:.2e}")
    assert worst <= INERTIA_REL_VS_DIRECT_COMPILE


@pytest.mark.parametrize("bp", AXES["base_plate"])
def test_total_mass(bp, model_for):
    model, _ = model_for(generate.Config(base_plate=bp))
    total = sum(float(i.find("mass").get("value")) for i in urdf_root(bp).iter("inertial"))
    print(f"{bp}: URDF total {total:.6f} kg, MuJoCo total {model.body_subtreemass[0]:.6f} kg")
    assert abs(model.body_subtreemass[0] - total) <= MASS_REL * total


def test_negative_control_phantom_mass(model_for, tmp_path):
    """Without inertiafromgeom="false" in the file (MjSpec.to_xml() in MuJoCo 3.9 drops it), MuJoCo gives the massless URDF links a
    mass from their geometry. The total-mass check must catch exactly that."""
    _, path = model_for(generate.Config())
    text = path.read_text()
    import re  # noqa: PLC0415

    phantom_text, n = re.subn(r'\s*inertiafromgeom="false"', "", text)
    assert n == 1
    (tmp_path / "meshes").symlink_to(path.parent / "meshes")
    (tmp_path / "phantom.xml").write_text(phantom_text)
    phantom = mujoco.MjModel.from_xml_path(str(tmp_path / "phantom.xml"))
    total = sum(float(i.find("mass").get("value")) for i in urdf_root().iter("inertial"))
    extra = phantom.body_subtreemass[0] - total
    print(f"phantom mass without inertiafromgeom=false: +{extra:.3f} kg")
    assert extra > 1.0


def test_negative_control_eigensolver_bound_is_tight_enough():
    """The inertia bound is not vacuous: a 1e-5 rad error in a real arm link's principal frame exceeds it.

    Rotating about the middle principal axis mixes the largest and smallest moments, which changes the tensor by
    theta * (l_max - l_min); the bound is 2 * sqrt(2e-12) = 2.83e-6 times that, so theta = 1e-5 is 3.5 times over."""
    t = _urdf_inertials(urdf_root())["openarm_left_link4"][2]
    _, vec = np.linalg.eigh(t)
    axis, th = vec[:, 1], 1e-5
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    r = np.eye(3) + np.sin(th) * k + (1 - np.cos(th)) * k @ k
    assert np.abs(r @ t @ r.T - t).max() > eigensolver_bound(t)


def test_masses_are_the_roadmap_mass_export():
    """Issue #58 asks for "our masses" (docs mass_summary.json, the total of the roadmap mass export). The URDF reads its
    masses from the in-repo copy of that export (roadmap_masses), so per export link the twin must have the export's mass
    and centre of mass in the link frame exactly (1e-9), and the link where the export places it: link_origin_m is
    written with 4 decimals, so 5e-5 m.

    The arms are one pseudo-link "openarm" in the export (12.26 kg, external_nominal: the URDF keeps upstream's per-link
    masses, which sum to 12.260007 kg), so they are checked by total mass at the export's 0.01 kg precision (+-0.005).
    That lumping (12.26 kg at (0.36, 0, 0.75) m; the URDF arms at zero are at (0.239, 0, 0.806) m) is the whole of the
    difference between mass_summary.json's COM, (0.0311, -0.0026, 0.3267) m, and the twin's: 17.0 mm in x, -7.8 mm in z.
    (The export also adds com_m to link_origin_m without the link's rotation, which is right only for unrotated links.)"""
    import yaml  # noqa: PLC0415
    from conftest import DESCRIPTION  # noqa: PLC0415

    export = yaml.safe_load(open(DESCRIPTION / "config" / "roadmap_mass_export.yaml"))
    model = mujoco.MjModel.from_xml_path(str(generate.share("openamrobot_upperbody_mujoco") / "mjcf" / "openamrobot_bp000.xml"))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)  # qpos0 = L1000, arms at zero, base_footprint at the origin; bp000 as the export
    checked = 0
    for gname, group in export["groups"].items():
        for link, e in group["links"].items():
            b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, link)
            if gname == "openarm":
                assert b < 0, "the export's arm entry is a pseudo-link, not a URDF link"
                arms = sum(model.body_mass[i] for i in range(1, model.nbody)
                           if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i).startswith("openarm_"))
                assert abs(arms - e["mass_kg"]) <= 0.005, f"arms {arms} kg vs export {e['mass_kg']} kg"
                continue
            assert b >= 0, f"export link {link} has no body"
            assert abs(model.body_mass[b] - e["mass_kg"]) <= 1e-9 * e["mass_kg"], link
            assert np.abs(model.body_ipos[b] - np.array(e["com_m"])).max() <= 1e-9, link
            assert np.abs(data.xpos[b] - np.array(e["link_origin_m"])).max() <= 5e-5 + 1e-12, link
            checked += 1
    total = float(model.body_subtreemass[0])
    print(f"{checked} export links equal; total {total:.6f} kg (mass_summary.json: 87.11 kg)")
    # openamrobot-docs ff8ca57 scripts/roadmap_model/mass_summary.json totals.L1000.mass_kg (the export pin of test_mass_export_source)
    assert abs(total - 87.11) <= 0.005, "mass_summary.json total, written to 0.01 kg"
