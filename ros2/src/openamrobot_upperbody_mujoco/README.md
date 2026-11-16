# openamrobot_upperbody_mujoco

A MuJoCo model of OpenAMRobot 2.0 (base, 350 mm lift, two OpenArm 2.0 arms with grippers), **generated from
`combined_robot.xacro` on every build**, so it is the same robot that Gazebo and MoveIt load. Tests J01–J07 check that
it is, against the URDF, the SRDF and the description's mass and dynamics files (issue #58).

Status: simulation only, like everything in this repository. Nothing here is a hardware result.

## Use it

```bash
colcon build --packages-up-to openamrobot_upperbody_mujoco    # generates the MJCF (a few seconds)
source install/setup.bash
SHARE=$(ros2 pkg prefix openamrobot_upperbody_mujoco)/share/openamrobot_upperbody_mujoco/mjcf
python3 -m mujoco.viewer --mjcf=$SHARE/scene_openamrobot_bp000.xml     # floor, table, two cubes
```

| File (`share/openamrobot_upperbody_mujoco/mjcf/`) | What |
|---|---|
| `openamrobot_bp000.xml`, `openamrobot_bp050.xml` | the robot at the baseline configuration (column `dold_v4_350`, `hp30`, `bt10`), wheeled base. A component: include it from a top-level file |
| `scene_openamrobot_bp0x0.xml` | the robot, a floor, a table and two 40 mm cubes; keyframes `L1000`, `L1175`, `L1350` |
| `floor_openamrobot_bp0x0.xml` | the robot on an empty floor (for driving the base); same keyframes |
| `NOTICE.md`, `LICENSE-Apache-2.0.txt` | where every mesh and the model data come from, and their licences |

Other configurations, the fixed base, or arm gravity compensation (fixed base only):

```bash
ros2 run openamrobot_upperbody_mujoco generate_mjcf --out-dir /tmp/twin --base-plate bp050 --head-pitch hp25 \
    [--base-mode fixed [--arm-gravity-compensation]]
```

```python
import mujoco
from ament_index_python.packages import get_package_share_directory

share = get_package_share_directory("openamrobot_upperbody_mujoco") + "/mjcf"
m = mujoco.MjModel.from_xml_path(f"{share}/scene_openamrobot_bp000.xml")
d = mujoco.MjData(m)
mujoco.mj_resetDataKeyframe(m, d, m.key("L1175").id)    # lift at the L1175 stop, arms at SRDF home
d.ctrl[m.actuator("openarm_left_joint4").id] = 1.5      # arm actuators are position servos, named as the joints
d.ctrl[m.actuator("lift_joint").id] = 0.018             # the lift's input is a SPEED (m/s), clamped to the URDF velocity
d.ctrl[m.actuator("left_wheel_joint").id] = 1.0         # wheels: velocity servos (rad/s)
```

The lift is commanded by speed because the DOLD column is a speed-limited screw drive. The mock and Gazebo
`ros2_control` profiles give `lift_joint` a position command interface instead, so a future ROS bridge has to turn
position targets into a speed (at most 0.018 m/s) toward the target.

Cameras `head`, `left_wrist` and `right_wrist` render with `mujoco.Renderer`. On an RTX 5060 with `MUJOCO_GL=egl`,
the head view (1280×720) takes 2.0 ms and the wrist views (640×480) 1.0 ms per frame, with depth from MuJoCo's own
buffer.

## Where the numbers come from

Nothing is tuned by eye. Every number is taken from the URDF, from the pinned OpenArm description, from the description
package's own data files, or from `config/mujoco_twin.yaml`, where each entry has `value / source / status` (the
convention of `datums.yaml`).

| Part | Model | Source |
|---|---|---|
| Masses, COMs, inertias | the URDF's, nothing added (`inertiafromgeom="false"`) | URDF, which reads the roadmap mass export (`roadmap_masses`); J03 checks both |
| Arm joints | position servos, kp/kd = the MIT gains the real arm gets | `openarm_description/.../config/arm/control/control_gains.yaml` at the pin, read at build time: the file the `real` profile uses |
| Arm force limits | URDF `effort` (40/40/27/27/7/7/7 N m) | URDF; J04 |
| Arm joint damping, friction, armature | per joint (motor classes DM8009/DM4340/DM4310, finger) | the description's `config/sim_dynamics/openarm_v2.yaml` (#66): damping and friction reach MuJoCo through the URDF (`sim_dynamics:=on`), and armature is set from the same file. This is the one source Gazebo uses too |
| Gravity compensation | off by default. `--arm-gravity-compensation` (fixed base only) compensates the arm bodies through the arm and lift actuators, inside their force limits; no DOF gets any of it as a passive force | the pinned `openarm_ros2` driver sends `{kp, kd, q, qd, tau}`, and a position controller leaves `tau` at 0; J04 |
| Grippers | `finger_joint1` servo (kp 10, kv 0.2, ±7 N m); `finger_joint2` follows it through a joint equality built from the URDF `<mimic>` | openarm_mujoco finger gains; URDF |
| Lift | speed-limited servo (`intvelocity`): input speed ≤ URDF velocity (0.018 m/s), force ≤ URDF effort (800 N); stiffness sized for 0.5 mm static error under the load it carries | URDF (datums `lift_max_*`); J06 |
| Base | free-floating; drive wheels are velocity servos, wheel friction 1.5, casters frictionless | platform-sw `<gazebo>` surfaces. The wheel gain is a placeholder (no hub-motor data) |
| Self-collision | on, minus the SRDF's disabled pairs (MoveIt's matrix; 180 at main `37b93f5`) | SRDF; J05 |
| Solver | `timestep 0.001`, implicitfast, elliptic cone, `impratio 10` | openarm_mujoco |
| Cameras | on the URDF optical frames | head fovy 52° (provisional, Enactic's ZED model); the wrist fovy is a placeholder |

## Tests

`colcon test --packages-select openamrobot_upperbody_mujoco`: 88 pytest cases (colcon's summary also counts the 7
CTest targets, 95 in all), about a minute, passing on MuJoCo 3.9.0 and 4.0.1. Recorded 2026-11-16 on main `37b93f5`:

| ID | Checks | Result |
|---|---|---|
| J01 | every value of every configuration axis generates and loads, one body per URDF link; the installed MJCF is what the current source generates; gravity compensation refuses a free base | pass |
| J02 | every link's pose = URDF forward kinematics, at L1000/L1175/L1350 × bp000/bp050 and 100 random joint samples (1e-9) | worst 4.6e-16 m, 7.9e-16 rad |
| J03 | per-link mass, COM, inertia = the URDF's; every link of the roadmap mass export; total = mass_summary.json | mass and COM exact; total 87.109007 kg (mass_summary.json: 87.11); inertia identical to MuJoCo's own URDF compile and within its eigensolver bound of the URDF |
| J04 | joint ranges, actuator ranges, gains, joint dynamics (all 18 sim_dynamics entries), mimic, contact parameters, mirrored arm meshes, gravity-compensation routing; MoveIt's joint_limits.yaml adds no position or velocity limit | exact |
| J05 | exclusions = the SRDF's pairs; no self-contact at the SRDF named poses | 0 contacts at 18 pose × stop combinations |
| J06 | lift speed, stroke and sag; arms hold `hands_up` under gravity by their equation of motion; base drives straight | lift peak 0.0180 m/s, stroke 19.43 s; arms droop up to 0.072 rad without gravity compensation (kp 70 at the shoulder); straight 0.0999 m/s for 0.100 |
| J07 | cameras on the optical frames; RGB and depth render; the head camera sees the scene at L1000 | pass |

Each check has a negative control that shows it can fail: a 1 mm lift offset, the phantom mass, removed exclusions, and
a frame error beyond the inertia bound.

## Known gaps (measured, not fixed)

- **Turning on the spot** (J06, a strict xfail). The URDF's drive wheel is a rigid cylinder 46 mm wide. MuJoCo
  contacts it at both rim edges, and the two edges scrub at full friction when the base turns. With the placeholder
  wheel servo the base turns at 0.15 rad/s where a thin wheel gives 0.50; with a stiff servo it reaches 0.31. Gazebo
  (DART) on the same wheels turns at 0.48 rad/s but goes straight at 0.081 m/s, where MuJoCo gives 0.0999. The
  wheel-ground model and the hub motor's torque and acceleration limits are for the maintainers to decide.
- **The drive wheels overlap the floor by 2 mm in the URDF.** The axle is at `base_link` height (0.098 m) and the
  wheel radius is 0.100 m.
- **The wrist cameras do not see the fingers** (0.0 % of pixels; their poses are upstream placeholders, issue #24).
- **The head camera (hp30) loses the near workspace as the lift rises.** The cubes, 0.68 m ahead at table height, fill
  0.5 % of the view at L1000 and none at L1175 or L1350.
- **mass_summary.json's centre of mass is 17.0 mm ahead of the URDF's and 7.8 mm lower** (87.11 kg either way). The
  export lumps both arms into one 12.26 kg block at (0.36, 0, 0.75) m, while the URDF arms at zero are at
  (0.239, 0, 0.806) m; every other group matches exactly.
- MoveIt's joint acceleration limits are not modelled; the lift starts at full speed (MoveIt plans 0.01 m/s²).
- The visual meshes are the collision STLs (MuJoCo has no COLLADA reader), so camera images show collision shells.
- MuJoCo collides meshes as their convex hulls; MoveIt (FCL) uses the meshes themselves.
- No ROS bridge yet: drive it from Python.

## Why a colcon package

Issue #58 asks for a `mujoco/` package. This is one, kept under `ros2/src` like the other five: generating at build
time from the installed description, with the same `$(find …)` resolution Gazebo and MoveIt get, needs colcon. Renaming
or moving it is easy.

## Credits

- [enactic/openarm_mujoco](https://github.com/enactic/openarm_mujoco) (Apache-2.0): finger gains, solver options and the
  head-camera model. The joint dynamics come from it too, through the description's sim_dynamics profile.
- [OpenRAL](https://github.com/OpenRAL/openral) (Apache-2.0), Adrian Llopart's OpenAMRobot 2.0 twin (openral#497):
  one mesh asset per (file, scale) for the mirrored arm, the mimic as a joint equality, and the optical-frame camera
  rotation. Reimplemented, not copied.
