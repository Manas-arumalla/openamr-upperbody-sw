# SPDX-License-Identifier: MIT
cmake_minimum_required(VERSION 3.8)
project(openamrobot_upperbody_mujoco)
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter)

ament_python_install_package(${PROJECT_NAME})
install(DIRECTORY config DESTINATION share/${PROJECT_NAME})
install(PROGRAMS scripts/generate_mjcf DESTINATION lib/${PROJECT_NAME})

# The MJCF is generated from the INSTALLED description on every build (no OUTPUT: always out of date), so it cannot
# drift from the URDF that Gazebo and MoveIt load. Baseline configuration, both base plates, wheeled base.
set(MJCF_DIR ${CMAKE_CURRENT_BINARY_DIR}/mjcf)
add_custom_target(mjcf ALL
  COMMAND ${CMAKE_COMMAND} -E rm -rf ${MJCF_DIR}
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/${PROJECT_NAME}/generate.py
          --out-dir ${MJCF_DIR} --file-name openamrobot_bp000 --base-plate bp000
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/${PROJECT_NAME}/generate.py
          --out-dir ${MJCF_DIR} --file-name openamrobot_bp050 --base-plate bp050
  COMMENT "Generating the OpenAMRobot 2.0 MJCF from combined_robot.xacro"
  VERBATIM)
install(DIRECTORY ${MJCF_DIR}/ DESTINATION share/${PROJECT_NAME}/mjcf)

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  # ament_add_pytest_test only warns when pytest is missing, which would leave an empty suite reported as
  # "0 failures". Refuse that instead (as the description package does).
  ament_has_pytest(_has_pytest QUIET)
  if(NOT _has_pytest)
    message(FATAL_ERROR "pytest not importable by ${Python3_EXECUTABLE}; tests would be silently empty")
  endif()
  # J01 every configuration generates and loads; the installed MJCF is current (no stale file)
  ament_add_pytest_test(j01_generate test/test_j01_generate.py TIMEOUT 600)
  # J02 kinematic parity with the URDF: every link, three lift stops x two base plates + 100 random samples (1e-9)
  ament_add_pytest_test(j02_kinematic_parity test/test_j02_kinematic_parity.py TIMEOUT 600)
  # J03 mass, COM and inertia per link = the URDF's (inertia within MuJoCo's own eigensolver bound)
  ament_add_pytest_test(j03_mass_parity test/test_j03_mass_parity.py TIMEOUT 600)
  # J04 limits, gains and joint dynamics = their sources (URDF, pinned control_gains.yaml, motor classes)
  ament_add_pytest_test(j04_limits_parity test/test_j04_limits_parity.py TIMEOUT 600)
  # J05 self-collision on, exclusions = the SRDF's disabled pairs, no self-contact at the SRDF named poses
  ament_add_pytest_test(j05_collision test/test_j05_collision.py TIMEOUT 600)
  # J06 lift speed/stroke/sag, arms under gravity, wheeled base (one strict xfail: wheel spin, see the test)
  ament_add_pytest_test(j06_dynamics test/test_j06_dynamics.py TIMEOUT 600)
  # J07 cameras on the URDF optical frames; RGB + depth offscreen (skips with the reason when no OpenGL)
  ament_add_pytest_test(j07_cameras test/test_j07_cameras.py TIMEOUT 600)
endif()

ament_package()
