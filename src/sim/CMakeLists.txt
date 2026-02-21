cmake_minimum_required(VERSION 3.5)
project(sim)

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(trans REQUIRED)
find_package(mujocodl REQUIRED)
find_package(assets REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(yaml-cpp REQUIRED)

set(TARGET_DEPENDENCE
  ament_cmake
  ament_index_cpp
  trans
  mujocodl
  sensor_msgs
  rclcpp
  geometry_msgs
  tf2
  tf2_ros
  nav_msgs
  cv_bridge
  yaml-cpp
)

# Locate MuJoCo shared library.
# Prefer installed locations, but fall back to the vendored copy in this repo.
set(MUJOCO_LIB_PATH "")

# 1) merge-install layout (install/lib)
file(GLOB _mujoco_candidates_install
  "${CMAKE_INSTALL_PREFIX}/lib/libmujoco.so*"
)
if(_mujoco_candidates_install)
  list(SORT _mujoco_candidates_install)
  list(GET _mujoco_candidates_install -1 MUJOCO_LIB_PATH)
endif()

# 2) isolated layout (install/<pkg>/lib) - keep for compatibility
if(MUJOCO_LIB_PATH STREQUAL "")
  get_filename_component(COLCON_PREFIX_PATH "${CMAKE_INSTALL_PREFIX}" DIRECTORY)
  file(GLOB _mujoco_candidates_isolated
    "${COLCON_PREFIX_PATH}/mujocodl/lib/libmujoco.so*"
  )
  if(_mujoco_candidates_isolated)
    list(SORT _mujoco_candidates_isolated)
    list(GET _mujoco_candidates_isolated -1 MUJOCO_LIB_PATH)
  endif()
endif()

# 3) vendored library in source tree
if(MUJOCO_LIB_PATH STREQUAL "")
  file(GLOB _mujoco_candidates_vendored
    "${CMAKE_CURRENT_SOURCE_DIR}/../third_parties/mujocodl/mujoco/lib/libmujoco.so*"
  )
  if(_mujoco_candidates_vendored)
    list(SORT _mujoco_candidates_vendored)
    list(GET _mujoco_candidates_vendored -1 MUJOCO_LIB_PATH)
  endif()
endif()

if(MUJOCO_LIB_PATH STREQUAL "" OR NOT EXISTS "${MUJOCO_LIB_PATH}")
  message(FATAL_ERROR "MuJoCo library not found. Tried install and vendored locations.")
endif()

add_library(mujoco SHARED IMPORTED)
set_target_properties(mujoco PROPERTIES IMPORTED_LOCATION ${MUJOCO_LIB_PATH})

add_executable(sim src/sim.cc src/sim/SimPublisher.cc)
target_include_directories(sim PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)

target_link_libraries(sim
  ${YAML_CPP_LIBRARIES}
  mujoco
)
ament_target_dependencies(sim ${TARGET_DEPENDENCE})

install(TARGETS sim
  DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME})

ament_package()
