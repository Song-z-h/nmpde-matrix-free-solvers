cmake_minimum_required(VERSION 3.12.0)
project(lab06 LANGUAGES CXX C)

include(../common/cmake-common.cmake)

add_executable(lab-06 src/lab-06.cpp src/Heat.cpp)
deal_ii_setup_target(lab-06)


# -----------------------------
# matrix-free scaling executable
# -----------------------------
add_executable(heat_mf_scaling
  src/heat_mf_scaling.cpp  
  src/Heat.cpp           
)
deal_ii_setup_target(heat_mf_scaling)