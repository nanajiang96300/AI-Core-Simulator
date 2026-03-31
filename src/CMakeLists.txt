CMAKE_MINIMUM_REQUIRED(VERSION 3.15)

# project setting
set(LIB_NAME "Simulator")

# set source and headers
file(GLOB_RECURSE SRC_FILES
  "${CMAKE_SOURCE_DIR}/src/*.h"
  "${CMAKE_SOURCE_DIR}/src/*.cc"
)

# Ensure new custom LS operator, ChannelModel, and TraceLogger sources are linked
list(APPEND SRC_FILES
  "${CMAKE_SOURCE_DIR}/src/operations/LSEstimatorOp.cc"
  "${CMAKE_SOURCE_DIR}/src/models/ChannelModel.cc"
  "${CMAKE_SOURCE_DIR}/src/TraceLogger.cc"
  "${CMAKE_SOURCE_DIR}/src/operations/LDLDecompOp.cc"
  "${CMAKE_SOURCE_DIR}/src/models/LDLModel.cc"
  "${CMAKE_SOURCE_DIR}/src/operations/LDLDecompNoBlockOp.cc"
  "${CMAKE_SOURCE_DIR}/src/models/LDLNoBlockModel.cc"
  "${CMAKE_SOURCE_DIR}/src/operations/DeepUnfoldNPUOp.cc"
  "${CMAKE_SOURCE_DIR}/src/models/DeepUnfoldModel.cc"
  "${CMAKE_SOURCE_DIR}/src/operations/DeepUnfoldNPUOptOp.cc"
  "${CMAKE_SOURCE_DIR}/src/models/DeepUnfoldNPUOptModel.cc"
  "${CMAKE_SOURCE_DIR}/src/operations/CholeskyInvOp.cc"
  "${CMAKE_SOURCE_DIR}/src/models/CholeskyModel.cc"
  "${CMAKE_SOURCE_DIR}/src/operations/CholeskyInvNoBlockOp.cc"
  "${CMAKE_SOURCE_DIR}/src/models/CholeskyNoBlockModel.cc"
  "${CMAKE_SOURCE_DIR}/src/operations/NewtonSchulzOptOp.cc"
  "${CMAKE_SOURCE_DIR}/src/models/NewtonSchulzOptModel.cc"
  "${CMAKE_SOURCE_DIR}/src/operations/SeriesInverseOp.cc"
  "${CMAKE_SOURCE_DIR}/src/models/SeriesInverseModel.cc")

# build
add_executable(${LIB_NAME} ${SRC_FILES})
add_library(${LIB_NAME}_lib ${SRC_FILES})
