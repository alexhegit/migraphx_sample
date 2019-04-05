cmake_minimum_required(VERSION 3.5)
project (MIGRaphX_driver)

set (MIGRAPHX_FOLDER /home/mev/source/AMDMIGraphX)
set (MIGRAPHX_BUILD ${MIGRAPHX_FOLDER}/build)
set (CMAKE_CXX_COMPILER /opt/rocm/bin/hcc)
set (CMAKE_CXX_STANDARD 14)
set (MIOPEN_LIBS /opt/rocm/miopen/lib)
set (HIP_LIBS /opt/rocm/hip/lib)

include_directories(${MIGRAPHX_FOLDER}/src/include
		    ${MIGRAPHX_FOLDER}/src/targets/gpu/include
		    ${MIGRAPHX_FOLDER}/src/targets/cpu/include
		    inc)

link_directories(${MIGRAPHX_BUILD}/src/onnx
                 ${MIGRAPHX_BUILD}/src
                 ${MIGRAPHX_BUILD}/src/targets/gpu
                 ${MIGRAPHX_BUILD}/src/targets/cpu
		 ${MIOPEN_LIBS}
		 ${HIP_LIBS})

file(GLOB SRC_FILES "src/*.cpp")
set(bin_name migx)
add_executable(${bin_name} ${SRC_FILES})
target_link_libraries(${bin_name} migraphx migraphx_cpu migraphx_device migraphx_gpu migraphx_onnx hip_hcc MIOpen)
