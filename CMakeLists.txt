cmake_minimum_required(VERSION 3.22)
project(vllm-cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# CPU compiler flags
set(CPU_FLAGS "-O3;-march=native;-fpic;-Wno-unused-result")

# Detect CUDA
find_package(CUDA 11.8 QUIET)

if(CUDA_FOUND)
    enable_language(CUDA)
    set(CUDA_ARCHS "80;90")
    set(CUDA_FLAGS "-Xcompiler=-O3,-fpic -Xcompiler=-fPIC")
    foreach(arch ${CUDA_ARCHS})
        list(APPEND CUDA_FLAGS "--generate-code=arch=compute_${arch},code=sm_${arch}")
    endforeach()
    set(CUDA_FLAGS "${CUDA_FLAGS} -O3 -use_fast_math -Xcompiler=-Wno-unused-result")
    message(STATUS "CUDA found: ${CUDA_TOOLKIT_ROOT_DIR}")
    message(STATUS "CUDA architectures: ${CUDA_ARCHS}")
else()
    message(STATUS "CUDA not found - building CPU-only")
endif()

# Core library (CPU)
add_library(vllm-core STATIC src/model.cpp)
target_include_directories(vllm-core PUBLIC src)
target_compile_options(vllm-core PRIVATE ${CPU_FLAGS})

# CPU runner
add_executable(cpu_runner src/cpu_runner.cpp)
target_link_libraries(cpu_runner vllm-core)
target_compile_options(cpu_runner PRIVATE ${CPU_FLAGS})

# HTTP server
add_executable(http_server src/http_server.cpp)
target_link_libraries(http_server vllm-core)
target_compile_options(http_server PRIVATE ${CPU_FLAGS})

# CUDA library
if(CUDA_FOUND)
    add_library(vllm-cuda STATIC
        cuda/attention.cu
        cuda/moe_dispatch.cu
    )
    target_include_directories(vllm-cuda PUBLIC src cuda)
    set_target_properties(vllm-cuda PROPERTIES
        CUDA_ARCHITECTURES "${CUDA_ARCHS}"
        CUDA_FLAGS "${CUDA_FLAGS}"
    )
    target_link_libraries(vllm-cuda vllm-core)
    message(STATUS "CUDA library: vllm-cuda")
endif()

# Install
install(TARGETS cpu_runner http_server DESTINATION bin)
if(CUDA_FOUND)
    install(TARGETS vllm-cuda DESTINATION lib)
endif()
install(DIRECTORY src/ DESTINATION include/vllm-cpp FILES_MATCHING PATTERN "*.h")
if(CUDA_FOUND)
    install(DIRECTORY cuda/ DESTINATION include/vllm-cpp/cuda FILES_MATCHING PATTERN "*.h" PATTERN "*.cuh")
endif()
