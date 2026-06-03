cmake_minimum_required(VERSION 3.22)
project(vllm-cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# CPU compiler flags
set(CPU_FLAGS "-O3;-march=native;-fpic;-Wno-unused-result")

# Detect MLX (macOS Metal acceleration)
find_path(MLX_INCLUDE_DIR mlx/mlx.h
    HINTS /opt/homebrew/lib/python3.11/site-packages/mlx/include
    DOC "MLX C++ headers"
)
find_library(MLX_LIBRARY mlx
    HINTS /opt/homebrew/lib/python3.11/site-packages/mlx/lib
    DOC "MLX shared library"
)
find_library(MLX_JACCL_LIBRARY jaccl
    HINTS /opt/homebrew/lib/python3.11/site-packages/mlx/lib
    DOC "MLX jaccl shared library"
)

if(MLX_INCLUDE_DIR AND MLX_LIBRARY)
    set(MLX_FOUND TRUE)
    message(STATUS "MLX found: ${MLX_LIBRARY}")
    message(STATUS "MLX include: ${MLX_INCLUDE_DIR}")

    add_library(mlx_backend SHARED src/mlx_backend.cpp)
    target_include_directories(mlx_backend PUBLIC src ${MLX_INCLUDE_DIR})
    target_link_libraries(mlx_backend PRIVATE vllm-core ${MLX_LIBRARY} ${MLX_JACCL_LIBRARY})
    target_compile_options(mlx_backend PRIVATE ${CPU_FLAGS})
    target_compile_definitions(mlx_backend PUBLIC HAS_MLX)
    set_target_properties(mlx_backend PROPERTIES
        BUILD_RPATH "${CMAKE_CURRENT_SOURCE_DIR}/build;/opt/homebrew/lib/python3.11/site-packages/mlx/lib"
        INSTALL_RPATH "/opt/homebrew/lib/python3.11/site-packages/mlx/lib"
    )

    # MLX runner
    add_executable(mlx_runner src/mlx_runner.cpp)
    target_include_directories(mlx_runner PRIVATE src ${MLX_INCLUDE_DIR})
    target_link_libraries(mlx_runner vllm-core mlx_backend ${MLX_LIBRARY} ${MLX_JACCL_LIBRARY})
    target_compile_options(mlx_runner PRIVATE ${CPU_FLAGS})
    set_target_properties(mlx_runner PROPERTIES
        BUILD_RPATH "${CMAKE_CURRENT_BINARY_DIR};${CMAKE_CURRENT_SOURCE_DIR}/build;/opt/homebrew/lib/python3.11/site-packages/mlx/lib"
        INSTALL_RPATH "/opt/homebrew/lib/python3.11/site-packages/mlx/lib"
    )

    # Unified HTTP server with MLX GPU
    add_executable(server src/server.cpp)
    target_include_directories(server PRIVATE src ${MLX_INCLUDE_DIR})
    target_link_libraries(server vllm-core mlx_backend ${MLX_LIBRARY} ${MLX_JACCL_LIBRARY})
    target_compile_options(server PRIVATE ${CPU_FLAGS})
    set_target_properties(server PROPERTIES
        BUILD_RPATH "${CMAKE_CURRENT_BINARY_DIR};${CMAKE_CURRENT_SOURCE_DIR}/build;/opt/homebrew/lib/python3.11/site-packages/mlx/lib"
        INSTALL_RPATH "/opt/homebrew/lib/python3.11/site-packages/mlx/lib"
    )

    # Interactive CLI with MLX GPU
    add_executable(interactive src/interactive.cpp)
    target_include_directories(interactive PRIVATE src ${MLX_INCLUDE_DIR})
    target_link_libraries(interactive vllm-core mlx_backend ${MLX_LIBRARY} ${MLX_JACCL_LIBRARY})
    target_compile_options(interactive PRIVATE ${CPU_FLAGS})
    set_target_properties(interactive PROPERTIES
        BUILD_RPATH "${CMAKE_CURRENT_BINARY_DIR};${CMAKE_CURRENT_SOURCE_DIR}/build;/opt/homebrew/lib/python3.11/site-packages/mlx/lib"
        INSTALL_RPATH "/opt/homebrew/lib/python3.11/site-packages/mlx/lib"
    )
else()
    message(STATUS "MLX not found — building CPU-only server/CLI")
    add_executable(server src/server.cpp)
    target_link_libraries(server vllm-core)
    target_compile_options(server PRIVATE ${CPU_FLAGS})

    add_executable(interactive src/interactive.cpp)
    target_link_libraries(interactive vllm-core)
    target_compile_options(interactive PRIVATE ${CPU_FLAGS})
endif()

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
add_library(vllm-core STATIC
    src/model.cpp
    src/quantization.cpp
    src/quant_loader.cpp
    src/kv_cache_manager.cpp
    src/paged_attention.cpp
    src/paged_engine.cpp
    src/scheduler.cpp
)
target_include_directories(vllm-core PUBLIC src)
target_compile_options(vllm-core PRIVATE ${CPU_FLAGS})

# CPU runner
add_executable(cpu_runner src/cpu_runner.cpp)
target_link_libraries(cpu_runner vllm-core)
target_compile_options(cpu_runner PRIVATE ${CPU_FLAGS})

# (server and interactive built with MLX above)

# CUDA library
if(CUDA_FOUND)
    find_library(NCCL_LIBRARY nccl HINTS /usr/lib /usr/local/lib ${CUDA_TOOLKIT_ROOT_DIR}/lib64)
    find_path(NCCL_INCLUDE_DIR nccl.h HINTS /usr/include /usr/local/include ${CUDA_TOOLKIT_ROOT_DIR}/include)

    if(NCCL_LIBRARY AND NCCL_INCLUDE_DIR)
        message(STATUS "NCCL found: ${NCCL_LIBRARY}")
    else()
        message(WARNING "NCCL not found - tensor parallel support disabled")
    endif()

    add_library(vllm-cuda STATIC
        cuda/attention.cu
        cuda/moe_dispatch.cu
        cuda/tp.cu
        cuda/tp_model.cu
    )
    target_include_directories(vllm-cuda PUBLIC src cuda ${NCCL_INCLUDE_DIR})
    set_target_properties(vllm-cuda PROPERTIES
        CUDA_ARCHITECTURES "${CUDA_ARCHS}"
        CUDA_FLAGS "${CUDA_FLAGS}"
    )
    find_library(CUBLAS_LIBRARY cublas HINTS ${CUDA_TOOLKIT_ROOT_DIR}/lib64)
    target_link_libraries(vllm-cuda vllm-core ${NCCL_LIBRARY} ${CUBLAS_LIBRARY})
    if(NCCL_LIBRARY)
        target_compile_definitions(vllm-cuda PUBLIC HAS_NCCL)
    endif()
    message(STATUS "CUDA library: vllm-cuda")
endif()

# Install
install(TARGETS cpu_runner server DESTINATION bin)
if(CUDA_FOUND)
    install(TARGETS vllm-cuda DESTINATION lib)
endif()
install(DIRECTORY src/ DESTINATION include/vllm-cpp FILES_MATCHING PATTERN "*.h")
if(CUDA_FOUND)
    install(DIRECTORY cuda/ DESTINATION include/vllm-cpp/cuda FILES_MATCHING PATTERN "*.h" PATTERN "*.cuh")
endif()
