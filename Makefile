CXX      := c++
CXXFLAGS := -std=c++20 -O3 -march=native -fpic -Wno-unused-result
LDFLAGS  :=

BUILD_DIR := build
SRC_DIR   := src
CUDA_DIR  := cuda

# --- MLX (macOS Metal) ---
MLX_PREFIX := /opt/homebrew/lib/python3.11/site-packages/mlx
MLX_INC    := $(MLX_PREFIX)/include
MLX_LIB    := $(MLX_PREFIX)/lib
HAS_MLX    := $(shell [ -f $(MLX_LIB)/libmlx.dylib ] && echo 1 || echo 0)

MLX_INC_FLAG  := $(shell [ $(HAS_MLX) = 1 ] && echo "-I$(MLX_INC)")
MLX_LIB_FLAG  := $(shell [ $(HAS_MLX) = 1 ] && echo "-L$(MLX_LIB) -lmlx -ljaccl")
MLX_RPATH     := $(shell [ $(HAS_MLX) = 1 ] && echo "-Wl,-rpath,$(MLX_LIB)")

# --- CUDA ---
HAS_CUDA := $(shell command -v nvcc >/dev/null 2>&1 && echo 1 || echo 0)
ifeq ($(HAS_CUDA),1)
  NVCC      := nvcc
  CUDA_ARCH := 80 90
  CUDA_FLAGS:= -O3 -use_fast_math -Xcompiler=-O3,-fpic -Xcompiler=-Wno-unused-result
  CUDA_GEN  := $(foreach a,$(CUDA_ARCH),-gencode arch=compute_$(a),code=sm_$(a))
endif

# --- Targets ---
.PHONY: all clean cpu mlx cuda info

all: cpu
ifeq ($(HAS_MLX),1)
all: mlx
endif
ifeq ($(HAS_CUDA),1)
all: cuda
endif
	@echo "[DONE] all targets built"

# ============================================================
# Build directory
# ============================================================
$(BUILD_DIR):
	@mkdir -p $(BUILD_DIR)

# ============================================================
# Core library
# ============================================================
CORE_SRC  := $(SRC_DIR)/model.cpp $(SRC_DIR)/quantization.cpp $(SRC_DIR)/quant_loader.cpp
CORE_OBJ  := $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(CORE_SRC))
CORE_LIB  := $(BUILD_DIR)/libvllm-core.a

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -I$(SRC_DIR) -c $< -o $@

$(CORE_LIB): $(CORE_OBJ)
	ar rcs $@ $^

# ============================================================
# CPU targets
# ============================================================
cpu: $(BUILD_DIR)/cpu_runner
ifeq ($(HAS_MLX),0)
cpu: $(BUILD_DIR)/server
endif
	@echo "[DONE] cpu targets built"

$(BUILD_DIR)/cpu_runner: $(SRC_DIR)/cpu_runner.cpp $(CORE_LIB) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -I$(SRC_DIR) $< $(CORE_LIB) -o $@

# --- CPU server (PagedEngine backend, Linux) ---
ifeq ($(HAS_MLX),0)
PAGED_CPP := $(SRC_DIR)/paged_engine.cpp $(SRC_DIR)/paged_attention.cpp \
             $(SRC_DIR)/kv_cache_manager.cpp $(SRC_DIR)/scheduler.cpp
PAGED_OBJ := $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(PAGED_CPP))

$(BUILD_DIR)/server: $(SRC_DIR)/server.cpp $(CORE_LIB) $(PAGED_OBJ) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -I$(SRC_DIR) $< $(PAGED_OBJ) $(CORE_LIB) -o $@

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp $(SRC_DIR)/%.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -I$(SRC_DIR) -c $< -o $@
endif

# ============================================================
# MLX targets (macOS Metal)
# ============================================================
ifeq ($(HAS_MLX),1)
MLX_BACKEND_SRC := $(SRC_DIR)/mlx_backend.cpp
MLX_BACKEND_OBJ := $(BUILD_DIR)/mlx_backend.o
MLX_BACKEND_LIB := $(BUILD_DIR)/libmlx_backend.dylib

$(MLX_BACKEND_OBJ): $(MLX_BACKEND_SRC) $(SRC_DIR)/mlx_backend.h $(SRC_DIR)/model.h $(SRC_DIR)/config.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -I$(SRC_DIR) $(MLX_INC_FLAG) -c $< -o $@

$(MLX_BACKEND_LIB): $(MLX_BACKEND_OBJ) $(CORE_LIB)
	$(CXX) -shared $(CXXFLAGS) $< $(CORE_LIB) $(MLX_LIB_FLAG) $(MLX_RPATH) -Wl,-rpath,$(CURDIR)/$(BUILD_DIR) -o $@

mlx: $(BUILD_DIR)/mlx_runner $(BUILD_DIR)/server $(BUILD_DIR)/interactive
	@echo "[DONE] mlx targets built"

$(BUILD_DIR)/mlx_runner: $(SRC_DIR)/mlx_runner.cpp $(MLX_BACKEND_LIB) $(CORE_LIB) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -I$(SRC_DIR) $(MLX_INC_FLAG) $< $(CORE_LIB) $(MLX_BACKEND_LIB) $(MLX_LIB_FLAG) $(MLX_RPATH) -Wl,-rpath,$(CURDIR)/$(BUILD_DIR) -o $@

$(BUILD_DIR)/server: $(SRC_DIR)/server.cpp $(MLX_BACKEND_LIB) $(CORE_LIB) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -DHAS_MLX -I$(SRC_DIR) $(MLX_INC_FLAG) $< $(CORE_LIB) $(MLX_BACKEND_LIB) $(MLX_LIB_FLAG) $(MLX_RPATH) -Wl,-rpath,$(CURDIR)/$(BUILD_DIR) -o $@

$(BUILD_DIR)/interactive: $(SRC_DIR)/interactive.cpp $(MLX_BACKEND_LIB) $(CORE_LIB) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -I$(SRC_DIR) $(MLX_INC_FLAG) $< $(CORE_LIB) $(MLX_BACKEND_LIB) $(MLX_LIB_FLAG) $(MLX_RPATH) -Wl,-rpath,$(CURDIR)/$(BUILD_DIR) -o $@
endif

# ============================================================
# CUDA targets
# ============================================================
ifeq ($(HAS_CUDA),1)
CUDA_SRCS := $(CUDA_DIR)/attention.cu $(CUDA_DIR)/moe_dispatch.cu
CUDA_OBJS := $(patsubst $(CUDA_DIR)/%.cu,$(BUILD_DIR)/%.o,$(CUDA_SRCS))
CUDA_LIB  := $(BUILD_DIR)/libvllm-cuda.a

$(BUILD_DIR)/%.o: $(CUDA_DIR)/%.cu $(CUDA_DIR)/%.h $(CUDA_DIR)/utils.cuh | $(BUILD_DIR)
	$(NVCC) $(CUDA_FLAGS) $(CUDA_GEN) -I$(SRC_DIR) -I$(CUDA_DIR) -c $< -o $@

$(CUDA_LIB): $(CUDA_OBJS)
	ar rcs $@ $^

cuda: $(CUDA_LIB)
	@echo "[DONE] cuda targets built"
endif

# ============================================================
# Clean
# ============================================================
clean:
	rm -rf $(BUILD_DIR)

# ============================================================
# Info
# ============================================================
info:
	@echo "CXX:        $(CXX)"
	@echo "BUILD_DIR:  $(BUILD_DIR)"
	@echo "MLX:        $(if $(filter 1,$(HAS_MLX)),yes ($(MLX_PREFIX)),no)"
	@echo "CUDA:       $(if $(filter 1,$(HAS_CUDA)),yes,no)"
	@echo ""
	@echo "Targets:"
	@echo "  make          - build all available targets"
	@echo "  make cpu      - build CPU-only targets (runner + server on Linux)"
	@echo "  make mlx      - build MLX backend + runner + server"
	@echo "  make cuda     - build CUDA kernels"
	@echo "  make clean    - remove build artifacts"
	@echo "  make info     - show this info"
