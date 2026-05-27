#pragma once
#include <string>
#include <iostream>

// Auto-detect best available compute backend at runtime.
// Priority: MLX Metal GPU > CPU

struct DeviceInfo {
    enum Backend { CPU, MLX };
    Backend backend = CPU;
    std::string name = "CPU";
    bool gpu_available = false;

    static DeviceInfo detect() {
        DeviceInfo info;
#ifdef HAS_MLX
        info.backend = MLX;
        info.name = "MLX (Metal GPU)";
        info.gpu_available = true;
#else
        info.backend = CPU;
        info.name = "CPU (NEON)";
        info.gpu_available = false;
#endif
        return info;
    }

    static void print_info() {
        auto info = detect();
        std::cerr << "[INFO] device: " << info.name
                  << "  gpu=" << (info.gpu_available ? "yes" : "no")
                  << std::endl;
    }
};
