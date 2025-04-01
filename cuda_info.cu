#include <stdio.h>
#include <cuda_runtime.h>

int getCoresPerSM(int major, int minor) {
    // 根据NVIDIA架构确定每个SM的核心数
    if (major == 9) {  // Hopper
        return 128;
    } else if (major == 8) { // Ampere
        return (minor == 0) ? 64 : 128;
    } else if (major == 7) { // Volta/Ampere
        return 64;
    } else if (major == 6) { // Pascal
        return (minor == 1 || minor == 2) ? 128 : 64;
    } else if (major == 5) { // Maxwell
        return 128;
    } else if (major == 3) { // Kepler
        return 192;
    } else if (major == 2) { // Fermi
        return 32;
    }
    return 0; // 未知架构
}

int main() {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    if (deviceCount == 0) {
        printf("No CUDA-capable devices detected.\n");
        return 0;
    }

    printf("Found %d CUDA device(s):\n", deviceCount);

    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);

        printf("\nDevice %d: \"%s\"\n", i, prop.name);
        printf("  Compute Capability:    %d.%d\n", prop.major, prop.minor);
        
        int cores_per_sm = getCoresPerSM(prop.major, prop.minor);
        if (cores_per_sm > 0) {
            printf("  CUDA Cores:           %d (per SM) * %d SMs = %d\n",
                   cores_per_sm,
                   prop.multiProcessorCount,
                   cores_per_sm * prop.multiProcessorCount);
        }
        
        printf("  Global Memory:        %.2f GB\n", 
               (double)prop.totalGlobalMem / (1024*1024*1024));
        printf("  SM Count:             %d\n", prop.multiProcessorCount);
        printf("  Clock Rate:           %.2f MHz\n", prop.clockRate/1000.0);
        printf("  Max Threads/Block:    %d\n", prop.maxThreadsPerBlock);
        printf("  Warp Size:            %d\n", prop.warpSize);
        printf("  L2 Cache Size:        %d bytes\n", prop.l2CacheSize);
        printf("  Concurrent Kernels:   %s\n", prop.concurrentKernels ? "Yes" : "No");
        printf("  ECC Support:          %s\n", prop.ECCEnabled ? "Enabled" : "Disabled");
        printf("  Integrated GPU:       %s\n", prop.integrated ? "Yes" : "No");
        printf("  Memory Clock Rate:    %.2f MHz\n", prop.memoryClockRate/1000.0);
        printf("  Memory Bus Width:     %d-bit\n", prop.memoryBusWidth);
        printf("  Peak Memory Bandwidth: %.2f GB/s\n\n",
               2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
    }

    return 0;
}
