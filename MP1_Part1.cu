#include "cuda_runtime.h"
#include <stdio.h>

int main(){

    int numdevices;
    cudaGetDeviceCount(&numdevices);
    printf("Cuda devices detected: %d\n", numdevices);

    for (int i = 0; i < numdevices; i++)
    {
        cudaDeviceProp cudaspecs;
        cudaGetDeviceProperties(&cudaspecs, i);

        int coresPerSM = 0;

        if (prop.major == 2) {
            coresPerSM = (prop.minor == 1) ? 48 : 32;
        } else if (prop.major == 3) {
            coresPerSM = 192;
        } else if (prop.major == 5) {
            coresPerSM = 128;
        } else if (prop.major == 6) {
            coresPerSM = (prop.minor == 1) ? 128 : 64;
        } else if (prop.major == 7) {
            coresPerSM = 64;
        } else if (prop.major == 8) {
            coresPerSM = 128;
        } else if (prop.major == 9) {
            coresPerSM = 128;
        } else {
            coresPerSM = 64;
        }
        int totalCores = specs.multiProcessorCount * coresPerSM;

        printf("Device %d: %s\n", i, cudaspecs.name);
        printf("Clock Rate: %.2f GHz", cudaspecs.clockRate / 1.0e6);
        printf("Number of Streaming Multiprocessors: %d\n", cudaspecs.multiProcessorCount);
        printf("Number of Cores: %d\n",totalCores);
        printf("Warp Size: %d\n", cudaspecs.warpSize);
        printf("Global Memory: %.2f GB\n", cudaprop.totalGlobalMem / (1024.0*1024.0*1024.0));
        printf("Total Constant Memory: %.2f KB\n", cudaspecs.totalConstMem / 1024.0);
        printf("Shared Memory per Block: %.2f KB\n", cudaspecs.sharedMemPerBlock / 1024.0);
        printf("Registers per Block: %d\n", cudaspecs.regsPerBlock);
        printf("Maximum Number of Threads per Block: %d\n", cudaspecs.maxThreadsPerBlock);
        printf("Maximum size of dimension of Each Block Dimension: (x = %d, y = %d, z = %d) \n", cudaspecs.maxThreadsDim[0], cudaspecs.maxThreadsDim[1], cudaspecs.maxThreadsDim[2]);
        printf("Maximum size of each dimension of a grid: (x = %d, y = %d, z = %d) \n", cudaspecs.maxGridSize[0], cudaspecs.maxGridSize[1], cudaspecs.maxGridSize[2]); 


    }
    
    return 0;

}