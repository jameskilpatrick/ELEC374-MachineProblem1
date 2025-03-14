#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>         // For memory cleanup
#include <math.h>           // For fabs (absolute value)
#include <time.h>           // For CPU timing with clock()


// matrix multiplication on GPU (from CPU) using CUDA kernels
__global__ void matMulKernelPart2(const float* M, const float* N, float* P, int width) {
    // Only one thread (0,0) does the entire multiplication
    if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
        for (int r = 0; r < width; r++) {
            for (int c = 0; c < width; c++) {
                float sum = 0.0f;
                for (int k = 0; k < width; k++) {
                    sum += M[r * width + k] * N[k * width + c];
                }
                P[r * width + c] = sum;
            }
        }
    }
}

__global__ void matMulKernelPart3(const float* M, const float* N, float* P, int width) {
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    int col = blockIdx.x*blockDim.x + threadIdx.x;
    if (row < width && col < width) {
        float sum = 0.0f;
        for (int k = 0; k < width; k++) {
            sum += M[row * width + k] * N[k * width + col];
        }
        P[row * width + col] = sum;
    }
}


// CPU implementation of matrix multiplication
void matMulCPU(const float* M, const float* N, float* P, int width) {
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < width; j++) {
            float sum = 0.0f;
            for (int k = 0; k < width; k++) {
                sum += M[i * width + k] * N[k * width + j];
            }
            P[i * width + j] = sum;
        }
    }
}

// Verify arrays A and B are within tolerance.
bool verifyArrays(const float* A, const float* B, int size, float tolerance) {
    for (int i = 0; i < size; i++) {
        if (fabs(A[i] - B[i]) > tolerance) {
            return false;
        }
    }
    return true;
}

// Helper to measure kernel execution time (ignoring data transfer)
void measureKernelTime(const float* dM, const float* dN, float* dP,
    int width, int blockWidth, float& kernelTimeMs)
{
    dim3 block(blockWidth, blockWidth);
    dim3 grid((width + blockWidth - 1) / blockWidth,
        (width + blockWidth - 1) / blockWidth);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    matMulKernelPart3<<<grid, block>>>(dM, dN, dP, width);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&kernelTimeMs, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main()
{
    const int sizes1[] = { 256, 512, 1024, 2048, 4096 };
    const int numSizes1 = sizeof(sizes1) / sizeof(int);

    printf("Experiment 1: H->D and D->H Transfer Times\n");
    printf("MatrixSizes: 256, 512, 1024, 2048, 4096\n\n");

    float hToDTimes[numSizes1];
    float dToHTimes[numSizes1];

    for (int idx = 0; idx < numSizes1; idx++) {
        int width = sizes1[idx];
        size_t bytes = width * (size_t)width * sizeof(float);

        float* hM = (float*)malloc(bytes);
        float* hN = (float*)malloc(bytes);
        float* hP = (float*)malloc(bytes);

        srand(0);
        for (int i = 0; i < width * width; i++) {
            hM[i] = (float)(rand() % 10);
            hN[i] = (float)(rand() % 10);
        }

        float* dM;
        float* dN;
        float* dP;
        cudaMalloc((void**)&dM, bytes);
        cudaMalloc((void**)&dN, bytes);
        cudaMalloc((void**)&dP, bytes);

        cudaEvent_t startHtoD, stopHtoD;
        cudaEventCreate(&startHtoD);
        cudaEventCreate(&stopHtoD);
        cudaEventRecord(startHtoD);
        cudaMemcpy(dM, hM, bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(dN, hN, bytes, cudaMemcpyHostToDevice);
        cudaEventRecord(stopHtoD);
        cudaEventSynchronize(stopHtoD);

        float timeHtoD = 0.0f;
        cudaEventElapsedTime(&timeHtoD, startHtoD, stopHtoD);
        hToDTimes[idx] = timeHtoD;

        cudaEvent_t startDtoH, stopDtoH;
        cudaEventCreate(&startDtoH);
        cudaEventCreate(&stopDtoH);
        cudaEventRecord(startDtoH);
        cudaMemcpy(hP, dM, bytes, cudaMemcpyDeviceToHost);
        cudaMemcpy(hP, dN, bytes, cudaMemcpyDeviceToHost);
        cudaEventRecord(stopDtoH);
        cudaEventSynchronize(stopDtoH);

        float timeDtoH = 0.0f;
        cudaEventElapsedTime(&timeDtoH, startDtoH, stopDtoH);
        dToHTimes[idx] = timeDtoH;

        cudaFree(dM);
        cudaFree(dN);
        cudaFree(dP);
        free(hM);
        free(hN);
        free(hP);
        cudaEventDestroy(startHtoD);
        cudaEventDestroy(stopHtoD);
        cudaEventDestroy(startDtoH);
        cudaEventDestroy(stopDtoH);
    }

    printf("Host->Device Transfer Times (ms) by Matrix Size:\n");
    for (int i = 0; i < numSizes1; i++) {
        printf("  Size %d x %d : %.3f ms\n", sizes1[i], sizes1[i], hToDTimes[i]);
    }
    printf("\nDevice->Host Transfer Times (ms) by Matrix Size:\n");
    for (int i = 0; i < numSizes1; i++) {
        printf("  Size %d x %d : %.3f ms\n", sizes1[i], sizes1[i], dToHTimes[i]);
    }
    printf("\n");


    const int sizes2[] = { 256, 512, 1024 };
    const int numSizes2 = sizeof(sizes2) / sizeof(int);

    printf("Experiment (2): CPU vs GPU (Single Block/Thread)\n");
    printf("MatrixSizes: 256, 512, 1024\n\n");

    for (int idx = 0; idx < numSizes2; idx++) {
        int width = sizes2[idx];
        size_t bytes = width * (size_t)width * sizeof(float);

        float* hM = (float*)malloc(bytes);
        float* hN = (float*)malloc(bytes);
        float* hP = (float*)malloc(bytes);
        float* hRef = (float*)malloc(bytes);

        srand(0);
        for (int i = 0; i < width * width; i++) {
            hM[i] = (float)(rand() % 10);
            hN[i] = (float)(rand() % 10);
        }

        float* dM, * dN, * dP;
        cudaMalloc((void**)&dM, bytes);
        cudaMalloc((void**)&dN, bytes);
        cudaMalloc((void**)&dP, bytes);

        cudaEvent_t startHtoD, stopHtoD;
        cudaEventCreate(&startHtoD);
        cudaEventCreate(&stopHtoD);
        cudaEventRecord(startHtoD);
        cudaMemcpy(dM, hM, bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(dN, hN, bytes, cudaMemcpyHostToDevice);
        cudaEventRecord(stopHtoD);
        cudaEventSynchronize(stopHtoD);

        float hToD_ms = 0.0f;
        cudaEventElapsedTime(&hToD_ms, startHtoD, stopHtoD);

        dim3 block(1, 1);
        dim3 grid(1, 1);

        cudaEvent_t startKernel, stopKernel;
        cudaEventCreate(&startKernel);
        cudaEventCreate(&stopKernel);

        cudaEventRecord(startKernel);
        matMulKernelPart2<<<grid, block>>>(dM, dN, dP, width);
        cudaEventRecord(stopKernel);
        cudaEventSynchronize(stopKernel);

        float gpuKernel_ms = 0.0f;
        cudaEventElapsedTime(&gpuKernel_ms, startKernel, stopKernel);

        cudaEvent_t startDtoH, stopDtoH;
        cudaEventCreate(&startDtoH);
        cudaEventCreate(&stopDtoH);
        cudaEventRecord(startDtoH);
        cudaMemcpy(hP, dP, bytes, cudaMemcpyDeviceToHost);
        cudaEventRecord(stopDtoH);
        cudaEventSynchronize(stopDtoH);

        float dToH_ms = 0.0f;
        cudaEventElapsedTime(&dToH_ms, startDtoH, stopDtoH);

        clock_t cpuStart = clock();
        matMulCPU(hM, hN, hRef, width);
        clock_t cpuEnd = clock();
        float cpu_ms = 1000.0f * (float)(cpuEnd - cpuStart) / CLOCKS_PER_SEC;

        float gpuNoXfer = gpuKernel_ms;
        float gpuWithXfer = gpuKernel_ms + hToD_ms + dToH_ms;

        bool pass = verifyArrays(hRef, hP, width * width, 1e-3f);

        printf("Matrix Size %d x %d\n", width, width);
        printf("CPU Time (ms): %.3f\n", cpu_ms);
        printf("GPU Time (1 block,1 thread) (ms): %.3f (NO Transfer), %.3f (WITH Transfer)\n",
            gpuNoXfer, gpuWithXfer);
        printf("Transfer Times: H->D = %.3f ms, D->H = %.3f ms\n", hToD_ms, dToH_ms);
        printf("%s\n\n", pass ? "Test PASSED" : "Test FAILED");

        cudaFree(dM);
        cudaFree(dN);
        cudaFree(dP);
        free(hM);
        free(hN);
        free(hP);
        free(hRef);
        cudaEventDestroy(startHtoD);
        cudaEventDestroy(stopHtoD);
        cudaEventDestroy(startKernel);
        cudaEventDestroy(stopKernel);
        cudaEventDestroy(startDtoH);
        cudaEventDestroy(stopDtoH);
    }


    printf("Experiment 3: Kernel Times vs. Block Width & Matrix Size\n");
    printf("MatrixSizes: 256, 512, 1024, 2048, 4096\n");
    printf("BlockWidth: 2,4,8,16,32\n\n");

    int blockWidths[5] = { 2, 4, 8, 16, 32 };
    int sizes3[] = { 256, 512, 1024, 2048, 4096 };
    int numSizes3 = 5;

    for (int bwIdx = 0; bwIdx < 5; bwIdx++) {
        int bWidth = blockWidths[bwIdx];
        for (int sIdx = 0; sIdx < numSizes3; sIdx++) {
            int width = sizes3[sIdx];
            size_t bytes = width * (size_t)width * sizeof(float);

            float* hM = (float*)malloc(bytes);
            float* hN = (float*)malloc(bytes);
            float* hP = (float*)malloc(bytes);

            srand(0);
            for (int i = 0; i < width * width; i++) {
                hM[i] = (float)(rand() % 10);
                hN[i] = (float)(rand() % 10);
            }

            float* dM, * dN, * dP;
            cudaMalloc((void**)&dM, bytes);
            cudaMalloc((void**)&dN, bytes);
            cudaMalloc((void**)&dP, bytes);

            cudaMemcpy(dM, hM, bytes, cudaMemcpyHostToDevice);
            cudaMemcpy(dN, hN, bytes, cudaMemcpyHostToDevice);

            float kernelMs = 0.0f;
            measureKernelTime(dM, dN, dP, width, bWidth, kernelMs);

            printf("Size %d x %d, BlockWidth %d : %f ms\n",
                width, width, bWidth, kernelMs);

            cudaFree(dM);
            cudaFree(dN);
            cudaFree(dP);
            free(hM);
            free(hN);
            free(hP);
        }
        printf("\n");
    }

    return 0;
}
