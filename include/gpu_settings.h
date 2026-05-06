//> Needed for printf inside cudacheck macro (host and nvcc compilation)
#include <cstdio>

//> Macro definitions for GPU settings
#define DEVICE_ID (0)

#define NUM_OF_THREADBLOCKS_SPATIAL_GRID_MAPPING (256)
#define GRID_CELL_SIZE (16)

//> CUDA error check
#define cudacheck( a )  do { \
    cudaError_t e = a; \
    if(e != cudaSuccess) { \
        printf("\033[1;31m"); \
        printf("Error in %s:%d %s\n", __func__, __LINE__, cudaGetErrorString(e)); \
        printf("\033[0m"); \
    } \
} while(0)
