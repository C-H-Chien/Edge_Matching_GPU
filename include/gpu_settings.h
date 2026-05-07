//> Needed for printf inside cudacheck macro (host and nvcc compilation)
#include <cstdio>
#include "definitions.h"

//> Macro definitions for GPU settings
#define DEVICE_ID (0)

#define COPY_DATA_FROM_HOST_TO_DEVICE_FOR_TESTING (true)

#define NUM_OF_THREADBLOCKS_SPATIAL_GRID_MAPPING (256)
#define GRID_CELL_SIZE (16)
#define FACTOR_OF_NUM_OF_LEFT_EDGES_FOR_STEREO_EDGE_MATCHES_AFTER_EPIPOLAR_RASTERIZATION (100)

//> Macros related to NCC filter
#define PATCH_RADIUS                    ((PATCH_SIZE / 2) - 1)
#define TOTAL_NUM_OF_PATCH_PIXELS       (PATCH_SIZE * PATCH_SIZE)
#define OFFSET_DIST                     ((PATCH_SIZE / 2) + 1)

//> CUDA error check
#define cudacheck( a )  do { \
    cudaError_t e = a; \
    if(e != cudaSuccess) { \
        printf("\033[1;31m"); \
        printf("Error in %s:%d %s (code %d)\n", __func__, __LINE__, cudaGetErrorString(e), (int)e); \
        printf("\033[0m"); \
    } \
} while(0)
