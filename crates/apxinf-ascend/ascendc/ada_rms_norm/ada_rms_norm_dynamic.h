#ifndef ADA_RMS_NORM_DYNAMIC_H
#define ADA_RMS_NORM_DYNAMIC_H

#include <cstdint>

#pragma pack(push, 8)
struct AdaRmsNormTilingData {
    int32_t rows;        // total rows of x [rows, cols]
    int32_t cols;        // row width in fp16 elements (8-multiple, <= 2048)
    int32_t rowsPerCore; // ceil(rows / numCores)
    int32_t tileRows;    // rows moved per DMA tile (TILE_ROWS)
    int32_t epsBits;     // eps as raw IEEE-754 bits
    int32_t diag;        // 0 = full fused path, 1 = y=shift passthrough,
                         // 2 = y=x passthrough (kernel bring-up probes)
};
#pragma pack(pop)

#endif
