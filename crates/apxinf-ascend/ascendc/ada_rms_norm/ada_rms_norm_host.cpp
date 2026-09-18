// Host wrapper for the fused ada-rms-norm kernel: a plain C ABI called
// from Rust (dlopen + libloading), mirroring the skill's kernellaunch
// pattern -- tiling uploaded to a persistent GM buffer, then the
// generated strong-typed launch entry. No stream sync (capture windows
// forbid it; single-stream ordering carries correctness).
#include "acl/acl.h"
#include "acl/acl_rt.h"
#include "aclrtlaunch_ada_rms_norm_dynamic.h"
#include "ada_rms_norm_dynamic.h"

#include <cstdio>
#include <cstring>

static void *g_tiling_buf = nullptr;

extern "C" int ada_rms_norm_init() {
    if (g_tiling_buf != nullptr) {
        return 0;
    }
    aclError ret = aclrtMalloc(&g_tiling_buf, 64, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
        fprintf(stderr, "[ada_rms_norm] aclrtMalloc tiling ret=%d\n", (int)ret);
        return -(int)ret;
    }
    return 0;
}

extern "C" int ada_rms_norm_run(void *x, void *scale, void *shift, void *y,
                                int32_t rows, int32_t cols, float eps,
                                int32_t num_cores, void *stream, int32_t diag) {
    if (g_tiling_buf == nullptr) {
        int rc = ada_rms_norm_init();
        if (rc != 0) {
            return rc;
        }
    }
    AdaRmsNormTilingData tiling;
    tiling.rows = rows;
    tiling.cols = cols;
    tiling.rowsPerCore = (rows + num_cores - 1) / num_cores;
    tiling.tileRows = 4;
    tiling.epsBits = 0;
    static_assert(sizeof(tiling.epsBits) == sizeof(eps), "eps bit reinterpret");
    memcpy(&tiling.epsBits, &eps, sizeof(eps));
    tiling.diag = diag;

    aclError ret = aclrtMemcpy(g_tiling_buf, sizeof(tiling), &tiling, sizeof(tiling),
                               ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) {
        fprintf(stderr, "[ada_rms_norm] tiling h2d ret=%d\n", (int)ret);
        return -(int)ret;
    }
    aclrtlaunch_ada_rms_norm_dynamic(num_cores, (aclrtStream)stream,
                                     (uint8_t *)x, (uint8_t *)scale, (uint8_t *)shift,
                                     (uint8_t *)y, (uint8_t *)g_tiling_buf);
    return 0;
}
