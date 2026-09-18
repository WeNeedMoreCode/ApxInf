// Standalone driver for the GE static-OM POC (no Rust, runs anywhere the
// CANN python/tbe stack is complete -- the apxinf_npu container):
//   ge_poc_main verify   -> small-shape parity vs a host fp32 reference
//   ge_poc_main bench    -> prefix-gate_up shape, back-to-back executes
// Run from this directory (loads ./build/libge_matmul_poc.so).
#include "acl/acl.h"
#include "acl/acl_rt.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

extern "C" int ge_poc_init(const char *soc_version);
extern "C" int ge_poc_build(int32_t pairs, int32_t m, int32_t k, int32_t n);
extern "C" int ge_poc_run(void *x, void *w1, void *w2, void *y, void *stream);
extern "C" int ge_poc_fini();

static void FillHalf(std::vector<uint16_t> &out, size_t n, std::mt19937 &rng) {
    out.resize(n);
    std::uniform_real_distribution<float> d(-0.25f, 0.25f);
    for (size_t i = 0; i < n; i++) {
        // fp16 bits via manual conversion (half.h not needed for a probe)
        float v = d(rng);
        uint32_t f;
        std::memcpy(&f, &v, 4U);
        uint32_t sign = (f >> 16U) & 0x8000U;
        int32_t exp = static_cast<int32_t>((f >> 23U) & 0xFFU) - 127 + 15;
        uint32_t man = (f >> 13U) & 0x3FFU;
        if (exp <= 0) {
            out[i] = static_cast<uint16_t>(sign); // flush tiny to zero, fine for parity
        } else {
            out[i] = static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10U) | man);
        }
    }
}

static float HalfToF(uint16_t h) {
    uint32_t sign = (h & 0x8000U) << 16U;
    uint32_t exp = (h >> 10U) & 0x1FU;
    uint32_t man = h & 0x3FFU;
    uint32_t f;
    if (exp == 0U) {
        f = sign; // subnormals ~0 for this probe
    } else {
        f = sign | ((exp - 15U + 127U) << 23U) | (man << 13U);
    }
    float v;
    std::memcpy(&v, &f, 4U);
    return v;
}

static void *Upload(aclrtContext ctx, const std::vector<uint16_t> &h) {
    void *p = nullptr;
    (void)ctx;
    if (aclrtMalloc(&p, h.size() * 2U, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
        std::fprintf(stderr, "malloc failed\n");
        std::exit(2);
    }
    (void)aclrtMemcpy(p, h.size() * 2U, h.data(), h.size() * 2U, ACL_MEMCPY_HOST_TO_DEVICE);
    return p;
}

static int32_t EnvInt(const char *name, int32_t fallback) {
    const char *v = getenv(name);
    return (v != nullptr) ? static_cast<int32_t>(std::atoi(v)) : fallback;
}

int main(int argc, char **argv) {
    const bool verify = (argc > 1) && (std::strcmp(argv[1], "verify") == 0);
    int32_t m = verify ? 64 : 832;
    int32_t k = verify ? 512 : 2048;
    int32_t n = verify ? 256 : 32768;
    int32_t pairs = verify ? 1 : 8;
    if (!verify) {
        // bench shape override for the m-scaling fit (time = a*m + b separates
        // per-row compute from per-task dispatch tax)
        m = EnvInt("GE_POC_M", m);
        k = EnvInt("GE_POC_K", k);
        n = EnvInt("GE_POC_N", n);
        pairs = EnvInt("GE_POC_PAIRS", pairs);
    }

    if (aclInit(nullptr) != ACL_SUCCESS) {
        // already-inited is fine in some hosts; only abort on real failure
    }
    (void)aclrtSetDevice(0);
    aclrtStream stream = nullptr;
    (void)aclrtCreateStream(&stream);

    if (ge_poc_init("Ascend310P3") != 0) {
        std::fprintf(stderr, "ge_poc_init failed\n");
        return 1;
    }
    if (ge_poc_build(pairs, m, k, n) != 0) {
        std::fprintf(stderr, "ge_poc_build failed\n");
        return 1;
    }

    std::mt19937 rng(7);
    std::vector<uint16_t> xh, w1h, w2h;
    FillHalf(xh, static_cast<size_t>(m) * k, rng);
    FillHalf(w1h, static_cast<size_t>(k) * n, rng);
    FillHalf(w2h, static_cast<size_t>(n) * k, rng);
    void *x = Upload(nullptr, xh);
    void *w1 = Upload(nullptr, w1h);
    void *w2 = Upload(nullptr, w2h);
    void *y = nullptr;
    (void)aclrtMalloc(&y, static_cast<size_t>(m) * k * 2U, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ge_poc_run(x, w1, w2, y, stream) != 0) {
        std::fprintf(stderr, "ge_poc_run failed\n");
        return 1;
    }
    (void)aclrtSynchronizeStream(stream);

    std::vector<uint16_t> yh(static_cast<size_t>(m) * k);
    (void)aclrtMemcpy(yh.data(), yh.size() * 2U, y, yh.size() * 2U, ACL_MEMCPY_DEVICE_TO_HOST);

    if (verify) {
        // host fp32 reference: y = x @ w1 @ w2
        std::vector<float> ref(static_cast<size_t>(m) * k, 0.0f);
        for (int32_t i = 0; i < m; i++) {
            for (int32_t o = 0; o < n; o++) {
                float acc = 0.0f;
                for (int32_t j = 0; j < k; j++) {
                    acc += HalfToF(xh[static_cast<size_t>(i) * k + j]) * HalfToF(w1h[static_cast<size_t>(j) * n + o]);
                }
                for (int32_t o2 = 0; o2 < k; o2++) {
                    ref[static_cast<size_t>(i) * k + o2] += acc * HalfToF(w2h[static_cast<size_t>(o) * k + o2]);
                }
            }
        }
        float maxDiff = 0.0f;
        for (size_t i = 0; i < yh.size(); i++) {
            maxDiff = std::max(maxDiff, std::fabs(HalfToF(yh[i]) - ref[i]));
        }
        std::printf("verify: max_diff=%f (fp16 chain vs fp32 ref)\n", maxDiff);
        std::printf("verify: y[0..3]=%.4f,%.4f,%.4f,%.4f ref[0..3]=%.4f,%.4f,%.4f,%.4f\n", HalfToF(yh[0]),
                    HalfToF(yh[1]), HalfToF(yh[2]), HalfToF(yh[3]), ref[0], ref[1], ref[2], ref[3]);
        size_t zeros = 0;
        for (size_t i = 0; i < yh.size(); i++) {
            if (yh[i] == 0U) {
                zeros++;
            }
        }
        std::printf("verify: zeros=%zu/%zu\n", zeros, yh.size());
    } else {
        // bench: back-to-back executes
        for (int w = 0; w < 3; w++) {
            (void)ge_poc_run(x, w1, w2, y, stream);
        }
        (void)aclrtSynchronizeStream(stream);
        std::vector<double> ts;
        for (int r = 0; r < 30; r++) {
            auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < 10; i++) {
                (void)ge_poc_run(x, w1, w2, y, stream);
            }
            (void)aclrtSynchronizeStream(stream);
            if (r >= 3) {
                std::chrono::duration<double, std::milli> dt = std::chrono::steady_clock::now() - t0;
                ts.push_back(dt.count() / (10.0 * static_cast<double>(pairs) * 2.0));
            }
        }
        std::sort(ts.begin(), ts.end());
        std::printf("bench: %.4f ms/matmul (median) m=%d n=%d -- aclnn+ACLGraph ref 5.86\n",
                    ts[ts.size() / 2], m, n);
    }

    (void)ge_poc_fini();
    std::printf("GE_POC_MAIN_OK\n");
    return 0;
}
