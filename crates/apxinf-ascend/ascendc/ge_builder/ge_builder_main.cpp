// Standalone driver for ge_builder (C1-①): the same MatMul-pair chain as
// the ge_poc POC, but defined through the GENERIC FFI -- proves the layer
// reproduces the POC numbers -- plus the transposed-B variant (C1-②).
//
//   geb_main verify [trans] [matmul|v2]   parity vs host fp32 reference
//   geb_main bench  [trans] [matmul|v2]   prefix-gate_up shape, back-to-back
//
//   trans   : weights supplied physically transposed ([n,k]/[k,n]) with
//             transpose_x2/transpose_b=true -- the layout the engine's
//             NzCache pipeline produces; POC used untransposed [k,n]
//   matmul  : classic MatMul (default v2 = MatMulV2, whose infershape has
//             the rank gate; classic is the atc-classic offline path)
//
// Env overrides (bench only): GEB_M / GEB_K / GEB_N / GEB_PAIRS.
// Run from this directory (loads ./build/libge_builder.so).
#include "acl/acl.h"
#include "acl/acl_rt.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

extern "C" int geb_init(const char *soc_version);
extern "C" int64_t geb_model_begin(const char *graph_name);
extern "C" int geb_set_option(const char *key, const char *value);
extern "C" int geb_add_data(const char *name, int64_t index, const int64_t *dims, int32_t n_dims,
                            const char *dtype);
extern "C" int geb_add_op(const char *name, const char *type);
extern "C" int geb_set_input_desc(const char *op, const char *port, const int64_t *dims,
                                  int32_t n_dims, const char *dtype);
extern "C" int geb_set_output_desc(const char *op, const char *port, const int64_t *dims,
                                   int32_t n_dims, const char *dtype);
extern "C" int geb_set_attr_bool(const char *op, const char *attr, int32_t value);
extern "C" int geb_link(const char *dst_op, const char *dst_port, const char *src_op);
extern "C" int geb_graph_inputs(const char *const *names, int32_t n);
extern "C" int geb_graph_outputs(const char *const *names, int32_t n);
extern "C" int geb_model_build(int64_t handle);
extern "C" int32_t geb_num_inputs(int64_t h);
extern "C" int32_t geb_num_outputs(int64_t h);
extern "C" int64_t geb_input_size(int64_t h, int32_t idx);
extern "C" int32_t geb_input_dims(int64_t h, int32_t idx, int64_t *out_dims, int32_t cap);
extern "C" int geb_run(int64_t h, void *const *inputs, int32_t n_in, void *const *outputs,
                       int32_t n_out, void *stream);
extern "C" int geb_fini(void);

#define CHK(expr)                                                                        \
    do {                                                                                 \
        const int _rc = (expr);                                                          \
        if (_rc != 0) {                                                                  \
            std::fprintf(stderr, "%s -> rc=%d\n", #expr, _rc);                           \
            return _rc;                                                                  \
        }                                                                                \
    } while (0)

static void FillHalf(std::vector<uint16_t> &out, size_t n, std::mt19937 &rng) {
    out.resize(n);
    std::uniform_real_distribution<float> d(-0.25f, 0.25f);
    for (size_t i = 0; i < n; i++) {
        float v = d(rng);
        uint32_t f;
        std::memcpy(&f, &v, 4U);
        uint32_t sign = (f >> 16U) & 0x8000U;
        int32_t exp = static_cast<int32_t>((f >> 23U) & 0xFFU) - 127 + 15;
        uint32_t man = (f >> 13U) & 0x3FFU;
        if (exp <= 0) {
            out[i] = static_cast<uint16_t>(sign);
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
        f = sign;
    } else {
        f = sign | ((exp - 15U + 127U) << 23U) | (man << 13U);
    }
    float v;
    std::memcpy(&v, &f, 4U);
    return v;
}

static void *Upload(const std::vector<uint16_t> &h) {
    void *p = nullptr;
    if (aclrtMalloc(&p, h.size() * 2U, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
        std::fprintf(stderr, "malloc failed\n");
        std::exit(2);
    }
    (void)aclrtMemcpy(p, h.size() * 2U, h.data(), h.size() * 2U, ACL_MEMCPY_HOST_TO_DEVICE);
    return p;
}

// src [rows, cols] -> dst [cols, rows] (host fp16 transpose, NzCache layout)
static void Transpose(const std::vector<uint16_t> &src, std::vector<uint16_t> &dst, int64_t rows,
                      int64_t cols) {
    dst.assign(src.size(), 0);
    for (int64_t r = 0; r < rows; r++) {
        for (int64_t c = 0; c < cols; c++) {
            dst[static_cast<size_t>(c) * rows + r] = src[static_cast<size_t>(r) * cols + c];
        }
    }
}

static int32_t EnvInt(const char *name, int32_t fallback) {
    const char *v = getenv(name);
    return (v != nullptr) ? static_cast<int32_t>(std::atoi(v)) : fallback;
}

// the POC chain, defined through the generic FFI: x -> (mm1 -> mm2) x pairs
static int64_t BuildChain(int32_t pairs, int32_t m, int32_t k, int32_t n, bool trans, bool use_v2) {
    const char *mm_type = use_v2 ? "MatMulV2" : "MatMul";
    const char *attr_tb = use_v2 ? "transpose_x2" : "transpose_b";
    const char *attr_ta = use_v2 ? "transpose_x1" : "transpose_a";

    // physical weight descs: transposed ([n,k]/[k,n]) with transpose_b, or
    // straight ([k,n]/[n,k]) like the POC
    const std::vector<int64_t> dx = {m, k};
    const std::vector<int64_t> dy1 = {m, n};
    const std::vector<int64_t> dy = {m, k};
    const std::vector<int64_t> dw1 = trans ? std::vector<int64_t>{n, k} : std::vector<int64_t>{k, n};
    const std::vector<int64_t> dw2 = trans ? std::vector<int64_t>{k, n} : std::vector<int64_t>{n, k};

    const int64_t h = geb_model_begin("geb_matmul");
    if (h < 0) {
        return -1;
    }
    CHK(geb_add_data("x", 0, dx.data(), 2, "fp16"));
    CHK(geb_add_data("w1", 1, dw1.data(), 2, "fp16"));
    CHK(geb_add_data("w2", 2, dw2.data(), 2, "fp16"));

    std::string prev = "x";
    for (int32_t i = 0; i < pairs; i++) {
        const std::string mm1 = "mm1_" + std::to_string(i);
        const std::string mm2 = "mm2_" + std::to_string(i);
        CHK(geb_add_op(mm1.c_str(), mm_type));
        CHK(geb_set_input_desc(mm1.c_str(), "x1", dx.data(), 2, "fp16"));
        CHK(geb_set_input_desc(mm1.c_str(), "x2", dw1.data(), 2, "fp16"));
        CHK(geb_set_output_desc(mm1.c_str(), "y", dy1.data(), 2, "fp16"));
        CHK(geb_set_attr_bool(mm1.c_str(), attr_ta, 0));
        CHK(geb_set_attr_bool(mm1.c_str(), attr_tb, trans ? 1 : 0));
        CHK(geb_link(mm1.c_str(), "x1", prev.c_str()));
        CHK(geb_link(mm1.c_str(), "x2", "w1"));

        CHK(geb_add_op(mm2.c_str(), mm_type));
        CHK(geb_set_input_desc(mm2.c_str(), "x1", dy1.data(), 2, "fp16"));
        CHK(geb_set_input_desc(mm2.c_str(), "x2", dw2.data(), 2, "fp16"));
        CHK(geb_set_output_desc(mm2.c_str(), "y", dy.data(), 2, "fp16"));
        CHK(geb_set_attr_bool(mm2.c_str(), attr_ta, 0));
        CHK(geb_set_attr_bool(mm2.c_str(), attr_tb, trans ? 1 : 0));
        CHK(geb_link(mm2.c_str(), "x1", mm1.c_str()));
        CHK(geb_link(mm2.c_str(), "x2", "w2"));
        prev = mm2;
    }
    const char *ins[3] = {"x", "w1", "w2"};
    const char *outs[1] = {prev.c_str()};
    CHK(geb_graph_inputs(ins, 3));
    CHK(geb_graph_outputs(outs, 1));

    CHK(geb_set_option("input_format", "ND"));
    std::string shape = "x:" + std::to_string(m) + "," + std::to_string(k) +
                        ";w1:" + std::to_string(dw1[0]) + "," + std::to_string(dw1[1]) +
                        ";w2:" + std::to_string(dw2[0]) + "," + std::to_string(dw2[1]);
    CHK(geb_set_option("input_shape", shape.c_str()));
    CHK(geb_model_build(h));

    // exercise the introspection surface (the Rust side sizes buffers from it)
    std::fprintf(stderr, "[main] model io: n_in=%d n_out=%d\n", geb_num_inputs(h),
                 geb_num_outputs(h));
    for (int32_t i = 0; i < geb_num_inputs(h); i++) {
        int64_t dims[8] = {0};
        const int32_t nd = geb_input_dims(h, i, dims, 8);
        std::fprintf(stderr, "[main]   in[%d] size=%lld dims=", i,
                     static_cast<long long>(geb_input_size(h, i)));
        for (int32_t d = 0; d < nd; d++) {
            std::fprintf(stderr, "%s%lld", d == 0 ? "" : ",", static_cast<long long>(dims[d]));
        }
        std::fprintf(stderr, "\n");
    }
    return h;
}

int main(int argc, char **argv) {
    bool verify = false;
    bool trans = false;
    bool use_v2 = true;
    for (int i = 1; i < argc; i++) {
        if (std::strcmp(argv[i], "verify") == 0) verify = true;
        if (std::strcmp(argv[i], "trans") == 0) trans = true;
        if (std::strcmp(argv[i], "matmul") == 0) use_v2 = false;
        if (std::strcmp(argv[i], "v2") == 0) use_v2 = true;
    }
    int32_t m = verify ? 64 : 832;
    int32_t k = verify ? 512 : 2048;
    int32_t n = verify ? 256 : 32768;
    int32_t pairs = verify ? 1 : 8;
    if (!verify) {
        m = EnvInt("GEB_M", m);
        k = EnvInt("GEB_K", k);
        n = EnvInt("GEB_N", n);
        pairs = EnvInt("GEB_PAIRS", pairs);
    }

    if (aclInit(nullptr) != ACL_SUCCESS) {
        // already-inited is fine in some hosts; only abort on real failure
    }
    (void)aclrtSetDevice(0);
    aclrtStream stream = nullptr;
    (void)aclrtCreateStream(&stream);

    CHK(geb_init("Ascend310P3"));
    const int64_t h = BuildChain(pairs, m, k, n, trans, use_v2);
    if (h < 0) {
        std::fprintf(stderr, "BuildChain failed\n");
        return 1;
    }

    // logical weights [k,n]/[n,k]; trans mode uploads physical transposes
    std::mt19937 rng(7);
    std::vector<uint16_t> xh, w1h, w2h, w1up, w2up;
    FillHalf(xh, static_cast<size_t>(m) * k, rng);
    FillHalf(w1h, static_cast<size_t>(k) * n, rng);
    FillHalf(w2h, static_cast<size_t>(n) * k, rng);
    if (trans) {
        Transpose(w1h, w1up, k, n);  // [k,n] -> [n,k]
        Transpose(w2h, w2up, n, k);  // [n,k] -> [k,n]
    } else {
        w1up = w1h;
        w2up = w2h;
    }
    void *x = Upload(xh);
    void *w1 = Upload(w1up);
    void *w2 = Upload(w2up);
    void *y = nullptr;
    (void)aclrtMalloc(&y, static_cast<size_t>(m) * k * 2U, ACL_MEM_MALLOC_HUGE_FIRST);

    void *ins[3] = {x, w1, w2};
    CHK(geb_run(h, ins, 3, &y, 1, stream));
    (void)aclrtSynchronizeStream(stream);

    std::vector<uint16_t> yh(static_cast<size_t>(m) * k);
    (void)aclrtMemcpy(yh.data(), yh.size() * 2U, y, yh.size() * 2U, ACL_MEMCPY_DEVICE_TO_HOST);

    if (verify) {
        // host fp32 reference on the LOGICAL weights: y = x @ w1 @ w2
        // (identical math in trans mode; only the physical layout differs)
        std::vector<float> ref(static_cast<size_t>(m) * k, 0.0f);
        for (int32_t i = 0; i < m; i++) {
            for (int32_t o = 0; o < n; o++) {
                float acc = 0.0f;
                for (int32_t j = 0; j < k; j++) {
                    acc += HalfToF(xh[static_cast<size_t>(i) * k + j]) *
                           HalfToF(w1h[static_cast<size_t>(j) * n + o]);
                }
                for (int32_t o2 = 0; o2 < k; o2++) {
                    ref[static_cast<size_t>(i) * k + o2] +=
                        acc * HalfToF(w2h[static_cast<size_t>(o) * k + o2]);
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
            if (yh[i] == 0U) zeros++;
        }
        std::printf("verify: zeros=%zu/%zu\n", zeros, yh.size());
    } else {
        for (int w = 0; w < 3; w++) {
            CHK(geb_run(h, ins, 3, &y, 1, stream));
        }
        (void)aclrtSynchronizeStream(stream);
        std::vector<double> ts;
        for (int r = 0; r < 30; r++) {
            auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < 10; i++) {
                CHK(geb_run(h, ins, 3, &y, 1, stream));
            }
            (void)aclrtSynchronizeStream(stream);
            if (r >= 3) {
                std::chrono::duration<double, std::milli> dt = std::chrono::steady_clock::now() - t0;
                ts.push_back(dt.count() / (10.0 * static_cast<double>(pairs) * 2.0));
            }
        }
        std::sort(ts.begin(), ts.end());
        std::printf("bench: %.4f ms/matmul (median) m=%d n=%d trans=%d op=%s -- refs: GE-POC "
                    "4.745 / aclnn+ACLGraph 5.812\n",
                    ts[ts.size() / 2], m, n, trans ? 1 : 0, use_v2 ? "MatMulV2" : "MatMul");
    }

    (void)geb_fini();
    std::printf("GEB_MAIN_OK\n");
    return 0;
}
