// Fused ada-rms-norm for PI-0.5 on Ascend 310P3 (AI Vector Core):
//   y[r,:] = x[r,:]/rms(x[r,:]) * scale[:] + shift[:]
// scale = (1+style[0:w]) is pre-folded host-side (cuda kernel semantics:
// normalization.cuh ada_rms_norm_bf16). Rows are independent -- an
// embarrassingly-parallel kernel, no cross-core sync.
//
// Row reduction is a fp32 cast + binary tree of vector Adds (8 trailing
// elements via scalar reads -- the skill-verified components only).
// Replaces the add_rms_norm(x,zeros)+mul+add composition that msprof
// pinned at 2 of 3 big-matrix kernels per ada-norm call (~400/inference).
// kernel_operator.h lives outside the interface/ include dir; pull in the
// same four leaves it wraps
#include "kernel_tpipe.h"
#include "kernel_tensor.h"
#include "kernel_type.h"
#include "kernel_operator_intf.h"
#include "ada_rms_norm_dynamic.h"

using namespace AscendC;

constexpr int32_t TILE_ROWS = 4;
constexpr int32_t MAX_COLS = 2048;
constexpr int32_t TILING_ELEMS = 8; // 5 int32 -> 8 (32B DataCopy alignment)

// device side has no libm and ccec lowers __builtin_sqrtf to a libm call
// at link time -- rsqrt via the Quake magic seed + Newton iterations
__aicore__ inline float AdaRsqrt(float x) {
    int32_t i = __builtin_bit_cast(int32_t, x);
    i = 0x5f3759df - (i >> 1);
    float y = __builtin_bit_cast(float, i);
    const float halfX = 0.5f * x;
    for (int k = 0; k < 4; k++) {
        y = y * (1.5f - halfX * y * y);
    }
    return y;
}

class KernelAdaRmsNorm {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR scale, GM_ADDR shift, GM_ADDR y, GM_ADDR tiling) {
        // tiling buffer must be allocated before any DataCopy (skill rule)
        pipe.InitBuffer(tilingBuf, TILING_ELEMS * sizeof(int32_t));

        GlobalTensor<int32_t> tilingGm;
        tilingGm.SetGlobalBuffer((__gm__ int32_t *)tiling, TILING_ELEMS);
        auto tLoc = tilingBuf.Get<int32_t>();
        DataCopy(tLoc, tilingGm, TILING_ELEMS);
        pipe_barrier(PIPE_V);

        rows_ = tLoc.GetValue(0);
        cols_ = tLoc.GetValue(1);
        rowsPerCore_ = tLoc.GetValue(2);
        tileRows_ = tLoc.GetValue(3);
        int32_t epsBits = tLoc.GetValue(4);
        eps_ = __builtin_bit_cast(float, epsBits);
        diag_ = tLoc.GetValue(5);

        xGm_.SetGlobalBuffer((__gm__ half *)x, static_cast<uint64_t>(rows_) * cols_);
        scaleGm_.SetGlobalBuffer((__gm__ half *)scale, cols_);
        shiftGm_.SetGlobalBuffer((__gm__ half *)shift, cols_);
        yGm_.SetGlobalBuffer((__gm__ half *)y, static_cast<uint64_t>(rows_) * cols_);

        // per-core constants + row scratch (UB budget ~92KB of 192KB)
        pipe.InitBuffer(scaleBuf, MAX_COLS * sizeof(half));
        pipe.InitBuffer(shiftBuf, MAX_COLS * sizeof(half));
        pipe.InitBuffer(sqBuf, MAX_COLS * sizeof(float));      // x^2 fp32
        pipe.InitBuffer(redBuf, (MAX_COLS / 2) * sizeof(float)); // tree-reduce partner
        pipe.InitBuffer(xrBuf, MAX_COLS * sizeof(half));       // x * rstd
        pipe.InitBuffer(xsBuf, MAX_COLS * sizeof(half));       // (x*rstd) * scale
        pipe.InitBuffer(xQue, 2, TILE_ROWS * MAX_COLS * sizeof(half));
        pipe.InitBuffer(yQue, 2, TILE_ROWS * MAX_COLS * sizeof(half));
    }

    __aicore__ inline void Process() {
        auto scaleLoc = scaleBuf.Get<half>();
        auto shiftLoc = shiftBuf.Get<half>();
        DataCopy(scaleLoc, scaleGm_, cols_);
        DataCopy(shiftLoc, shiftGm_, cols_);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);

        const int32_t coreId = GetBlockIdx();
        int32_t rowBegin = coreId * rowsPerCore_;
        int32_t rowEnd = rowBegin + rowsPerCore_;
        if (rowEnd > rows_) {
            rowEnd = rows_;
        }
        if (rowBegin >= rowEnd) {
            return;
        }

        auto sq = sqBuf.Get<float>();
        auto red = redBuf.Get<float>();
        auto xr = xrBuf.Get<half>();
        auto xs = xsBuf.Get<half>();

        for (int32_t row0 = rowBegin; row0 < rowEnd; row0 += tileRows_) {
            int32_t rowsThis = rowEnd - row0;
            if (rowsThis > tileRows_) {
                rowsThis = tileRows_;
            }
            int32_t elems = rowsThis * cols_;

            auto xLoc = xQue.AllocTensor<half>();
            DataCopy(xLoc, xGm_[static_cast<uint64_t>(row0) * cols_], elems);
            xQue.EnQue(xLoc);
            xLoc = xQue.DeQue<half>();

            auto yLoc = yQue.AllocTensor<half>();
            // NOTE: no pipe_barrier between the vector ops below -- they
            // share PIPE_V and execute in order; the per-row GetValue pin
            // at the end of the loop body is the only ordering dependency
            // (barriers here serialized the tree and dominated the cost)
            for (int32_t r = 0; r < rowsThis; r++) {
                auto xRow = xLoc[r * cols_];
                auto yRow = yLoc[r * cols_];

                // sum(x^2) in fp32 via a binary tree of vector adds
                Cast(sq, xRow, RoundMode::CAST_NONE, cols_);
                if (diag_ == 4) {
                    // y = half(sq[0]): probes ONLY the fp16->fp32 cast
                    AscendC::Duplicate(yRow, static_cast<half>(sq.GetValue(0)), cols_);
                    continue;
                }
                if (diag_ == 5) {
                    // y = half(sq[0] + sq[cols/2]): cast + one tree level
                    Add(red, sq, sq[cols_ / 2], cols_ / 2);
                    AscendC::Duplicate(yRow, static_cast<half>(red.GetValue(0)), cols_);
                    continue;
                }
                Mul(red, sq, sq, cols_); // square in fp32 (dst != src)
                // hardware ReduceSum is arch-gated (2201/3510/5102 only --
                // NOT on 310P3), so an Add halving tree is the only reduce
                bool inSq = false; // partial sums live in red after the mul
                int32_t n = cols_;
                while (n > 8) {
                    n /= 2;
                    auto src = inSq ? sq : red;
                    auto dst = inSq ? red : sq;
                    Add(dst, src, src[n], n);
                    inSq = !inSq;
                }
                auto s = inSq ? sq : red;
                if (diag_ == 6) {
                    float dsum = 0.0f;
                    for (int32_t i = 0; i < n; i++) {
                        dsum += s.GetValue(i);
                    }
                    AscendC::Duplicate(yRow, static_cast<half>(dsum), cols_);
                    continue;
                }
                float sum = 0.0f;
                for (int32_t i = 0; i < n; i++) {
                    sum += s.GetValue(i);
                }
                const float rstd = AdaRsqrt(sum / static_cast<float>(cols_) + eps_);
                if (diag_ == 1) {
                    // passthrough shift: probes the scale/shift MTE2 lane
                    // and the y write-back, nothing else
                    Add(yRow, shiftLoc, shiftLoc, cols_);
                    continue;
                }
                if (diag_ == 2) {
                    // passthrough x: probes the xQue lane end to end
                    Add(yRow, xRow, xRow, cols_);
                    continue;
                }
                if (diag_ == 3) {
                    // broadcast the row rstd: probes the fp32 cast +
                    // tree reduction + rsqrt in isolation
                    AscendC::Duplicate(yRow, static_cast<half>(rstd), cols_);
                    continue;
                }

                // y = x * rstd * scale + shift (independent buffers only)
                Muls(xr, xRow, static_cast<half>(rstd), cols_);
                Mul(xs, xr, scaleLoc, cols_);
                Add(yRow, xs, shiftLoc, cols_);
                // pin: without reads on BOTH scratch tensors the compiler
                // pipelines the next row's Cast/Mul over this row's
                // partially-read tree (skill TBuf cross-iteration trap)
                float _dep = sq.GetValue(0);
                _dep += red.GetValue(0);
                (void)_dep;
            }
            yQue.EnQue<half>(yLoc);
            yLoc = yQue.DeQue<half>();
            DataCopy(yGm_[static_cast<uint64_t>(row0) * cols_], yLoc, elems);
            xQue.FreeTensor(xLoc);
            yQue.FreeTensor(yLoc);
        }
    }

private:
    TPipe pipe;
    TBuf<TPosition::VECCALC> tilingBuf, scaleBuf, shiftBuf, sqBuf, redBuf, xrBuf, xsBuf;
    TQue<TPosition::VECIN, 2> xQue;
    TQue<TPosition::VECOUT, 2> yQue;
    GlobalTensor<half> xGm_, scaleGm_, shiftGm_, yGm_;
    int32_t rows_ = 0, cols_ = 0, rowsPerCore_ = 0, tileRows_ = TILE_ROWS, diag_ = 0;
    float eps_ = 1e-6f;
};

extern "C" __global__ __aicore__ void ada_rms_norm_dynamic(GM_ADDR x, GM_ADDR scale, GM_ADDR shift,
                                                           GM_ADDR y, GM_ADDR tiling) {
    KernelAdaRmsNorm op;
    op.Init(x, scale, shift, y, tiling);
    op.Process();
}
