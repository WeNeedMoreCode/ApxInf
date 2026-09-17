//! Wide-N entry-point bake-off. aclnnMatmul on 310P3 crashes at N>=12288
//! and reads wrong data on the transposed path for N in [4096, 8192],
//! while torch_npu computes the same shapes correctly -- so compare the
//! sibling entries aclnnMm (aten::mm mirror) and aclnnGemm (native
//! transB) at the cliff shapes in a fresh process.
//!   cargo run --example matmul_wide_n_probe --release -p apxinf-ascend
use apxinf_ascend::ops;
use apxinf_ascend::{AscendContext, AscendStream};
use half::f16;

fn main() {
    let ctx = AscendContext::new(0).expect("ctx");
    let stream = AscendStream::new().expect("stream");
    let m = 8i64;
    let k = 2048i64;

    let mut seed = 0x5eedu64;
    let mut rnd = move || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((seed >> 33) as i32 % 1000 - 500) as f32 / 500.0
    };
    let ha: Vec<f16> = (0..(m * k) as usize).map(|_| f16::from_f32(rnd())).collect();
    let da = ctx.malloc((m * k * 2) as usize).unwrap();
    ctx.copy_h2d(&da, bytemuck::cast_slice(&ha)).unwrap();

    let check = |tag: &str, n: i64, hb: &[f16], out: apxinf_ascend::DeviceBuffer| {
        let mut back = vec![0u8; (m * n * 2) as usize];
        // sync BEFORE d2h: synchronous aclrtMemcpy does not wait for
        // pending async kernels on our stream (race -> stale reads,
        // the "silently wrong data" ghost of 2026-09-18).
        if let Err(e) = stream.synchronize().and_then(|_| ctx.copy_d2h(&out, &mut back)) {
            println!("  {tag} N={n} sync/readback ERR {e:?}");
            return;
        }
        let got: Vec<f16> = bytemuck::cast_slice(&back).to_vec();
        let mut mr = 0f32;
        for i in 0..m as usize {
            for j in 0..n as usize {
                let r = (got[i * n as usize + j].to_f32()
                    - (0..k as usize)
                        .map(|p| ha[i * k as usize + p].to_f32() * hb[p * n as usize + j].to_f32())
                        .sum::<f32>())
                    .abs();
                mr = mr.max(r);
            }
        }
        println!("  {tag} N={n} OK, max abs {mr:.4}");
    };

    for n in [2560i64, 8192, 12288, 16384] {
        let hb: Vec<f16> = (0..(k * n) as usize).map(|_| f16::from_f32(rnd())).collect();
        let db = ctx.malloc((k * n * 2) as usize).unwrap();
        ctx.copy_h2d(&db, bytemuck::cast_slice(&hb)).unwrap();
        let ht = ops::host_transpose(bytemuck::cast_slice(&hb), k, n); // [n, k]
        let dbt = ctx.malloc(ht.len()).unwrap();
        ctx.copy_h2d(&dbt, &ht).unwrap();
        stream.synchronize().unwrap();
        println!("N={n}:");
        match ops::mm_fp16(&ctx, &stream, &da, [m, k], &db, [k, n]) {
            Ok(o) => check("aclnnMm plain  ", n, &hb, o),
            Err(e) => println!("  aclnnMm plain   N={n} ERR {e:?}"),
        }
        match ops::mm_b_t_fp16(&ctx, &stream, &da, [m, k], &dbt, k, n) {
            Ok(o) => check("aclnnMm t-b    ", n, &hb, o),
            Err(e) => println!("  aclnnMm t-b     N={n} ERR {e:?}"),
        }
        match ops::gemm_b_t_fp16(&ctx, &stream, &da, [m, k], &dbt, k, n) {
            Ok(o) => check("aclnnGemm tB=1 ", n, &hb, o),
            Err(e) => println!("  aclnnGemm tB=1  N={n} ERR {e:?}"),
        }
    }
    println!("WIDE_N_DONE");
}
