//! Targeted probe for the action-expert down matmul fault:
//! [50->64, 4096] x [4096, 1024] (transposed-b path) faults with kernel
//! MatMulV2_ND_ND_FP16_FP32_false_true while bigger language matmuls
//! pass. Ladder the exact geometry and its neighbors, fresh process.
//!   cargo run --example matmul_action_down_probe --release -p apxinf-ascend
use apxinf_ascend::ops;
use apxinf_ascend::{AscendContext, AscendStream};
use half::f16;

fn main() {
    let ctx = AscendContext::new(0).expect("ctx");
    let stream = AscendStream::new().expect("stream");

    let mut seed = 0x1235u64;
    let mut rnd = move || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((seed >> 33) as i32 % 1000 - 500) as f32 / 500.0
    };

    let cases: [(i64, i64, i64); 13] = [
        (64, 4096, 1024),  // the faulting geometry (padded 50 -> 64)
        (50, 4096, 1024),  // the in-graph call: M-pad path (50 -> 64)
        (50, 1024, 8192),  // gate_up in-graph (pad path, passes in-graph)
        (50, 1024, 2560),  // qkv in-graph (pad path)
        (64, 1024, 1024),  // output-proj geometry (passes in-graph)
        (64, 1024, 8192),  // gate_up geometry (passes in-graph)
        (64, 1024, 2560),  // qkv geometry (passes in-graph)
        (16, 4096, 1024),
        (32, 4096, 1024),
        (128, 4096, 1024),
        (64, 2048, 1024),
        (64, 8192, 1024),
        (64, 4096, 2048),
    ];
    for (m, k, n) in cases {
        let ha: Vec<f16> = (0..(m * k) as usize).map(|_| f16::from_f32(rnd())).collect();
        let hb: Vec<f16> = (0..(k * n) as usize).map(|_| f16::from_f32(rnd())).collect();
        let da = ctx.malloc((m * k * 2) as usize).unwrap();
        ctx.copy_h2d(&da, bytemuck::cast_slice(&ha)).unwrap();
        let ht = ops::host_transpose(bytemuck::cast_slice(&hb), k, n);
        let dbt = ctx.malloc(ht.len()).unwrap();
        ctx.copy_h2d(&dbt, &ht).unwrap();
        stream.synchronize().unwrap();
        // row-0 CPU ref
        let mut ref_row = vec![0f32; n as usize];
        for j in 0..n as usize {
            ref_row[j] = (0..k as usize)
                .map(|p| ha[p].to_f32() * hb[p * n as usize + j].to_f32())
                .sum();
        }
        match ops::matmul_b_t_fp16(&ctx, &stream, &da, [m, k], &dbt, k, n) {
            Ok(o) => {
                let mut back = vec![0u8; (m * n * 2) as usize];
                match stream.synchronize().and_then(|_| ctx.copy_d2h(&o, &mut back)) {
                    Ok(_) => {
                        let got: Vec<f16> = bytemuck::cast_slice(&back).to_vec();
                        let mut mr = 0f32;
                        for j in 0..n as usize {
                            mr = mr.max((got[j].to_f32() - ref_row[j]).abs());
                        }
                        println!("m={m} k={k} n={n} OK max_abs_row0={mr:.4}");
                    }
                    Err(e) => println!("m={m} k={k} n={n} sync ERR {e:?}"),
                }
            }
            Err(e) => println!("m={m} k={k} n={n} ERR {e:?}"),
        }
    }
    println!("ACTION_DOWN_PROBE_DONE");
}
