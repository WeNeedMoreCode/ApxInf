//! M-dimension ladder: every failure so far is at m=8 (tiny M) x wide N.
//! Production M is hundreds of tokens; if the aclnnMatmul tiling bug is
//! tiny-M-specific, zero-padding rows is a complete workaround.
//!   cargo run --example matmul_m_probe --release -p apxinf-ascend
use apxinf_ascend::ops;
use apxinf_ascend::{AscendContext, AscendStream};
use half::f16;

fn main() {
    let ctx = AscendContext::new(0).expect("ctx");
    let stream = AscendStream::new().expect("stream");
    let k = 2048i64;

    let mut seed = 0xd00du64;
    let mut rnd = move || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((seed >> 33) as i32 % 1000 - 500) as f32 / 500.0
    };

    for n in [8192i64, 16384, 32768] {
        let hb: Vec<f16> = (0..(k * n) as usize).map(|_| f16::from_f32(rnd())).collect();
        let db = ctx.malloc((k * n * 2) as usize).unwrap();
        ctx.copy_h2d(&db, bytemuck::cast_slice(&hb)).unwrap();
        let ht = ops::host_transpose(bytemuck::cast_slice(&hb), k, n);
        let dbt = ctx.malloc(ht.len()).unwrap();
        ctx.copy_h2d(&dbt, &ht).unwrap();
        println!("N={n}:");
        for m in [8i64, 16, 32, 64, 128, 256, 812, 820, 828, 832] {
            let ha: Vec<f16> = (0..(m * k) as usize).map(|_| f16::from_f32(rnd())).collect();
            let da = ctx.malloc((m * k * 2) as usize).unwrap();
            ctx.copy_h2d(&da, bytemuck::cast_slice(&ha)).unwrap();
            // CPU ref for row 0
            let mut ref_row = vec![0f32; n as usize];
            for j in 0..n as usize {
                ref_row[j] = (0..k as usize)
                    .map(|p| ha[p].to_f32() * hb[p * n as usize + j].to_f32())
                    .sum();
            }
            // plain ND b
            match ops::matmul_fp16(&ctx, &stream, &da, [m, k], &db, [k, n]) {
                Ok(o) => {
                    let mut back = vec![0u8; (m * n * 2) as usize];
                    match stream.synchronize().and_then(|_| ctx.copy_d2h(&o, &mut back)) {
                        Ok(_) => {
                            let got: Vec<f16> = bytemuck::cast_slice(&back).to_vec();
                            let mut mr = 0f32;
                            for j in 0..n as usize {
                                mr = mr.max((got[j].to_f32() - ref_row[j]).abs());
                            }
                            println!("  m={m:3} plain   OK max_abs_row0={mr:.4}");
                        }
                        Err(e) => println!("  m={m:3} plain   sync ERR {e:?}"),
                    }
                }
                Err(e) => println!("  m={m:3} plain   ERR {e:?}"),
            }
            // transposed b
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
                            println!("  m={m:3} t-b     OK max_abs_row0={mr:.4}");
                        }
                        Err(e) => println!("  m={m:3} t-b     sync ERR {e:?}"),
                    }
                }
                Err(e) => println!("  m={m:3} t-b     ERR {e:?}"),
            }
        }
    }
    println!("M_PROBE_DONE");
}
