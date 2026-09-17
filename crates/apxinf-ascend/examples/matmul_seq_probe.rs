//! Sequence probe: the action down matmul faults in-graph right after
//! geglu (gelu -> mul) with the mul's OUTPUT as its input, but every
//! isolated shape passes. Replicate the exact op sequence:
//! gelu -> mul -> (M-pad) matmul [50->64, 4096] x [4096, 1024].
//!   cargo run --example matmul_seq_probe --release -p apxinf-ascend
use apxinf_ascend::ops;
use apxinf_ascend::{AscendContext, AscendStream};
use half::f16;

fn main() {
    let ctx = AscendContext::new(0).expect("ctx");
    let stream = AscendStream::new().expect("stream");
    let tokens = 50i64;
    let width = 1024i64;
    let inter = 4096i64;

    let mut seed = 0xabcd1234u64;
    let mut rnd = move || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((seed >> 33) as i32 % 1000 - 500) as f32 / 500.0
    };

    // gate_up output [tokens, 2*inter] as the mul/gelu source
    let h_gate_up: Vec<f16> =
        (0..(tokens * inter * 2) as usize).map(|_| f16::from_f32(rnd())).collect();
    let d_gate_up = ctx.malloc(h_gate_up.len() * 2).unwrap();
    ctx.copy_h2d(&d_gate_up, bytemuck::cast_slice(&h_gate_up)).unwrap();

    let gate = ops::take_rows_fp16(&ctx, &stream, &d_gate_up, 0, tokens, inter).unwrap();
    let up = ops::take_rows_fp16(&ctx, &stream, &d_gate_up, tokens, tokens, inter).unwrap();
    let gate_g = ops::gelu_fp16(&ctx, &stream, &gate, &[tokens, inter], true).unwrap();
    let act = ops::mul_fp16(&ctx, &stream, &gate_g, &up, &[tokens, inter]).unwrap();
    stream.synchronize().unwrap();
    println!("geglu ok");

    // down weight [inter, width]
    let hb: Vec<f16> = (0..(inter * width) as usize).map(|_| f16::from_f32(rnd())).collect();
    let ht = ops::host_transpose(bytemuck::cast_slice(&hb), inter, width);
    let dbt = ctx.malloc(ht.len()).unwrap();
    ctx.copy_h2d(&dbt, &ht).unwrap();
    stream.synchronize().unwrap();
    println!("weight staged");

    let mut ref_row = vec![0f32; width as usize];
    {
        // CPU ref over the actual geglu result: read act back
        let mut back = vec![0u8; (tokens * inter * 2) as usize];
        ctx.copy_d2h(&act, &mut back).unwrap();
        let act_h: Vec<f16> = bytemuck::cast_slice(&back).to_vec();
        for j in 0..width as usize {
            ref_row[j] = (0..inter as usize)
                .map(|p| act_h[p].to_f32() * hb[p * width as usize + j].to_f32())
                .sum();
        }
    }
    match ops::matmul_b_t_fp16(&ctx, &stream, &act, [tokens, inter], &dbt, inter, width) {
        Ok(o) => {
            let mut back = vec![0u8; (tokens * width * 2) as usize];
            match stream.synchronize().and_then(|_| ctx.copy_d2h(&o, &mut back)) {
                Ok(_) => {
                    let got: Vec<f16> = bytemuck::cast_slice(&back).to_vec();
                    let mut mr = 0f32;
                    for j in 0..width as usize {
                        mr = mr.max((got[j].to_f32() - ref_row[j]).abs());
                    }
                    println!("post-geglu down matmul OK max_abs_row0={mr:.4}");
                }
                Err(e) => println!("post-geglu down matmul sync ERR {e:?}"),
            }
        }
        Err(e) => println!("post-geglu down matmul ERR {e:?}"),
    }

    // repeat the geglu + down matmul a few times (in-graph the loop runs
    // per denoise step; check iteration dependence)
    for it in 0..3 {
        let gate = ops::take_rows_fp16(&ctx, &stream, &d_gate_up, 0, tokens, inter).unwrap();
        let up = ops::take_rows_fp16(&ctx, &stream, &d_gate_up, tokens, tokens, inter).unwrap();
        let gate_g = ops::gelu_fp16(&ctx, &stream, &gate, &[tokens, inter], true).unwrap();
        let act = ops::mul_fp16(&ctx, &stream, &gate_g, &up, &[tokens, inter]).unwrap();
        match ops::matmul_b_t_fp16(&ctx, &stream, &act, [tokens, inter], &dbt, inter, width) {
            Ok(_) => match stream.synchronize() {
                Ok(_) => println!("iter {it} geglu+down OK"),
                Err(e) => println!("iter {it} sync ERR {e:?}"),
            },
            Err(e) => println!("iter {it} ERR {e:?}"),
        }
    }
    println!("SEQ_PROBE_DONE");
}
