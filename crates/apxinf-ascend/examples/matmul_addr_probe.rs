//! Address-pressure probe: the action down matmul faults only in-graph
//! (after ~1GB+ of live allocations) and every isolated replication
//! passes. Device-log args show buffers whose low 32 bits sit near 4GB
//! (0x...c0082028) -- suspect a u32 address wrap in the ND kernel's MTE
//! offsets. Push the allocator up with dummy blocks, print real device
//! pointers, then run the same geglu -> down matmul sequence.
//!   cargo run --example matmul_addr_probe --release -p apxinf-ascend
use apxinf_ascend::ops;
use apxinf_ascend::{AscendContext, AscendStream};
use half::f16;

fn main() {
    let ctx = AscendContext::new(0).expect("ctx");
    let stream = AscendStream::new().expect("stream");
    let tokens = 50i64;
    let width = 1024i64;
    let inter = 4096i64;

    let mut seed = 0x5555aaaau64;
    let mut rnd = move || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((seed >> 33) as i32 % 1000 - 500) as f32 / 500.0
    };

    // dummy pressure: 1.6 GB in 64 blocks, kept alive
    let pressure: Vec<_> = (0..64)
        .map(|i| {
            let b = ctx.malloc(25 * 1024 * 1024).unwrap();
            if i % 16 == 0 {
                println!("pressure block {i}: ptr={:p}", b.as_ptr());
            }
            b
        })
        .collect();
    println!("pressure allocated: {} MB", pressure.len() * 25);

    let h_gate_up: Vec<f16> =
        (0..(tokens * inter * 2) as usize).map(|_| f16::from_f32(rnd())).collect();
    let d_gate_up = ctx.malloc(h_gate_up.len() * 2).unwrap();
    ctx.copy_h2d(&d_gate_up, bytemuck::cast_slice(&h_gate_up)).unwrap();
    println!("gate_up buf ptr={:p}", d_gate_up.as_ptr());

    let hb: Vec<f16> = (0..(inter * width) as usize).map(|_| f16::from_f32(rnd())).collect();
    let ht = ops::host_transpose(bytemuck::cast_slice(&hb), inter, width);
    let dbt = ctx.malloc(ht.len()).unwrap();
    ctx.copy_h2d(&dbt, &ht).unwrap();
    println!("down weight ptr={:p}", dbt.as_ptr());

    for it in 0..4 {
        let gate = ops::take_rows_fp16(&ctx, &stream, &d_gate_up, 0, tokens, inter).unwrap();
        let up = ops::take_rows_fp16(&ctx, &stream, &d_gate_up, tokens, tokens, inter).unwrap();
        let gate_g = ops::gelu_fp16(&ctx, &stream, &gate, &[tokens, inter], true).unwrap();
        let act = ops::mul_fp16(&ctx, &stream, &gate_g, &up, &[tokens, inter]).unwrap();
        println!("iter {it} act ptr={:p}", act.as_ptr());
        match ops::matmul_b_t_fp16(&ctx, &stream, &act, [tokens, inter], &dbt, inter, width) {
            Ok(o) => {
                println!("iter {it} out ptr={:p}", o.as_ptr());
                match stream.synchronize() {
                    Ok(_) => println!("iter {it} OK"),
                    Err(e) => {
                        println!("iter {it} sync ERR {e:?}");
                        break;
                    }
                }
            }
            Err(e) => {
                println!("iter {it} ERR {e:?}");
                break;
            }
        }
    }
    println!("ADDR_PROBE_DONE");
}
