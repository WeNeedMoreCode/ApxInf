//! AscendBackend trait smoke: to_device -> matmul/add/silu/rms_norm via the
//! apxinf_core::Backend seam -> to_cpu, vs CPU references.
//!
//!   source /data/apxinf/rust_env.sh
//!   ASCEND_RT_VISIBLE_DEVICES=5 cargo run --example backend_smoke --release

use apxinf_ascend::AscendBackend;
use apxinf_core::{Backend, Tensor};

fn main() {
    let be = AscendBackend::new(0).expect("backend");
    assert_eq!(format!("{}", be.device()), "ascend:0");

    let (m, k, n) = (16usize, 32usize, 24usize);
    let a = Tensor::from_f32(vec![m, k], &(0..m * k).map(|i| (i as f32 % 7.0) - 3.0).collect::<Vec<_>>()).unwrap();
    let b = Tensor::from_f32(vec![k, n], &(0..k * n).map(|i| ((i * 3) as f32 % 5.0) - 2.0).collect::<Vec<_>>()).unwrap();

    let da = be.to_device(&a).unwrap();
    let db = be.to_device(&b).unwrap();
    assert_eq!(format!("{}", da.device()), "ascend:0");

    // matmul through the trait
    let dc = be.matmul(&da, &db).unwrap();
    be.synchronize().unwrap();
    let c = be.to_cpu(&dc).unwrap().to_f32_vec().unwrap();
    let mut max_rel = 0f32;
    for i in 0..m {
        for j in 0..n {
            let mut want = 0f32;
            for p in 0..k {
                want += a.to_f32_vec().unwrap()[i * k + p] * b.to_f32_vec().unwrap()[p * n + j];
            }
            let rel = (c[i * n + j] - want).abs() / want.abs().max(1.0);
            max_rel = max_rel.max(rel);
        }
    }
    println!("backend matmul max rel err {max_rel:.5}");
    assert!(max_rel < 0.01);

    // add + silu + rms_norm through the trait
    let gamma = Tensor::from_f32(vec![k], &vec![1.0f32; k]).unwrap();
    let dg = be.to_device(&gamma).unwrap();

    let dsum = be.add(&da, &da).unwrap(); // 2a
    let dact = be.silu(&dsum).unwrap();
    let dnorm = be.rms_norm(&dact, &dg, 1e-6).unwrap();
    be.synchronize().unwrap();

    let act = be.to_cpu(&dact).unwrap().to_f32_vec().unwrap();
    let norm = be.to_cpu(&dnorm).unwrap().to_f32_vec().unwrap();

    let mut worst_silu = 0f32;
    let mut rms_vals = Vec::new();
    for i in 0..m * k {
        let x = 2.0 * a.to_f32_vec().unwrap()[i];
        let want = x / (1.0 + (-x).exp());
        worst_silu = worst_silu.max((act[i] - want).abs());
        rms_vals.push(act[i]);
    }
    println!("backend silu max abs err {worst_silu:.5}");
    assert!(worst_silu < 0.01);

    // rms over each row of act
    let mut worst_norm = 0f32;
    for r in 0..m {
        let row = &rms_vals[r * k..(r + 1) * k];
        let ms = row.iter().map(|v| v * v).sum::<f32>() / k as f32;
        let rms = (ms + 1e-6).sqrt();
        for (c_i, &v) in row.iter().enumerate() {
            let want = v / rms;
            worst_norm = worst_norm.max((norm[r * k + c_i] - want).abs());
        }
    }
    println!("backend rms_norm max abs err {worst_norm:.5}");
    assert!(worst_norm < 0.02);

    // capture/replay through the trait (compile + run path)
    be.begin_capture().unwrap();
    let _ = be.silu(&dsum).unwrap();
    let g = be.end_capture().unwrap();
    g.replay().unwrap();
    be.synchronize().unwrap();
    println!("backend capture->replay ok");

    println!("BACKEND_SMOKE_OK");
}
