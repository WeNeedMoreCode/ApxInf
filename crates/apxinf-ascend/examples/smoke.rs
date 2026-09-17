//! Ascend ground-truth smoke: context + device-memory round-trip.
//!
//! Needs a real toolkit + chip (run in the apxinf_npu container):
//!   cd apxinf/crates/apxinf-ascend
//!   ASCEND_RT_VISIBLE_DEVICES=5 \
//!   LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/lib64:$LD_LIBRARY_PATH \
//!   cargo run --example smoke

use apxinf_ascend::AscendContext;

fn main() {
    let dev = std::env::args()
        .nth(1)
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(0);

    let ctx = AscendContext::new(dev).expect("context");
    println!("context up on device {}", ctx.device_id());

    const BYTES: usize = 4096;
    let buf = ctx.malloc(BYTES).expect("malloc");
    println!("device ptr {:p}, {} bytes", buf.as_ptr(), buf.len());

    let send: Vec<u8> = (0..BYTES as u32).map(|i| (i % 251) as u8).collect();
    ctx.copy_h2d(&buf, &send).expect("h2d");

    let mut back = vec![0u8; BYTES];
    ctx.copy_d2h(&buf, &mut back).expect("d2h");
    ctx.synchronize().expect("sync");

    let mismatches = send.iter().zip(&back).filter(|(a, b)| a != b).count();
    assert_eq!(mismatches, 0, "round-trip corrupted {mismatches} bytes");
    println!("round-trip clean ({BYTES} bytes, 0 mismatches)");
    println!("SMOKE_OK");
}
