//! ACLGraph capture/replay smoke from Rust (CANN 9.0.1+ container only).
//!
//! Mirrors the C probe v3 semantics: capture a memset on a side stream,
//! overwrite device memory, replay, verify the captured pattern is back.
//!
//!   cd crates/apxinf-ascend
//!   source /data/apxinf/rust_env.sh
//!   ASCEND_RT_VISIBLE_DEVICES=5 cargo run --example graph_smoke --release

use apxinf_ascend::graph::{self, CaptureMode};
use apxinf_ascend::{AscendContext, AscendStream};

fn main() {
    let ctx = AscendContext::new(0).expect("context");
    let stream = AscendStream::new().expect("stream");

    const BYTES: usize = 1024;
    let buf = ctx.malloc(BYTES).expect("malloc");

    // Warm the memset path on this stream, outside capture.
    ctx.memset_async(&buf, 0xAB, &stream).expect("warm memset");
    stream.synchronize().expect("warm sync");

    for mode in [CaptureMode::Global, CaptureMode::ThreadLocal, CaptureMode::Relaxed] {
        print!("---- capture {:?} ----\n", mode);
        graph::begin(&stream, mode).expect("capture begin");

        let payload = ctx.memset_async(&buf, 0xEE, &stream);
        if let Err(e) = payload {
            graph::abort(&stream).expect("abort");
            panic!("memset not capturable: {e:?}");
        }

        let g = graph::end(&stream).expect("capture end");

        // Corrupt device memory so only a true replay can restore 0xEE.
        ctx.memset_async(&buf, 0x11, &stream).expect("corrupt");
        stream.synchronize().expect("corrupt sync");

        g.replay_sync().expect("replay");

        let mut back = vec![0u8; BYTES];
        ctx.copy_d2h(&buf, &mut back).expect("d2h");
        let ok = back.iter().all(|&b| b == 0xEE);
        println!("replay restored captured pattern: {}", if ok { "YES" } else { "NO" });
        assert!(ok, "replay did not restore the captured memset pattern");
    }

    println!("GRAPH_SMOKE_OK");
}
