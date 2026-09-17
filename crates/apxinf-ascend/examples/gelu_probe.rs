//! Gelu state-corruption bisect: which prior op makes the next GeluV2
//! plan segfault? Runs gelu alone, then after add/euler/cat/pfa each.
use apxinf_ascend::ffi;
use apxinf_ascend::{AclTensor, AscendContext, AscendStream};

unsafe fn try_gelu_v2(ctx: &AscendContext, stream: &AscendStream, approximate: i64, tag: &str) -> i32 {
    let d = ctx.malloc(32).unwrap();
    let t = AclTensor::fp16_nd(&d, &[4, 4]).unwrap();
    let out = ctx.malloc(32).unwrap();
    let t_out = AclTensor::fp16_nd(&out, &[4, 4]).unwrap();
    let mut ws: u64 = 0;
    let mut ex: *mut std::ffi::c_void = std::ptr::null_mut();
    extern "C" {
        fn aclnnGeluV2GetWorkspaceSize(
            x: *mut std::ffi::c_void, approximate: i64, y: *mut std::ffi::c_void,
            ws: *mut u64, ex: *mut *mut std::ffi::c_void,
        ) -> i32;
        fn aclnnGeluV2(ws: *mut std::ffi::c_void, ws_size: u64, ex: *mut std::ffi::c_void, s: *mut std::ffi::c_void) -> i32;
    }
    let plan = aclnnGeluV2GetWorkspaceSize(t.handle(), approximate, t_out.handle(), &mut ws, &mut ex);
    println!("{tag} plan ret={plan} ws={ws}");
    if plan != 0 {
        return plan;
    }
    let run = aclnnGeluV2(std::ptr::null_mut(), ws, ex, stream.handle());
    println!("{tag} run ret={run}");
    let _ = stream.synchronize();
    run
}

fn main() {
    let ctx = AscendContext::new(0).expect("ctx");
    let stream = AscendStream::new().expect("stream");
    use apxinf_ascend::ops;
    let shape = [4i64, 4i64];
    let d1 = ctx.malloc(32).unwrap();
    let d2 = ctx.malloc(32).unwrap();

    println!("-- gelu alone x2 --");
    unsafe { try_gelu_v2(&ctx, &stream, 1, "gelu#1"); try_gelu_v2(&ctx, &stream, 1, "gelu#2"); }

    println!("-- add then gelu --");
    let _ = ops::add_fp16(&ctx, &stream, &d1, &d2, &shape).unwrap();
    let _ = ops::gelu_fp16(&ctx, &stream, &d1, &shape, true).unwrap();
    println!("add->gelu OK");

    println!("-- euler then gelu --");
    let _ = ops::euler_update_fp16(&ctx, &stream, &d1, &d2, 0.5, &shape).unwrap();
    let _ = ops::gelu_fp16(&ctx, &stream, &d1, &shape, true).unwrap();
    println!("euler->gelu OK");

    println!("-- cat then gelu --");
    let _ = ops::cat_fp16(&ctx, &stream, &[&d1, &d2], &[vec![4,4], vec![4,4]], 0, &[8,4]).unwrap();
    let _ = ops::gelu_fp16(&ctx, &stream, &d1, &shape, true).unwrap();
    println!("cat->gelu OK");

    println!("-- pfa then gelu --");
    let q = ctx.malloc(1024).unwrap();
    let _ = ops::prompt_flash_attention_fp16(&ctx, &stream, &q, &q, &q, [1,2,16,16], None).unwrap();
    let _ = ops::gelu_fp16(&ctx, &stream, &d1, &shape, true).unwrap();
    println!("pfa->gelu OK");

    println!("PROBE_DONE");
}
