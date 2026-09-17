//! Workspace-size + allocation-layout probe for the wide-N cliff.
//! Hypotheses: (a) GetWorkspaceSize truncates at the cliff (kernel then
//! writes OOB -> MTE fault); (b) exact-size aclrtMalloc placements fail
//! where 2MB-aligned slab placements (torch allocator style) succeed.
//!   cargo run --example matmul_ws_probe --release -p apxinf-ascend
use apxinf_ascend::ffi;
use apxinf_ascend::{AclTensor, AscendContext, AscendStream};
use half::f16;

fn main() {
    let ctx = AscendContext::new(0).expect("ctx");
    let stream = AscendStream::new().expect("stream");
    let m = 8i64;
    let k = 2048i64;

    let mut seed = 0xfeedu64;
    let mut rnd = move || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((seed >> 33) as i32 % 1000 - 500) as f32 / 500.0
    };

    for n in [8192i64, 9216, 10240, 11264, 12288, 16384] {
        let hb: Vec<f16> = (0..(k * n) as usize).map(|_| f16::from_f32(rnd())).collect();
        let ha: Vec<f16> = (0..(m * k) as usize).map(|_| f16::from_f32(rnd())).collect();
        let ht = apxinf_ascend::ops::host_transpose(bytemuck::cast_slice(&hb), k, n); // [n, k]

        // CPU ref (first row only, cheap enough)
        let mut ref_row = vec![0f32; n as usize];
        for j in 0..n as usize {
            ref_row[j] = (0..k as usize)
                .map(|p| ha[p].to_f32() * hb[p * n as usize + j].to_f32())
                .sum();
        }

        // plan once, print ws_size
        let da = ctx.malloc((m * k * 2) as usize).unwrap();
        ctx.copy_h2d(&da, bytemuck::cast_slice(&ha)).unwrap();
        let dbt = ctx.malloc(ht.len()).unwrap();
        ctx.copy_h2d(&dbt, &ht).unwrap();
        let out = ctx.malloc((m * n * 2) as usize).unwrap();

        let ta = AclTensor::fp16_nd(&da, &[m, k]).unwrap();
        let dims = [k, n];
        let stride = [1, k];
        let tb_raw = unsafe {
            ffi::aclCreateTensor(
                dims.as_ptr(), 2, ffi::ACL_FLOAT16,
                stride.as_ptr(), 0, ffi::ACL_FORMAT_ND,
                std::ptr::null(), 0, dbt.as_ptr(),
            )
        };
        let tb_handle = tb_raw;
        let tout = AclTensor::fp16_nd(&out, &[m, n]).unwrap();

        let mut ws_size: u64 = 0;
        let mut executor: *mut std::ffi::c_void = std::ptr::null_mut();
        let plan = unsafe {
            ffi::aclnnMatmulGetWorkspaceSize(ta.handle(), tb_handle, tout.handle(), 2, &mut ws_size, &mut executor)
        };
        println!("N={n}: plan ret={plan} ws_size={ws_size} (0x{ws_size:x})");

        if plan != 0 {
            continue;
        }

        // run with exact malloc
        let ws = if ws_size > 0 { Some(ctx.malloc(ws_size as usize).unwrap()) } else { None };
        let r = unsafe {
            ffi::aclnnMatmul(
                ws.as_ref().map(|w| w.as_ptr()).unwrap_or(std::ptr::null_mut()),
                ws_size, executor, stream.handle(),
            )
        };
        if r != 0 {
            println!("  exact-alloc run ret={r}");
            continue;
        }
        let mut back = vec![0u8; (m * n * 2) as usize];
        match stream.synchronize().and_then(|_| ctx.copy_d2h(&out, &mut back)) {
            Ok(_) => {
                let got: Vec<f16> = bytemuck::cast_slice(&back).to_vec();
                let mut mr = 0f32;
                for j in 0..n as usize {
                    mr = mr.max((got[j].to_f32() - ref_row[j]).abs());
                }
                println!("  exact-alloc OK max_abs_row0={mr:.4}");
            }
            Err(e) => println!("  exact-alloc sync ERR {e:?}"),
        }
        unsafe { ffi::aclDestroyTensor(tb_handle) };
    }
    println!("WS_PROBE_DONE");
}
