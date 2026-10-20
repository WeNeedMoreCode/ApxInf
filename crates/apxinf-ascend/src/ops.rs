//! Host-side weight-layout helpers for the Ascend backend.

/// Byte-level [rows, cols] -> [cols, rows] transpose of f16-LE data
/// (loader-native [k,n] weight rows into the engine's [n,k] GEMM layout).
pub fn host_transpose(host: &[u8], rows: i64, cols: i64) -> Vec<u8> {
    let src: Vec<u16> = host.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
    let mut out = vec![0u16; src.len()];
    for r in 0..rows as usize {
        for c in 0..cols as usize {
            out[c * rows as usize + r] = src[r * cols as usize + c];
        }
    }
    out.iter().flat_map(|&v| v.to_le_bytes()).collect()
}
