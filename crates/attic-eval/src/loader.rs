//! SFNN-1536 network-file (`nn.bin`) parsing and validation.
//!
//! Ported from the Rust NNUE reference implementation's `loader.rs`. The
//! serialized format's C++ ground truth is
//! `eval/nnue/evaluate_nnue.cpp`
//! (`ReadHeader` / `ReadParameters`) plus `nnue_common.h` and the layer headers
//! reached from `architectures/sfnn-1536.h`.
//!
//! Validation is reference-faithful to `374bdd72`, which deliberately lags the
//! submodule pin (`76d58ef`): the bucket / shard / small-FT architectures
//! upstream added in that span are out of scope. The SFNN-1536 serialization
//! itself is compatible across the two — the `76d58ef` reference loads the same
//! `nn.bin` and reproduces every eval fixture bit-identically — so the rules
//! below still describe the pin for the architecture this engine loads:
//! - The **version word** (`ReadHeader`) is a HARD failure on mismatch — the
//!   file is a different serialization format, so parsing cannot continue.
//! - The **file-level hash**, the **feature-transformer hash**, and each
//!   **layer-stack hash** are only WARNINGS on mismatch; the load continues and
//!   the parameters are read as usual (`LoadAndShare` / `Detail::ReadParameters`).
//!   These hashes are topology-derived, but the reference tolerates old files, so
//!   we do too — the warnings are surfaced as `info string` lines during
//!   `isready`.
//! - The **architecture string** is read (length-prefixed) but NEVER compared; it
//!   appears only inside the file-hash warning, rendered lossily so non-UTF-8
//!   bytes never fail the load.
//! - **Structural failures stay hard**: short reads, a bad LEB128 magic, an
//!   out-of-range value, and trailing bytes after the last stack all fail the
//!   load (`ReadHeader`/`ReadParameters` returning a read error, and the final
//!   EOF check).

use std::path::Path;

use crate::types::{NetDims, NetHeader, NnueError, NnueNetwork, NnueNetworkBuilder};

const NNUE_VERSION: u32 = 0x7AF3_2F16;
const NNUE_HASH_VALUE: u32 = 0x3C20_3B32;
const FT_HASH: u32 = 0x5F13_4AB8;
const NET_HASH: u32 = 0x6333_718A;
const ARCH_STRING: &str = "ModelType=SFNNWithoutPsqt;Features=HalfKA_hm(Friend)[73305->1536x2],Network=AffineTransform[1<-32](ClippedReLU[32](AffineTransform[32<-15](ClippedReLU[15](AffineTransform[15<-3072](InputSlice[3072(0:3072)]))))){LayerStack=9}";
const LEB128_MAGIC: &[u8; 17] = b"COMPRESSED_LEB128";

/// Warning body emitted (via the driver's `info string` sink) when a
/// feature-transformer or layer-stack hash does not match — mirrors the
/// reference `Detail::ReadParameters` (`evaluate_nnue.cpp`), spacing
/// and all.
const SECTION_HASH_WARNING: &str = "Warning : nn.bin hash mismatch.";

/// Reads and validates the SFNN-1536 network file at `path`, discarding any
/// non-fatal warnings. Use [`load_network_with_warnings`] to surface them.
pub fn load_network(path: &Path) -> Result<NnueNetwork, NnueError> {
    load_network_with_warnings(path).map(|(net, _warnings)| net)
}

/// Reads and validates the SFNN-1536 network file at `path`, returning the
/// network together with any non-fatal warning bodies (hash mismatches). The
/// caller surfaces each as an `info string` line; an empty vector means a clean
/// load. Structural problems and a version mismatch still fail with an error.
pub fn load_network_with_warnings(path: &Path) -> Result<(NnueNetwork, Vec<String>), NnueError> {
    let bytes = std::fs::read(path).map_err(|e| NnueError::Io {
        path: path.display().to_string(),
        source: e,
    })?;
    let sha256 = sha256::digest(&bytes);
    parse_from_bytes(&bytes, &NetDims::STANDARD, sha256)
}

fn parse_from_bytes(
    bytes: &[u8],
    dims: &NetDims,
    sha256: [u8; 32],
) -> Result<(NnueNetwork, Vec<String>), NnueError> {
    let mut warnings = Vec::new();
    let mut reader = ByteReader::new(bytes);
    let header = read_header(&mut reader, &mut warnings)?;
    // The whole parameter set is filled *in place* into one large-page arena:
    // the builder allocates it up front and hands out mutable views, so the
    // ~215 MiB `ft_weights` is decoded straight into its final home with no big
    // temporary, and every array shares one backing allocation.
    let mut builder = NnueNetworkBuilder::with_dims(header, sha256, dims);

    // The feature-transformer hash is a warning, not a hard error: the reference
    // still reads the parameters after `Detail::ReadParameters` logs the mismatch.
    let ft_hash = reader.read_u32_le()?;
    if ft_hash != FT_HASH {
        warnings.push(SECTION_HASH_WARNING.to_string());
    }
    read_leb128_i16_into(&mut reader, builder.ft_biases_mut())?;
    scale_ft_i16_x2_in_place(builder.ft_biases_mut())?;
    read_leb128_i16_into(&mut reader, builder.ft_weights_mut())?;
    scale_ft_i16_x2_in_place(builder.ft_weights_mut())?;
    for i in 0..dims.layer_stacks {
        read_network_block(&mut reader, &mut builder, i, &mut warnings)?;
    }
    reader.assert_eof()?;
    Ok((builder.build(), warnings))
}

fn read_header(
    reader: &mut ByteReader,
    warnings: &mut Vec<String>,
) -> Result<NetHeader, NnueError> {
    let version = reader.read_u32_le()?;
    if version != NNUE_VERSION {
        // `ReadHeader` (evaluate_nnue.cpp): a version mismatch is a HARD
        // failure with this exact message shape.
        return Err(NnueError::InvalidFormat {
            reason: format!(
                "NNUE header version mismatch: expected {} got {}",
                NNUE_VERSION, version
            ),
        });
    }
    let hash = reader.read_u32_le()?;
    let arch_size = reader.read_u32_le()?;
    // The architecture string is read but NEVER compared (evaluate_nnue.cpp): it
    // only feeds the file-hash warning below. `read_slice` already bounds the
    // length against the file (a short file fails structurally), and non-UTF-8 is
    // rendered lossily rather than rejected.
    let arch_bytes = reader.read_slice(arch_size as usize)?;
    let arch_id = String::from_utf8_lossy(arch_bytes).into_owned();
    // `LoadAndShare` (evaluate_nnue.cpp): a file-level hash mismatch is a
    // WARNING; the load continues. The message names the in-file and expected
    // architecture strings.
    if hash != NNUE_HASH_VALUE {
        warnings.push(format!(
            "Warning: NNUE hash mismatch: expected {} got {} arch_in_file={} arch_expected={}",
            NNUE_HASH_VALUE, hash, arch_id, ARCH_STRING
        ));
    }
    Ok(NetHeader {
        version,
        hash,
        arch_id,
    })
}

/// Decode one signed-LEB128 block straight into `out` (its length is the value
/// count). The target is a mutable view of the arena, so the ~215 MiB
/// `ft_weights` block is filled in place with no intermediate copy.
fn read_leb128_i16_into(reader: &mut ByteReader, out: &mut [i16]) -> Result<(), NnueError> {
    let count = out.len();
    let magic = reader.read_slice(LEB128_MAGIC.len())?;
    if magic != LEB128_MAGIC {
        return Err(NnueError::InvalidFormat {
            reason: format!(
                "expected LEB128 magic {:?}, got {:?}",
                std::str::from_utf8(LEB128_MAGIC).unwrap_or("<non-utf8>"),
                String::from_utf8_lossy(magic),
            ),
        });
    }
    let bytes_left = reader.read_u32_le()? as usize;
    // Worst-case signed-LEB128 for an i16 is 3 bytes; bounds the allocation against bad input.
    let upper_bound = count.saturating_mul(3);
    if bytes_left > upper_bound {
        return Err(NnueError::InvalidFormat {
            reason: format!(
                "LEB128 bytes_left {} exceeds upper bound {} for {} i16 values",
                bytes_left, upper_bound, count
            ),
        });
    }
    let payload = reader.read_slice(bytes_left)?;
    let mut pos = 0usize;
    for slot in out.iter_mut() {
        let v = read_signed_leb128(payload, &mut pos)?;
        if !(i16::MIN as i64..=i16::MAX as i64).contains(&v) {
            return Err(NnueError::InvalidFormat {
                reason: format!("LEB128 value {} out of i16 range", v),
            });
        }
        *slot = v as i16;
    }
    if pos != payload.len() {
        return Err(NnueError::InvalidFormat {
            reason: format!(
                "LEB128 block has {} unused bytes after {} values",
                payload.len() - pos,
                count
            ),
        });
    }
    Ok(())
}

fn scale_ft_i16_x2_in_place(values: &mut [i16]) -> Result<(), NnueError> {
    for slot in values.iter_mut() {
        let scaled = (*slot as i32) * 2;
        if !(i16::MIN as i32..=i16::MAX as i32).contains(&scaled) {
            return Err(NnueError::InvalidFormat {
                reason: format!(
                    "feature-transformer value {} out of range: ×2 scale overflows i16 (pre-scale bound is [-16_384, 16_383])",
                    slot
                ),
            });
        }
        *slot = scaled as i16;
    }
    Ok(())
}

fn read_signed_leb128(bytes: &[u8], pos: &mut usize) -> Result<i64, NnueError> {
    let mut result: i64 = 0;
    let mut shift: u32 = 0;
    loop {
        if *pos >= bytes.len() {
            return Err(NnueError::InvalidFormat {
                reason: "LEB128 value truncated".to_string(),
            });
        }
        let byte = bytes[*pos];
        *pos += 1;
        if shift >= 64 {
            return Err(NnueError::InvalidFormat {
                reason: "LEB128 value exceeds 64 bits".to_string(),
            });
        }
        result |= ((byte & 0x7F) as i64).wrapping_shl(shift);
        shift += 7;
        if byte & 0x80 == 0 {
            if shift < 64 && (byte & 0x40) != 0 {
                result |= (-1i64).wrapping_shl(shift);
            }
            return Ok(result);
        }
    }
}

fn read_network_block(
    reader: &mut ByteReader,
    builder: &mut NnueNetworkBuilder,
    stack: usize,
    warnings: &mut Vec<String>,
) -> Result<(), NnueError> {
    // Each layer-stack hash is a warning, not a hard error (the reference
    // `Detail::ReadParameters` logs and reads on).
    let net_hash = reader.read_u32_le()?;
    if net_hash != NET_HASH {
        warnings.push(SECTION_HASH_WARNING.to_string());
    }
    // The six arrays are filled directly into their arena sub-slices, in the
    // file's fc_0/fc_1/fc_2 order.
    read_i32_into(reader, builder.fc_0_biases_mut(stack))?;
    read_i8_into(reader, builder.fc_0_weights_mut(stack))?;
    read_i32_into(reader, builder.fc_1_biases_mut(stack))?;
    read_i8_into(reader, builder.fc_1_weights_mut(stack))?;
    read_i32_into(reader, builder.fc_2_biases_mut(stack))?;
    read_i8_into(reader, builder.fc_2_weights_mut(stack))?;
    Ok(())
}

fn read_i32_into(reader: &mut ByteReader, out: &mut [i32]) -> Result<(), NnueError> {
    let bytes = reader.read_slice(out.len() * 4)?;
    for (i, slot) in out.iter_mut().enumerate() {
        let chunk = &bytes[i * 4..i * 4 + 4];
        *slot = i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
    }
    Ok(())
}

fn read_i8_into(reader: &mut ByteReader, out: &mut [i8]) -> Result<(), NnueError> {
    let bytes = reader.read_slice(out.len())?;
    for (i, slot) in out.iter_mut().enumerate() {
        *slot = bytes[i] as i8;
    }
    Ok(())
}

struct ByteReader<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> ByteReader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }

    fn read_slice(&mut self, n: usize) -> Result<&'a [u8], NnueError> {
        let end = self
            .pos
            .checked_add(n)
            .ok_or_else(|| NnueError::InvalidFormat {
                reason: "read offset overflows usize".to_string(),
            })?;
        if end > self.bytes.len() {
            return Err(NnueError::SizeMismatch {
                expected: end,
                got: self.bytes.len(),
            });
        }
        let s = &self.bytes[self.pos..end];
        self.pos = end;
        Ok(s)
    }

    fn read_u32_le(&mut self) -> Result<u32, NnueError> {
        let s = self.read_slice(4)?;
        Ok(u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
    }

    fn assert_eof(&self) -> Result<(), NnueError> {
        if self.pos != self.bytes.len() {
            Err(NnueError::SizeMismatch {
                expected: self.pos,
                got: self.bytes.len(),
            })
        } else {
            Ok(())
        }
    }
}

mod sha256 {
    const K: [u32; 64] = [
        0x428a_2f98,
        0x7137_4491,
        0xb5c0_fbcf,
        0xe9b5_dba5,
        0x3956_c25b,
        0x59f1_11f1,
        0x923f_82a4,
        0xab1c_5ed5,
        0xd807_aa98,
        0x1283_5b01,
        0x2431_85be,
        0x550c_7dc3,
        0x72be_5d74,
        0x80de_b1fe,
        0x9bdc_06a7,
        0xc19b_f174,
        0xe49b_69c1,
        0xefbe_4786,
        0x0fc1_9dc6,
        0x240c_a1cc,
        0x2de9_2c6f,
        0x4a74_84aa,
        0x5cb0_a9dc,
        0x76f9_88da,
        0x983e_5152,
        0xa831_c66d,
        0xb003_27c8,
        0xbf59_7fc7,
        0xc6e0_0bf3,
        0xd5a7_9147,
        0x06ca_6351,
        0x1429_2967,
        0x27b7_0a85,
        0x2e1b_2138,
        0x4d2c_6dfc,
        0x5338_0d13,
        0x650a_7354,
        0x766a_0abb,
        0x81c2_c92e,
        0x9272_2c85,
        0xa2bf_e8a1,
        0xa81a_664b,
        0xc24b_8b70,
        0xc76c_51a3,
        0xd192_e819,
        0xd699_0624,
        0xf40e_3585,
        0x106a_a070,
        0x19a4_c116,
        0x1e37_6c08,
        0x2748_774c,
        0x34b0_bcb5,
        0x391c_0cb3,
        0x4ed8_aa4a,
        0x5b9c_ca4f,
        0x682e_6ff3,
        0x748f_82ee,
        0x78a5_636f,
        0x84c8_7814,
        0x8cc7_0208,
        0x90be_fffa,
        0xa450_6ceb,
        0xbef9_a3f7,
        0xc671_78f2,
    ];

    const INIT: [u32; 8] = [
        0x6a09_e667,
        0xbb67_ae85,
        0x3c6e_f372,
        0xa54f_f53a,
        0x510e_527f,
        0x9b05_688c,
        0x1f83_d9ab,
        0x5be0_cd19,
    ];

    pub fn digest(data: &[u8]) -> [u8; 32] {
        let mut h = INIT;
        let bit_len: u64 = (data.len() as u64).wrapping_mul(8);

        let full_blocks = data.len() / 64;
        for i in 0..full_blocks {
            let mut block = [0u8; 64];
            block.copy_from_slice(&data[i * 64..(i + 1) * 64]);
            compress(&mut h, &block);
        }

        let remainder = &data[full_blocks * 64..];
        let mut tail = [0u8; 128];
        tail[..remainder.len()].copy_from_slice(remainder);
        tail[remainder.len()] = 0x80;
        if remainder.len() < 56 {
            tail[56..64].copy_from_slice(&bit_len.to_be_bytes());
            let mut block = [0u8; 64];
            block.copy_from_slice(&tail[..64]);
            compress(&mut h, &block);
        } else {
            tail[120..128].copy_from_slice(&bit_len.to_be_bytes());
            let mut b1 = [0u8; 64];
            b1.copy_from_slice(&tail[..64]);
            compress(&mut h, &b1);
            let mut b2 = [0u8; 64];
            b2.copy_from_slice(&tail[64..128]);
            compress(&mut h, &b2);
        }

        let mut out = [0u8; 32];
        for (i, word) in h.iter().enumerate() {
            out[i * 4..i * 4 + 4].copy_from_slice(&word.to_be_bytes());
        }
        out
    }

    fn compress(h: &mut [u32; 8], block: &[u8; 64]) {
        let mut w = [0u32; 64];
        for i in 0..16 {
            w[i] = u32::from_be_bytes([
                block[i * 4],
                block[i * 4 + 1],
                block[i * 4 + 2],
                block[i * 4 + 3],
            ]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }

        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh] = *h;
        for i in 0..64 {
            let big_s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ (!e & g);
            let temp1 = hh
                .wrapping_add(big_s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let big_s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let temp2 = big_s0.wrapping_add(maj);
            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(temp1);
            d = c;
            c = b;
            b = a;
            a = temp1.wrapping_add(temp2);
        }
        h[0] = h[0].wrapping_add(a);
        h[1] = h[1].wrapping_add(b);
        h[2] = h[2].wrapping_add(c);
        h[3] = h[3].wrapping_add(d);
        h[4] = h[4].wrapping_add(e);
        h[5] = h[5].wrapping_add(f);
        h[6] = h[6].wrapping_add(g);
        h[7] = h[7].wrapping_add(hh);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_DIMS: NetDims = NetDims {
        hidden_size: 4,
        num_features: 2,
        layer_stacks: 1,
        fc_0_output: 2,
        fc_0_padded_input: 4,
        fc_1_output: 2,
        fc_1_padded_input: 2,
        fc_2_output: 1,
        fc_2_padded_input: 2,
    };

    fn build_valid_bytes(dims: &NetDims, arch: &str) -> Vec<u8> {
        build_valid_bytes_arch(dims, arch.as_bytes())
    }

    /// Like [`build_valid_bytes`] but takes the architecture field as raw bytes,
    /// so a test can feed a non-UTF-8 arch string.
    fn build_valid_bytes_arch(dims: &NetDims, arch: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&NNUE_VERSION.to_le_bytes());
        out.extend_from_slice(&NNUE_HASH_VALUE.to_le_bytes());
        out.extend_from_slice(&(arch.len() as u32).to_le_bytes());
        out.extend_from_slice(arch);
        out.extend_from_slice(&FT_HASH.to_le_bytes());
        append_zero_leb128_block(&mut out, dims.hidden_size);
        append_zero_leb128_block(&mut out, dims.hidden_size * dims.num_features);
        for _ in 0..dims.layer_stacks {
            out.extend_from_slice(&NET_HASH.to_le_bytes());
            append_zeros(&mut out, dims.fc_0_output * 4);
            append_zeros(&mut out, dims.fc_0_output * dims.fc_0_padded_input);
            append_zeros(&mut out, dims.fc_1_output * 4);
            append_zeros(&mut out, dims.fc_1_output * dims.fc_1_padded_input);
            append_zeros(&mut out, dims.fc_2_output * 4);
            append_zeros(&mut out, dims.fc_2_output * dims.fc_2_padded_input);
        }
        out
    }

    fn append_zero_leb128_block(out: &mut Vec<u8>, count: usize) {
        out.extend_from_slice(LEB128_MAGIC);
        out.extend_from_slice(&(count as u32).to_le_bytes());
        out.resize(out.len() + count, 0);
    }

    fn append_zeros(out: &mut Vec<u8>, n: usize) {
        out.resize(out.len() + n, 0);
    }

    fn append_signed_leb128_block(out: &mut Vec<u8>, values: &[i16]) {
        out.extend_from_slice(LEB128_MAGIC);
        let len_pos = out.len();
        out.extend_from_slice(&0u32.to_le_bytes());
        let payload_start = out.len();
        for &v in values {
            encode_signed_leb128(out, v as i64);
        }
        let bytes_left = (out.len() - payload_start) as u32;
        out[len_pos..len_pos + 4].copy_from_slice(&bytes_left.to_le_bytes());
    }

    fn encode_signed_leb128(out: &mut Vec<u8>, mut value: i64) {
        loop {
            let byte = (value as u8) & 0x7F;
            // Arithmetic shift preserves the sign bit that signed LEB128 uses to end the byte stream.
            value >>= 7;
            let sign_bit = byte & 0x40;
            if (value == 0 && sign_bit == 0) || (value == -1 && sign_bit != 0) {
                out.push(byte);
                return;
            }
            out.push(byte | 0x80);
        }
    }

    fn build_bytes_with_ft(dims: &NetDims, arch: &str, biases: &[i16], weights: &[i16]) -> Vec<u8> {
        assert_eq!(biases.len(), dims.hidden_size);
        assert_eq!(weights.len(), dims.hidden_size * dims.num_features);
        let mut out = Vec::new();
        out.extend_from_slice(&NNUE_VERSION.to_le_bytes());
        out.extend_from_slice(&NNUE_HASH_VALUE.to_le_bytes());
        out.extend_from_slice(&(arch.len() as u32).to_le_bytes());
        out.extend_from_slice(arch.as_bytes());
        out.extend_from_slice(&FT_HASH.to_le_bytes());
        append_signed_leb128_block(&mut out, biases);
        append_signed_leb128_block(&mut out, weights);
        for _ in 0..dims.layer_stacks {
            out.extend_from_slice(&NET_HASH.to_le_bytes());
            append_zeros(&mut out, dims.fc_0_output * 4);
            append_zeros(&mut out, dims.fc_0_output * dims.fc_0_padded_input);
            append_zeros(&mut out, dims.fc_1_output * 4);
            append_zeros(&mut out, dims.fc_1_output * dims.fc_1_padded_input);
            append_zeros(&mut out, dims.fc_2_output * 4);
            append_zeros(&mut out, dims.fc_2_output * dims.fc_2_padded_input);
        }
        out
    }

    const SCALE_DIMS: NetDims = NetDims {
        hidden_size: 5,
        num_features: 2,
        layer_stacks: 1,
        fc_0_output: 2,
        fc_0_padded_input: 5,
        fc_1_output: 2,
        fc_1_padded_input: 2,
        fc_2_output: 1,
        fc_2_padded_input: 2,
    };

    #[test]
    fn valid_header_round_trips() {
        let bytes = build_valid_bytes(&TEST_DIMS, ARCH_STRING);
        let (net, warnings) =
            parse_from_bytes(&bytes, &TEST_DIMS, [0u8; 32]).expect("should parse");
        assert!(
            warnings.is_empty(),
            "clean file must not warn: {warnings:?}"
        );
        assert_eq!(net.header.version, NNUE_VERSION);
        assert_eq!(net.header.hash, NNUE_HASH_VALUE);
        assert_eq!(net.header.arch_id, ARCH_STRING);
        assert_eq!(net.ft_biases.len(), TEST_DIMS.hidden_size);
        assert_eq!(
            net.ft_weights.len(),
            TEST_DIMS.hidden_size * TEST_DIMS.num_features
        );
        assert_eq!(net.stacks.len(), TEST_DIMS.layer_stacks);
        let stack = &net.stacks[0];
        assert_eq!(stack.fc_0_biases.len(), TEST_DIMS.fc_0_output);
        assert_eq!(
            stack.fc_0_weights.len(),
            TEST_DIMS.fc_0_output * TEST_DIMS.fc_0_padded_input
        );
        assert_eq!(stack.fc_1_biases.len(), TEST_DIMS.fc_1_output);
        assert_eq!(
            stack.fc_1_weights.len(),
            TEST_DIMS.fc_1_output * TEST_DIMS.fc_1_padded_input
        );
        assert_eq!(stack.fc_2_biases.len(), TEST_DIMS.fc_2_output);
        assert_eq!(
            stack.fc_2_weights.len(),
            TEST_DIMS.fc_2_output * TEST_DIMS.fc_2_padded_input
        );
        assert!(net.ft_biases.iter().all(|&x| x == 0));
        assert!(net.ft_weights.iter().all(|&x| x == 0));
    }

    #[test]
    fn different_arch_string_loads_without_complaint() {
        // The architecture string is never compared; when the hashes match, a
        // different arch loads cleanly with no warning (reference: read but unused).
        let bytes = build_valid_bytes(&TEST_DIMS, "SFNNwoP1024");
        let (net, warnings) =
            parse_from_bytes(&bytes, &TEST_DIMS, [0u8; 32]).expect("should parse");
        assert!(
            warnings.is_empty(),
            "arch difference must not warn: {warnings:?}"
        );
        assert_eq!(net.header.arch_id, "SFNNwoP1024");
        assert_eq!(net.stacks.len(), TEST_DIMS.layer_stacks);
    }

    #[test]
    fn non_utf8_arch_string_loads_without_complaint() {
        // Raw, non-UTF-8 arch bytes are rendered lossily and never rejected.
        let bytes = build_valid_bytes_arch(&TEST_DIMS, &[0xFF, 0xFE, 0x00, 0x80]);
        let (net, warnings) =
            parse_from_bytes(&bytes, &TEST_DIMS, [0u8; 32]).expect("should parse");
        assert!(
            warnings.is_empty(),
            "arch bytes must not warn: {warnings:?}"
        );
        assert!(
            net.header.arch_id.contains('\u{FFFD}'),
            "non-UTF-8 bytes should render lossily, got {:?}",
            net.header.arch_id
        );
        assert_eq!(net.stacks.len(), TEST_DIMS.layer_stacks);
    }

    #[test]
    fn short_read_is_rejected() {
        let bytes = build_valid_bytes(&TEST_DIMS, ARCH_STRING);
        let truncated = &bytes[..bytes.len() - 5];
        let err = parse_from_bytes(truncated, &TEST_DIMS, [0u8; 32]).unwrap_err();
        assert!(
            matches!(err, NnueError::SizeMismatch { .. }),
            "expected SizeMismatch, got {:?}",
            err
        );
    }

    #[test]
    fn oversize_trailer_is_rejected() {
        let mut bytes = build_valid_bytes(&TEST_DIMS, ARCH_STRING);
        bytes.extend_from_slice(&[0xAA, 0xBB, 0xCC]);
        let err = parse_from_bytes(&bytes, &TEST_DIMS, [0u8; 32]).unwrap_err();
        match err {
            NnueError::SizeMismatch { expected, got } => {
                assert_eq!(got, expected + 3);
            }
            other => panic!("expected SizeMismatch, got {:?}", other),
        }
    }

    #[test]
    fn wrong_version_is_hard_rejected_with_reference_message() {
        let mut bytes = build_valid_bytes(&TEST_DIMS, ARCH_STRING);
        bytes[..4].copy_from_slice(&0xDEAD_BEEFu32.to_le_bytes());
        let err = parse_from_bytes(&bytes, &TEST_DIMS, [0u8; 32]).unwrap_err();
        match err {
            NnueError::InvalidFormat { reason } => {
                assert!(
                    reason.contains("NNUE header version mismatch: expected")
                        && reason.contains("got"),
                    "expected the reference version-mismatch message shape, got: {reason}"
                );
            }
            other => panic!("expected InvalidFormat, got {:?}", other),
        }
    }

    #[test]
    fn wrong_top_level_hash_loads_with_warning() {
        let mut bytes = build_valid_bytes(&TEST_DIMS, ARCH_STRING);
        // The top-level hash sits right after the 4-byte version.
        bytes[4..8].copy_from_slice(&0x1234_5678u32.to_le_bytes());
        let (net, warnings) = parse_from_bytes(&bytes, &TEST_DIMS, [0u8; 32]).expect("should load");
        // Exactly one warning: the file-level hash mismatch, naming both arches.
        assert_eq!(
            warnings.len(),
            1,
            "expected a single warning, got {warnings:?}"
        );
        let w = &warnings[0];
        assert!(
            w.contains("Warning: NNUE hash mismatch: expected"),
            "got: {w}"
        );
        assert!(w.contains(&format!("got {}", 0x1234_5678u32)), "got: {w}");
        assert!(
            w.contains(&format!("arch_expected={ARCH_STRING}")),
            "got: {w}"
        );
        // Parameters are still read intact.
        assert_eq!(net.ft_biases.len(), TEST_DIMS.hidden_size);
        assert_eq!(net.stacks.len(), TEST_DIMS.layer_stacks);
        assert!(net.ft_biases.iter().all(|&x| x == 0));
    }

    #[test]
    fn wrong_ft_hash_loads_with_section_warning() {
        let mut bytes = build_valid_bytes(&TEST_DIMS, ARCH_STRING);
        // ft_hash follows version(4) + hash(4) + arch_size(4) + arch bytes.
        let ft_hash_pos = 12 + ARCH_STRING.len();
        bytes[ft_hash_pos..ft_hash_pos + 4].copy_from_slice(&0x0BAD_F00Du32.to_le_bytes());
        let (net, warnings) = parse_from_bytes(&bytes, &TEST_DIMS, [0u8; 32]).expect("should load");
        assert_eq!(warnings, vec![SECTION_HASH_WARNING.to_string()]);
        assert_eq!(net.stacks.len(), TEST_DIMS.layer_stacks);
    }

    #[test]
    fn wrong_net_hash_loads_with_section_warning() {
        let mut bytes = build_valid_bytes(&TEST_DIMS, ARCH_STRING);
        // The single layer-stack's net_hash is the last 4-byte word before its
        // (all-zero) parameter blocks: version+hash+arch_size(12) + arch +
        // ft_hash(4) + ft bias block + ft weight block, then NET_HASH.
        let ft_bias_block = LEB128_MAGIC.len() + 4 + TEST_DIMS.hidden_size;
        let ft_weight_block =
            LEB128_MAGIC.len() + 4 + TEST_DIMS.hidden_size * TEST_DIMS.num_features;
        let net_hash_pos = 12 + ARCH_STRING.len() + 4 + ft_bias_block + ft_weight_block;
        bytes[net_hash_pos..net_hash_pos + 4].copy_from_slice(&0x0BAD_CAFEu32.to_le_bytes());
        let (net, warnings) = parse_from_bytes(&bytes, &TEST_DIMS, [0u8; 32]).expect("should load");
        assert_eq!(warnings, vec![SECTION_HASH_WARNING.to_string()]);
        assert_eq!(net.stacks.len(), TEST_DIMS.layer_stacks);
    }

    #[test]
    fn corrupted_leb128_magic_is_rejected() {
        let mut bytes = build_valid_bytes(&TEST_DIMS, ARCH_STRING);
        let magic_start = 12 + ARCH_STRING.len() + 4;
        bytes[magic_start] = b'X';
        let err = parse_from_bytes(&bytes, &TEST_DIMS, [0u8; 32]).unwrap_err();
        assert!(
            matches!(err, NnueError::InvalidFormat { .. }),
            "expected InvalidFormat, got {:?}",
            err
        );
    }

    #[test]
    fn ft_scale_doubles_biases_and_weights() {
        let biases: [i16; 5] = [0, 1, -1, 16_383, -16_384];
        let weight_row: [i16; 5] = [0, 1, -1, 16_383, -16_384];
        let mut weights = Vec::with_capacity(SCALE_DIMS.hidden_size * SCALE_DIMS.num_features);
        for _ in 0..SCALE_DIMS.num_features {
            weights.extend_from_slice(&weight_row);
        }

        let bytes = build_bytes_with_ft(&SCALE_DIMS, ARCH_STRING, &biases, &weights);
        let (net, _warnings) =
            parse_from_bytes(&bytes, &SCALE_DIMS, [0u8; 32]).expect("should parse");

        let expected_row: [i16; 5] = [0, 2, -2, 32_766, -32_768];
        assert_eq!(&*net.ft_biases, &expected_row[..]);
        assert_eq!(net.ft_weights.len(), weights.len());
        for chunk in net.ft_weights.chunks(expected_row.len()) {
            assert_eq!(chunk, &expected_row[..]);
        }
    }

    #[test]
    fn ft_scale_rejects_overflow_bias() {
        let biases: [i16; 5] = [16_384, 0, 0, 0, 0];
        let weights = vec![0i16; SCALE_DIMS.hidden_size * SCALE_DIMS.num_features];

        let bytes = build_bytes_with_ft(&SCALE_DIMS, ARCH_STRING, &biases, &weights);
        let err = parse_from_bytes(&bytes, &SCALE_DIMS, [0u8; 32]).unwrap_err();

        match err {
            NnueError::InvalidFormat { reason } => {
                assert!(
                    reason.contains("16384"),
                    "expected message to name the offending value, got: {}",
                    reason
                );
                assert!(
                    reason.contains("out of range") || reason.contains("overflow"),
                    "expected message to mention overflow / out-of-range, got: {}",
                    reason
                );
            }
            other => panic!("expected InvalidFormat, got {:?}", other),
        }
    }

    #[test]
    fn leb128_decoder_handles_boundary_values() {
        // signed-LEB128 stream: { -1, 0, 1, 127, -128 }
        let payload = [0x7F, 0x00, 0x01, 0xFF, 0x00, 0x80, 0x7F];
        let mut pos = 0usize;
        let mut decoded = Vec::new();
        for _ in 0..5 {
            decoded.push(read_signed_leb128(&payload, &mut pos).unwrap());
        }
        assert_eq!(decoded, vec![-1, 0, 1, 127, -128]);
        assert_eq!(pos, payload.len());
    }

    #[test]
    fn sha256_matches_nist_empty_vector() {
        let digest = sha256::digest(b"");
        let expected =
            hex_decode("e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
        assert_eq!(digest, expected);
    }

    #[test]
    fn sha256_matches_nist_abc_vector() {
        let digest = sha256::digest(b"abc");
        let expected =
            hex_decode("ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
        assert_eq!(digest, expected);
    }

    #[test]
    fn sha256_matches_nist_long_vector() {
        let digest = sha256::digest(b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq");
        let expected =
            hex_decode("248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");
        assert_eq!(digest, expected);
    }

    fn hex_decode(s: &str) -> [u8; 32] {
        assert_eq!(s.len(), 64);
        let mut out = [0u8; 32];
        for (i, chunk) in s.as_bytes().chunks(2).enumerate() {
            out[i] = u8::from_str_radix(std::str::from_utf8(chunk).unwrap(), 16).unwrap();
        }
        out
    }
}
