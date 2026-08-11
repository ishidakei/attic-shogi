//! Driver-level session tests for the runtime `FV_SCALE` option and the
//! `eval_options.txt` override file, against a **synthetic** all-zero SFNN-1536
//! network (hermetic — no staged `nn.bin` required).
//!
//! The synthetic network evaluates every position to 0 regardless of the scale,
//! so these tests assert the *override mechanism* — the isready-time info
//! string, silence when the file is absent, and the FIXED lock that makes a
//! later `setoption name FV_SCALE` a no-op — rather than a numeric eval effect
//! (that is covered on the real network by `attic-eval/tests/fv_scale.rs`).
//!
//! The FIXED-lock test reads the process-global eval scale after a `go`; it is
//! the only test in this binary that runs a `go`, so that global is never raced
//! by a sibling test. The byte layout mirrors `crates/attic-eval/src/loader.rs`
//! (as in `tests/eval_session.rs`).

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, Ordering};

use attic_protocol::UsiDriver;

// --- SFNN-1536 file-format constants (mirror attic-eval/src/loader.rs).
const NNUE_VERSION: u32 = 0x7AF3_2F16;
const NNUE_HASH_VALUE: u32 = 0x3C20_3B32;
const FT_HASH: u32 = 0x5F13_4AB8;
const NET_HASH: u32 = 0x6333_718A;
const LEB128_MAGIC: &[u8; 17] = b"COMPRESSED_LEB128";
const ARCH_STRING: &str = "ModelType=SFNNWithoutPsqt;Features=HalfKA_hm(Friend)[73305->1536x2],Network=AffineTransform[1<-32](ClippedReLU[32](AffineTransform[32<-15](ClippedReLU[15](AffineTransform[15<-3072](InputSlice[3072(0:3072)]))))){LayerStack=9}";

// --- Dimensions (mirror attic-eval/src/types.rs).
const HIDDEN_SIZE: usize = 1_536;
const NUM_FEATURES: usize = 73_305;
const LAYER_STACKS: usize = 9;
const FC_0_OUTPUT: usize = 16;
const FC_0_PADDED_INPUT: usize = 1_536;
const FC_1_OUTPUT: usize = 32;
const FC_1_PADDED_INPUT: usize = 32;
const FC_2_OUTPUT: usize = 1;
const FC_2_PADDED_INPUT: usize = 32;

struct TempDir {
    path: PathBuf,
}

impl TempDir {
    fn new(tag: &str) -> Self {
        static CTR: AtomicU32 = AtomicU32::new(0);
        let n = CTR.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "fv-scale-session-{}-{}-{n}",
            std::process::id(),
            tag
        ));
        std::fs::create_dir_all(&path).expect("create temp dir");
        Self { path }
    }

    fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}

fn build_zero_network_bytes() -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(&NNUE_VERSION.to_le_bytes());
    out.extend_from_slice(&NNUE_HASH_VALUE.to_le_bytes());
    out.extend_from_slice(&(ARCH_STRING.len() as u32).to_le_bytes());
    out.extend_from_slice(ARCH_STRING.as_bytes());
    out.extend_from_slice(&FT_HASH.to_le_bytes());
    append_zero_leb128_block(&mut out, HIDDEN_SIZE);
    append_zero_leb128_block(&mut out, HIDDEN_SIZE * NUM_FEATURES);
    for _ in 0..LAYER_STACKS {
        out.extend_from_slice(&NET_HASH.to_le_bytes());
        append_zeros(&mut out, FC_0_OUTPUT * 4);
        append_zeros(&mut out, FC_0_OUTPUT * FC_0_PADDED_INPUT);
        append_zeros(&mut out, FC_1_OUTPUT * 4);
        append_zeros(&mut out, FC_1_OUTPUT * FC_1_PADDED_INPUT);
        append_zeros(&mut out, FC_2_OUTPUT * 4);
        append_zeros(&mut out, FC_2_OUTPUT * FC_2_PADDED_INPUT);
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

fn write_synthetic_nn_bin(dir: &Path) {
    std::fs::write(dir.join("nn.bin"), build_zero_network_bytes()).expect("write nn.bin");
}

fn drive(input: &str) -> String {
    let output = std::sync::Arc::new(std::sync::Mutex::new(Vec::<u8>::new()));
    let driver = UsiDriver::new(input.as_bytes(), std::sync::Arc::clone(&output));
    driver.run().expect("driver run");
    let bytes = output.lock().expect("output lock").clone();
    String::from_utf8(bytes).expect("utf-8")
}

const OVERRIDE_LINE: &str = "info string engine option override. name = FV_SCALE , value = 24";

#[test]
fn eval_options_override_applies_at_isready() {
    let dir = TempDir::new("apply");
    write_synthetic_nn_bin(dir.path());
    std::fs::write(dir.path().join("eval_options.txt"), "FV_SCALE 24\n").expect("write file");
    let evaldir = dir.path().to_str().expect("utf-8 temp path");

    let out = drive(&format!(
        "usi\nsetoption name EvalDir value {evaldir}\nisready\nquit\n"
    ));

    // The "read engine options" notice names the eval_options.txt path, and the
    // override info string for FV_SCALE=24 is emitted.
    assert!(
        out.contains("info string read engine options, path = ")
            && out.contains("eval_options.txt"),
        "missing read-engine-options notice in:\n{out}"
    );
    assert!(
        out.contains(OVERRIDE_LINE),
        "missing FV_SCALE override info string in:\n{out}"
    );
    // The network still loads (override runs before eval load).
    assert!(out.contains("readyok"), "expected readyok in:\n{out}");
    assert!(
        !out.contains("eval load failed"),
        "unexpected load failure:\n{out}"
    );
}

#[test]
fn absent_eval_options_is_silent() {
    let dir = TempDir::new("absent");
    write_synthetic_nn_bin(dir.path());
    // Deliberately do NOT create eval_options.txt.
    let evaldir = dir.path().to_str().expect("utf-8 temp path");

    let out = drive(&format!(
        "usi\nsetoption name EvalDir value {evaldir}\nisready\nquit\n"
    ));

    // No override notice at all when neither engine_options.txt (cwd) nor
    // eval_options.txt (EvalDir) exists.
    assert!(
        !out.contains("read engine options"),
        "absent override files must be silent, got:\n{out}"
    );
    assert!(
        !out.contains("engine option override"),
        "unexpected override:\n{out}"
    );
    assert!(out.contains("readyok"), "expected readyok in:\n{out}");
}

#[test]
fn setoption_after_override_is_fixed() {
    let dir = TempDir::new("fixed");
    write_synthetic_nn_bin(dir.path());
    std::fs::write(dir.path().join("eval_options.txt"), "FV_SCALE 24\n").expect("write file");
    let evaldir = dir.path().to_str().expect("utf-8 temp path");

    // Override to 24 (locking it FIXED), then try to setoption back to 16, then
    // run a `go` — which pushes the *current* FV_SCALE option to the eval's live
    // scale. If the FIXED lock held, that value is still 24.
    let out = drive(&format!(
        "usi\n\
         setoption name Threads value 1\n\
         setoption name EvalDir value {evaldir}\n\
         isready\n\
         setoption name FV_SCALE value 16\n\
         position startpos\n\
         go depth 1\n\
         quit\n"
    ));

    assert!(out.contains(OVERRIDE_LINE), "missing override in:\n{out}");
    // setoption on a fixed option is a silent no-op (no rejection message).
    assert!(
        !out.contains("rejected"),
        "fixed setoption must be silent:\n{out}"
    );
    // The `go` propagated the still-24 option to the eval global, proving the
    // `setoption name FV_SCALE value 16` was ignored. This is the only test in
    // the binary that runs a `go`, so the global is not raced.
    assert_eq!(
        attic_search::fv_scale(),
        24,
        "fixed FV_SCALE must remain 24 after a setoption to 16"
    );
}
