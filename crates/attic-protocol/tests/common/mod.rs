//! Shared harness for driver-level session tests: a synthetic (all-zero,
//! format-valid) `nn.bin` builder, a temp-dir guard, an all-at-once `drive`, a
//! `.ybb` writer, and a streaming input harness for async-hold tests.

#![allow(dead_code)]

use std::io::{self, BufReader, Read};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::mpsc::{Receiver, Sender, channel};
use std::sync::{Arc, Mutex};

use attic_protocol::UsiDriver;
use attic_state::{Move, Position, parse_sfen, parse_usi_move, sfen_pack};

// SFNN-1536 file-format constants, mirroring `attic-eval`'s loader.
const NNUE_VERSION: u32 = 0x7AF3_2F16;
const NNUE_HASH_VALUE: u32 = 0x3C20_3B32;
const FT_HASH: u32 = 0x5F13_4AB8;
const NET_HASH: u32 = 0x6333_718A;
const LEB128_MAGIC: &[u8; 17] = b"COMPRESSED_LEB128";
const ARCH_STRING: &str = "ModelType=SFNNWithoutPsqt;Features=HalfKA_hm(Friend)[73305->1536x2],Network=AffineTransform[1<-32](ClippedReLU[32](AffineTransform[32<-15](ClippedReLU[15](AffineTransform[15<-3072](InputSlice[3072(0:3072)]))))){LayerStack=9}";

const HIDDEN_SIZE: usize = 1_536;
const NUM_FEATURES: usize = 73_305;
const LAYER_STACKS: usize = 9;
const FC_0_OUTPUT: usize = 16;
const FC_0_PADDED_INPUT: usize = 1_536;
const FC_1_OUTPUT: usize = 32;
const FC_1_PADDED_INPUT: usize = 32;
const FC_2_OUTPUT: usize = 1;
const FC_2_PADDED_INPUT: usize = 32;

/// A temp directory removed on drop.
pub struct TempDir {
    path: PathBuf,
}

impl TempDir {
    pub fn new(tag: &str) -> Self {
        static CTR: AtomicU32 = AtomicU32::new(0);
        let n = CTR.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "engine-book-session-{}-{}-{n}",
            std::process::id(),
            tag
        ));
        std::fs::create_dir_all(&path).expect("create temp dir");
        Self { path }
    }

    pub fn path(&self) -> &Path {
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

/// Write a synthetic `nn.bin` into `dir` and return its path.
pub fn write_synthetic_nn_bin(dir: &Path) -> PathBuf {
    let path = dir.join("nn.bin");
    std::fs::write(&path, build_zero_network_bytes()).expect("write nn.bin");
    path
}

/// Drive a full canned session in-process and return the transcript. `run`
/// joins any worker, so the buffer is complete on return.
///
/// The book / `rtime` PRNG seed is process entropy, so this suits only sessions
/// whose output does not depend on that randomisation.
pub fn drive(input: &str) -> String {
    let output = Arc::new(Mutex::new(Vec::<u8>::new()));
    let driver = UsiDriver::new(input.as_bytes(), Arc::clone(&output));
    driver.run().expect("driver run");
    let bytes = output.lock().expect("output lock").clone();
    String::from_utf8(bytes).expect("utf-8")
}

/// A fixed book-PRNG seed for reproducible book / `rtime` sessions.
pub const TEST_BOOK_SEED: u64 = 0x9E37_79B9_7F4A_7C15;

/// Like [`drive`] but with an explicit book-PRNG session seed, so book-move
/// selection (and `rtime`) is reproducible.
pub fn drive_with_seed(input: &str, book_seed: u64) -> String {
    let output = Arc::new(Mutex::new(Vec::<u8>::new()));
    let driver = UsiDriver::with_book_seed(input.as_bytes(), Arc::clone(&output), book_seed);
    driver.run().expect("driver run");
    let bytes = output.lock().expect("output lock").clone();
    String::from_utf8(bytes).expect("utf-8")
}

/// Like [`drive_with_seed`] but with an injected `engine_option_profile.txt`
/// path, so a session can exercise a non-default profile without one sitting in
/// the test process's working directory.
pub fn drive_with_profile(input: &str, book_seed: u64, profile: &Path) -> String {
    let output = Arc::new(Mutex::new(Vec::<u8>::new()));
    let driver =
        UsiDriver::with_option_profile(input.as_bytes(), Arc::clone(&output), book_seed, profile);
    driver.run().expect("driver run");
    let bytes = output.lock().expect("output lock").clone();
    String::from_utf8(bytes).expect("utf-8")
}

/// Write an `engine_option_profile.txt` into `dir` and return its path.
pub fn write_option_profile(dir: &Path, contents: &str) -> PathBuf {
    let path = dir.join("engine_option_profile.txt");
    std::fs::write(&path, contents).expect("write engine_option_profile.txt");
    path
}

pub fn bestmove_lines(out: &str) -> Vec<&str> {
    out.lines()
        .filter_map(|l| l.strip_prefix("bestmove "))
        .collect()
}

/// One book move for [`write_ybb`]: `(usi, value, depth)`.
pub type YbbMove<'a> = (&'a str, i16, u16);

/// Build and write a depth-carrying `.ybb` at `path` from `(sfen, moves)`
/// records, sorted by packed key as the format requires.
pub fn write_ybb(path: &Path, records: &[(&str, Vec<YbbMove<'_>>)]) {
    const MAGIC: &[u8; 16] = b"YANE-BINBOOK-V1\0";

    struct Rec {
        packed: [u8; 32],
        ply: u16,
        moves: Vec<(u16, i16, u16)>,
    }

    let mut recs: Vec<Rec> = records
        .iter()
        .map(|(sfen, moves)| {
            let pos = parse_sfen(sfen).expect("valid sfen");
            let moves = moves
                .iter()
                .map(|(usi, v, d)| {
                    let m16 = parse_usi_move(usi, &pos).expect("valid move").move16();
                    (m16, *v, *d)
                })
                .collect();
            Rec {
                packed: sfen_pack(&pos),
                ply: pos.ply(),
                moves,
            }
        })
        .collect();
    recs.sort_by_key(|r| r.packed);

    let mut header = Vec::new();
    header.extend_from_slice(MAGIC);
    header.extend_from_slice(&(recs.len() as u64).to_le_bytes());
    header.extend_from_slice(&1u64.to_le_bytes()); // flags: move-depth present

    let mut index = Vec::new();
    let mut moves = Vec::new();
    for r in &recs {
        let moves_offset = moves.len() as u64;
        index.extend_from_slice(&r.packed);
        index.extend_from_slice(&moves_offset.to_le_bytes());
        index.extend_from_slice(&r.ply.to_le_bytes());
        index.extend_from_slice(&(r.moves.len() as u16).to_le_bytes());
        for (m, v, d) in &r.moves {
            moves.extend_from_slice(&m.to_le_bytes());
            moves.extend_from_slice(&(*v as u16).to_le_bytes());
            moves.extend_from_slice(&d.to_le_bytes());
        }
    }
    let mut out = header;
    out.extend_from_slice(&index);
    out.extend_from_slice(&moves);
    std::fs::write(path, &out).expect("write ybb");
}

/// Copy the committed `tests/fixtures/book/sample.ybb` into `dir` under
/// `dest_name` and return its path.
pub fn stage_sample_ybb(dir: &Path, dest_name: &str) -> PathBuf {
    let src = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join("tests/fixtures/book/sample.ybb");
    let dest = dir.join(dest_name);
    std::fs::copy(&src, &dest).expect("copy sample.ybb");
    dest
}

pub fn parse(sfen: &str) -> Position {
    parse_sfen(sfen).expect("valid sfen")
}

pub fn legal(pos: &Position) -> Vec<Move> {
    let mut v = Vec::new();
    pos.generate_legal_all(&mut v);
    v
}

/// A blocking [`Read`] fed line-chunks over an mpsc channel, so a test can feed
/// the driver commands over time and observe output between them.
pub struct BlockingReader {
    rx: Receiver<Option<Vec<u8>>>,
    buf: Vec<u8>,
    pos: usize,
    done: bool,
}

impl Read for BlockingReader {
    fn read(&mut self, out: &mut [u8]) -> io::Result<usize> {
        loop {
            if self.pos < self.buf.len() {
                let n = std::cmp::min(out.len(), self.buf.len() - self.pos);
                out[..n].copy_from_slice(&self.buf[self.pos..self.pos + n]);
                self.pos += n;
                return Ok(n);
            }
            if self.done {
                return Ok(0);
            }
            match self.rx.recv() {
                Ok(Some(chunk)) => {
                    self.buf = chunk;
                    self.pos = 0;
                }
                Ok(None) | Err(_) => {
                    self.done = true;
                    return Ok(0);
                }
            }
        }
    }
}

/// A running driver on its own thread, fed incrementally.
pub struct StreamHarness {
    tx: Sender<Option<Vec<u8>>>,
    output: Arc<Mutex<Vec<u8>>>,
    handle: Option<std::thread::JoinHandle<()>>,
}

impl StreamHarness {
    pub fn start() -> Self {
        Self::start_with_seed(None)
    }

    /// A streaming harness with an explicit book-PRNG session seed, for
    /// reproducible book / `rtime` sessions. `None` uses process entropy.
    pub fn start_with_seed(book_seed: Option<u64>) -> Self {
        let (tx, rx) = channel();
        let output = Arc::new(Mutex::new(Vec::<u8>::new()));
        let out2 = Arc::clone(&output);
        let handle = std::thread::spawn(move || {
            let reader = BufReader::new(BlockingReader {
                rx,
                buf: Vec::new(),
                pos: 0,
                done: false,
            });
            match book_seed {
                Some(seed) => UsiDriver::with_book_seed(reader, out2, seed)
                    .run()
                    .expect("driver run"),
                None => UsiDriver::new(reader, out2).run().expect("driver run"),
            }
        });
        StreamHarness {
            tx,
            output,
            handle: Some(handle),
        }
    }

    /// Feed one command line; a `\n` is appended.
    pub fn send(&self, line: &str) {
        let mut bytes = line.as_bytes().to_vec();
        bytes.push(b'\n');
        self.tx.send(Some(bytes)).expect("send");
    }

    pub fn output(&self) -> String {
        String::from_utf8(self.output.lock().expect("lock").clone()).expect("utf-8")
    }

    /// Poll until `pred(output)` holds or `timeout` elapses.
    pub fn wait_until(&self, timeout_ms: u64, pred: impl Fn(&str) -> bool) -> bool {
        let mut waited = 0u64;
        loop {
            if pred(&self.output()) {
                return true;
            }
            if waited >= timeout_ms {
                return false;
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
            waited += 5;
        }
    }

    /// Send `quit`, close the input, and join the driver thread.
    pub fn quit_join(mut self) -> String {
        self.send("quit");
        let _ = self.tx.send(None);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
        self.output()
    }
}
