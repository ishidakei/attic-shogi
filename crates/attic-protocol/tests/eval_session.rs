//! Driver-level session tests against a synthetic SFNN-1536 network built in a
//! temp dir, so they run everywhere rather than only where the real network is
//! staged.
//!
//! That network is all zeros, so every position evaluates to the same constant
//! and the chosen move is fully deterministic — which lets these assert the
//! driver's `bestmove` equals a direct [`QSearch::run_root`] call for the same
//! network, position, and table sizing.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, Ordering};

use attic_protocol::UsiDriver;
use attic_search::{QSearch, RootKind, RootOutcome, Search};
use attic_state::{Move, Position, format_usi_move, parse_sfen, parse_usi_move};
use attic_storage::TranspositionTable;

/// Engine default `USI_Hash` in MiB, reproduced here so the direct `run_root`
/// call searches under the same table conditions the driver creates.
const HASH_MB: usize = 1024;

/// The USI-string form of a `run_root` outcome's bestmove.
fn bestmove_usi(outcome: &RootOutcome) -> String {
    match outcome.kind {
        RootKind::Resign => "resign".to_string(),
        RootKind::DeclarationWin => "win".to_string(),
        RootKind::Normal => format_usi_move(outcome.best_move),
    }
}

// SFNN-1536 file-format constants, mirroring `attic-eval`'s loader.
const NNUE_VERSION: u32 = 0x7AF3_2F16;
const NNUE_HASH_VALUE: u32 = 0x3C20_3B32;
const FT_HASH: u32 = 0x5F13_4AB8;
const NET_HASH: u32 = 0x6333_718A;
const LEB128_MAGIC: &[u8; 17] = b"COMPRESSED_LEB128";
const ARCH_STRING: &str = "ModelType=SFNNWithoutPsqt;Features=HalfKA_hm(Friend)[73305->1536x2],Network=AffineTransform[1<-32](ClippedReLU[32](AffineTransform[32<-15](ClippedReLU[15](AffineTransform[15<-3072](InputSlice[3072(0:3072)]))))){LayerStack=9}";

// Dimensions.
const HIDDEN_SIZE: usize = 1_536;
const NUM_FEATURES: usize = 73_305;
const LAYER_STACKS: usize = 9;
const FC_0_OUTPUT: usize = 16;
const FC_0_PADDED_INPUT: usize = 1_536;
const FC_1_OUTPUT: usize = 32;
const FC_1_PADDED_INPUT: usize = 32;
const FC_2_OUTPUT: usize = 1;
const FC_2_PADDED_INPUT: usize = 32;

/// A temp directory removed on drop, so a ~107 MiB synthetic network does not
/// outlive the test binary.
struct TempDir {
    path: PathBuf,
}

impl TempDir {
    fn new(tag: &str) -> Self {
        static CTR: AtomicU32 = AtomicU32::new(0);
        let n = CTR.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "attic-eval-session-{}-{}-{n}",
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

/// Build the raw bytes of an all-zero, format-valid SFNN-1536 `nn.bin`.
fn build_zero_network_bytes() -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(&NNUE_VERSION.to_le_bytes());
    out.extend_from_slice(&NNUE_HASH_VALUE.to_le_bytes());
    out.extend_from_slice(&(ARCH_STRING.len() as u32).to_le_bytes());
    out.extend_from_slice(ARCH_STRING.as_bytes());
    out.extend_from_slice(&FT_HASH.to_le_bytes());
    // Biases then weights, all zero: each i16 is a single 0x00 LEB128 byte, so
    // `bytes_left == count`.
    append_zero_leb128_block(&mut out, HIDDEN_SIZE);
    append_zero_leb128_block(&mut out, HIDDEN_SIZE * NUM_FEATURES);
    for _ in 0..LAYER_STACKS {
        out.extend_from_slice(&NET_HASH.to_le_bytes());
        append_zeros(&mut out, FC_0_OUTPUT * 4); // i32 biases
        append_zeros(&mut out, FC_0_OUTPUT * FC_0_PADDED_INPUT); // i8 weights
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
fn write_synthetic_nn_bin(dir: &Path) -> PathBuf {
    let path = dir.join("nn.bin");
    std::fs::write(&path, build_zero_network_bytes()).expect("write nn.bin");
    path
}

fn drive(input: &str) -> String {
    let output = std::sync::Arc::new(std::sync::Mutex::new(Vec::<u8>::new()));
    let driver = UsiDriver::new(input.as_bytes(), std::sync::Arc::clone(&output));
    driver.run().expect("driver run");
    let bytes = output.lock().expect("output lock").clone();
    String::from_utf8(bytes).expect("utf-8")
}

fn legal_moves(p: &Position) -> Vec<Move> {
    let mut moves = Vec::new();
    p.generate_legal_all(&mut moves);
    moves
}

fn bestmove_lines(out: &str) -> Vec<&str> {
    out.lines()
        .filter_map(|l| l.strip_prefix("bestmove "))
        .collect()
}

#[test]
#[cfg_attr(miri, ignore)]
fn synthetic_network_session_matches_direct_search_choice() {
    let dir = TempDir::new("session");
    let path = write_synthetic_nn_bin(dir.path());
    let evaldir = dir.path().to_str().expect("utf-8 temp path");

    // Scoped so the network and its 1024 MiB table free before the driver
    // session allocates its own.
    let startpos = parse_sfen(attic_state::STARTPOS_SFEN).expect("startpos SFEN");
    let expected_usi = {
        let search = Search::from_network_file(&path).expect("synthetic network loads");
        let mut tt = TranspositionTable::new();
        tt.resize(HASH_MB);
        let outcome = QSearch::new(search.network(), &tt).run_root(&startpos, 1);
        bestmove_usi(&outcome)
    };

    // The repeat `isready` exercises the idempotent reload. `Threads value 1`
    // makes the driver's choice match the direct `run_root` above: the default
    // four workers share and pollute the table.
    let session = format!(
        "usi\n\
         setoption name Threads value 1\n\
         setoption name EvalDir value {evaldir}\n\
         isready\n\
         isready\n\
         position startpos\n\
         go depth 1\n\
         quit\n"
    );
    let out = drive(&session);

    assert_eq!(
        out.matches("readyok\n").count(),
        2,
        "both isready must emit readyok in:\n{out}"
    );
    assert!(
        !out.contains("eval load failed"),
        "unexpected load failure in:\n{out}"
    );

    assert!(
        out.lines().any(|l| l.starts_with("info depth 1 ")),
        "missing search info report in:\n{out}"
    );

    let bestmoves = bestmove_lines(&out);
    assert_eq!(bestmoves.len(), 1, "expected one bestmove in:\n{out}");
    assert_eq!(
        bestmoves[0], expected_usi,
        "driver bestmove must equal attic-search's direct choice"
    );
    let parsed = parse_usi_move(bestmoves[0], &startpos).expect("well-formed USI");
    assert!(
        legal_moves(&startpos).contains(&parsed),
        "{} is not a legal startpos move",
        bestmoves[0]
    );
}

#[test]
#[cfg_attr(miri, ignore)]
fn isready_keep_alive_emits_bare_newline_during_heavy_load() {
    // With a very short injected poll the real heavy work here — the `nn.bin`
    // load and the 1024 MiB table zeroing — spans many keep-alive ticks.
    let dir = TempDir::new("keepalive");
    write_synthetic_nn_bin(dir.path());
    let evaldir = dir.path().to_str().expect("utf-8 temp path");

    let input = format!(
        "usi\n\
         setoption name EvalDir value {evaldir}\n\
         isready\n\
         quit\n"
    );
    let output = std::sync::Arc::new(std::sync::Mutex::new(Vec::<u8>::new()));
    let driver = UsiDriver::new(input.as_bytes(), std::sync::Arc::clone(&output))
        .with_keep_alive_poll(std::time::Duration::from_micros(100));
    driver.run().expect("driver run");
    let out = String::from_utf8(output.lock().expect("output lock").clone()).expect("utf-8");

    assert!(
        !out.contains("eval load failed"),
        "unexpected load failure in:\n{out:?}"
    );
    let readyok_pos = out.find("readyok\n").expect("readyok emitted");

    // No line the driver emits is otherwise empty, so any empty line is a
    // keep-alive newline.
    let before = &out[..readyok_pos];
    let bare_before = before.split('\n').filter(|s| s.is_empty()).count();
    assert!(
        bare_before >= 1,
        "expected a bare keep-alive newline before readyok in:\n{out:?}"
    );

    // The keep-alive newline goes through the shared writer as a whole line, so
    // no USI line is split by a stray newline.
    assert!(out.contains("usiok\n"), "usiok must be intact in:\n{out:?}");
    assert!(
        out.lines().any(|l| l == "readyok"),
        "readyok must be intact in:\n{out:?}"
    );
}

#[test]
#[cfg_attr(miri, ignore)]
fn synthetic_network_reuse_reset_and_mate_resign() {
    let dir = TempDir::new("reuse");
    write_synthetic_nn_bin(dir.path());
    let evaldir = dir.path().to_str().expect("utf-8 temp path");

    // One `isready` serves all three `go`s, pinning that the loaded network is
    // reused across positions without reloading.
    let mut post_7g7f = parse_sfen(attic_state::STARTPOS_SFEN).expect("startpos");
    let m = parse_usi_move("7g7f", &post_7g7f).expect("legal 7g7f");
    post_7g7f.do_move(m);
    let startpos = parse_sfen(attic_state::STARTPOS_SFEN).expect("startpos");

    // Reproduces the session's table lifecycle: one 1024 MiB table for the first
    // `go`, then the `usinewgame` clear before the second. Scoped so it frees
    // before the driver allocates its own.
    let (expected_after_7g7f, expected_startpos) = {
        let search =
            Search::from_network_file(&dir.path().join("nn.bin")).expect("synthetic network loads");
        let mut tt = TranspositionTable::new();
        tt.resize(HASH_MB);
        let e1 = bestmove_usi(&QSearch::new(search.network(), &tt).run_root(&post_7g7f, 1));
        tt.clear(); // usinewgame equivalent.
        let e2 = bestmove_usi(&QSearch::new(search.network(), &tt).run_root(&startpos, 1));
        (e1, e2)
    };

    // A mate for White: no legal move, so the search resigns.
    let mate = "4k4/4G4/3S5/9/9/9/9/9/4K4 w - 1";

    // `go depth 1` and `Threads value 1` make each search the deterministic
    // `run_root(pos, 1)` above; a bare `go` would be clock-driven.
    let session = format!(
        "usi\n\
         setoption name Threads value 1\n\
         setoption name EvalDir value {evaldir}\n\
         isready\n\
         position startpos moves 7g7f\n\
         go depth 1\n\
         usinewgame\n\
         go depth 1\n\
         position sfen {mate}\n\
         go depth 1\n\
         quit\n"
    );
    let out = drive(&session);

    assert_eq!(
        out.matches("readyok\n").count(),
        1,
        "one isready, one readyok in:\n{out}"
    );

    let bestmoves = bestmove_lines(&out);
    assert_eq!(bestmoves.len(), 3, "expected three bestmoves in:\n{out}");
    assert_eq!(
        bestmoves[0], expected_after_7g7f,
        "first go must reflect the post-7g7f position"
    );
    // `usinewgame` resets the position to startpos.
    assert_eq!(
        bestmoves[1], expected_startpos,
        "usinewgame must reset the position to startpos"
    );
    assert_eq!(bestmoves[2], "resign", "mate position must resign");
}

/// A CSA-27-point-declarable position for the side to move: the Black king
/// walled into the enemy field with 12 own pieces there.
const DECLARABLE_SFEN: &str = "+R+R+B+B5/3GKG3/2SGGGS2/9/9/9/9/9/4k4 b R 1";

/// A TryRule-declarable position: Black king on 5b, the try square 5a empty and
/// unattacked, the enemy king tucked away in the corner. The declaration move is
/// the king step 5b→5a (`5b5a`).
const TRYABLE_SFEN: &str = "9/4K4/9/9/9/9/9/9/8k b - 1";

#[test]
#[cfg_attr(miri, ignore)]
fn entering_king_default_declares_win_without_searching() {
    // The declaration shortcut fires before any worker runs, so no search `info`
    // line is emitted at all.
    let dir = TempDir::new("ekr-default");
    write_synthetic_nn_bin(dir.path());
    let evaldir = dir.path().to_str().expect("utf-8 temp path");

    let session = format!(
        "usi\n\
         setoption name EvalDir value {evaldir}\n\
         isready\n\
         position sfen {DECLARABLE_SFEN}\n\
         go depth 1\n\
         quit\n"
    );
    let out = drive(&session);

    let bestmoves = bestmove_lines(&out);
    assert_eq!(bestmoves.len(), 1, "expected one bestmove in:\n{out}");
    assert_eq!(bestmoves[0], "win", "default rule must declare in:\n{out}");
    assert!(
        !out.lines().any(|l| l.starts_with("info depth ")),
        "declaration shortcut must not run a search in:\n{out}"
    );
}

#[test]
#[cfg_attr(miri, ignore)]
fn entering_king_none_runs_a_real_search() {
    let dir = TempDir::new("ekr-none");
    write_synthetic_nn_bin(dir.path());
    let evaldir = dir.path().to_str().expect("utf-8 temp path");

    let session = format!(
        "usi\n\
         setoption name Threads value 1\n\
         setoption name EvalDir value {evaldir}\n\
         setoption name EnteringKingRule value NoEnteringKing\n\
         isready\n\
         position sfen {DECLARABLE_SFEN}\n\
         go depth 1\n\
         quit\n"
    );
    let out = drive(&session);

    let bestmoves = bestmove_lines(&out);
    assert_eq!(bestmoves.len(), 1, "expected one bestmove in:\n{out}");
    assert_ne!(
        bestmoves[0], "win",
        "NoEnteringKing must not declare in:\n{out}"
    );
    assert!(
        out.lines().any(|l| l.starts_with("info depth 1 ")),
        "a real search must run under NoEnteringKing in:\n{out}"
    );
    let p = parse_sfen(DECLARABLE_SFEN).expect("declarable SFEN");
    let best = bestmoves[0]
        .split_whitespace()
        .next()
        .expect("bestmove token");
    let parsed = parse_usi_move(best, &p).expect("well-formed USI move");
    assert!(
        legal_moves(&p).contains(&parsed),
        "{best} is not a legal move of the declarable position"
    );
}

#[test]
#[cfg_attr(miri, ignore)]
fn entering_king_try_rule_declares_the_king_move() {
    // Under `TryRule` the reply is the actual king move onto the try square,
    // not the bare `win` token.
    let dir = TempDir::new("ekr-try");
    write_synthetic_nn_bin(dir.path());
    let evaldir = dir.path().to_str().expect("utf-8 temp path");

    let session = format!(
        "usi\n\
         setoption name EvalDir value {evaldir}\n\
         setoption name EnteringKingRule value TryRule\n\
         isready\n\
         position sfen {TRYABLE_SFEN}\n\
         go depth 1\n\
         quit\n"
    );
    let out = drive(&session);

    let bestmoves = bestmove_lines(&out);
    assert_eq!(bestmoves.len(), 1, "expected one bestmove in:\n{out}");
    let best = bestmoves[0]
        .split_whitespace()
        .next()
        .expect("bestmove token");
    assert_eq!(best, "5b5a", "TryRule must play the king move in:\n{out}");
    assert!(
        !out.lines().any(|l| l.starts_with("info depth ")),
        "the try declaration must not run a search in:\n{out}"
    );
    let p = parse_sfen(TRYABLE_SFEN).expect("try-able SFEN");
    let parsed = parse_usi_move(best, &p).expect("well-formed USI move");
    assert!(
        legal_moves(&p).contains(&parsed),
        "{best} is not a legal move of the try-able position"
    );
}

#[test]
#[cfg_attr(miri, ignore)]
fn threads4_stop_then_quit_joins_all_workers_and_exits_cleanly() {
    // `drive` returning at all proves the join of every worker did not hang; the
    // bestmove count proves the parallel search resolves to a single reply.
    let dir = TempDir::new("threads4");
    write_synthetic_nn_bin(dir.path());
    let evaldir = dir.path().to_str().expect("utf-8 temp path");

    let session = format!(
        "usi\n\
         setoption name Threads value 4\n\
         setoption name EvalDir value {evaldir}\n\
         isready\n\
         position startpos\n\
         go infinite\n\
         stop\n\
         quit\n"
    );
    let out = drive(&session);

    let bestmoves = bestmove_lines(&out);
    assert_eq!(
        bestmoves.len(),
        1,
        "a Threads=4 go/stop/quit must emit exactly one bestmove in:\n{out}"
    );
    let startpos = parse_sfen(attic_state::STARTPOS_SFEN).expect("startpos");
    let best = bestmoves[0]
        .split_whitespace()
        .next()
        .expect("bestmove token");
    assert_ne!(best, "resign", "startpos is not mated in:\n{out}");
    let parsed = parse_usi_move(best, &startpos).expect("well-formed USI move");
    assert!(
        legal_moves(&startpos).contains(&parsed),
        "{best} is not a legal startpos move"
    );
}
