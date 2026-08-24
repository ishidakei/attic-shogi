//! End-to-end USI session tests for depth wiring, asynchronous stop, and time
//! management, driving the built `attic` binary as a subprocess.
//!
//! These pin the USI layer to the already-gated search: `go depth 2` / `go
//! depth 3` must reproduce the reference fixtures through the real driver, and
//! the time-managed forms (`go infinite` + `stop`, `go movetime`, a Fischer
//! mini-game) must each terminate promptly with exactly one `bestmove`.
//!
//! Like the other real-network tests, the whole file is skipped with a notice
//! when `nn.bin` is absent (a checkout without it staged), so the default
//! `cargo test` run stays green everywhere.

use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::time::{Duration, Instant};

use serde::Deserialize;

fn eval_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../eval")
}

fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures")
}

/// The gated subset of a search fixture. The depth-2/3 test asserts `bestmove`
/// plus the `nodes` count and the centipawn `score` parsed from the final `info`
/// line — the class of drift a transposition-table refactor could introduce.
#[derive(Debug, Deserialize)]
struct Fixture {
    #[serde(default)]
    moves: Vec<String>,
    bestmove: String,
    nodes: u64,
    score: Score,
}

/// The `score` object of a fixture. Every gated fixture is a non-mate centipawn
/// score, so only the `cp` arm is modelled.
#[derive(Debug, Deserialize)]
struct Score {
    cp: i64,
}

fn load_fixture(rel: &str) -> Fixture {
    let path = fixtures_dir().join(rel);
    let raw =
        std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    serde_json::from_str(&raw).unwrap_or_else(|e| panic!("parse {rel}: {e}"))
}

/// A live USI session over the spawned engine binary.
struct Engine {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
}

impl Engine {
    /// Spawn the engine, load the real network via `EvalDir`, and complete the
    /// `usi` / `isready` handshake. Returns `None` (with a printed notice) when
    /// the network is absent, so callers can skip.
    fn start() -> Option<Self> {
        let dir = eval_dir();
        if !dir.join("nn.bin").exists() {
            eprintln!(
                "skipping usi_time_management: {} is not present (staged only on the dev VM)",
                dir.join("nn.bin").display()
            );
            return None;
        }

        let exe = env!("CARGO_BIN_EXE_attic");
        let mut child = Command::new(exe)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("spawn engine");
        let stdin = child.stdin.take().expect("stdin piped");
        let stdout = BufReader::new(child.stdout.take().expect("stdout piped"));

        let mut eng = Engine {
            child,
            stdin,
            stdout,
        };
        eng.send("usi");
        eng.read_until(|l| l == "usiok");
        eng.send(&format!(
            "setoption name EvalDir value {}",
            dir.to_str().expect("utf-8 eval dir")
        ));
        eng.send("isready");
        eng.read_until(|l| l == "readyok")
            .expect("network must load (readyok)");
        Some(eng)
    }

    fn send(&mut self, cmd: &str) {
        self.stdin
            .write_all(cmd.as_bytes())
            .expect("write engine stdin");
        self.stdin.write_all(b"\n").expect("write newline");
        self.stdin.flush().expect("flush engine stdin");
    }

    /// Read (trimmed) lines until one satisfies `pred`; returns it, or `None` on
    /// EOF.
    fn read_until<F: Fn(&str) -> bool>(&mut self, pred: F) -> Option<String> {
        let mut line = String::new();
        loop {
            line.clear();
            let n = self
                .stdout
                .read_line(&mut line)
                .expect("read engine stdout");
            if n == 0 {
                return None;
            }
            let trimmed = line.trim_end_matches(['\r', '\n']);
            if pred(trimmed) {
                return Some(trimmed.to_string());
            }
        }
    }

    /// Read the `info` line of the given iterative-deepening `depth` and pull out
    /// its `nodes` count and centipawn `score cp` value. With `PvInterval 0` every
    /// completed iteration prints a line, not just the last one, so callers must
    /// target the depth they mean explicitly.
    fn read_info_nodes_cp(&mut self, depth: u32) -> (u64, i64) {
        let prefix = format!("info depth {depth} ");
        let line = self
            .read_until(|l| l.starts_with(&prefix) && l.contains(" nodes "))
            .expect("a search info line at the target depth must arrive");
        let toks: Vec<&str> = line.split_whitespace().collect();
        let after = |key: &str| -> &str {
            let i = toks
                .iter()
                .position(|&t| t == key)
                .unwrap_or_else(|| panic!("`{key}` token missing in info line: {line:?}"));
            toks.get(i + 1)
                .unwrap_or_else(|| panic!("value after `{key}` missing in info line: {line:?}"))
        };
        // `score cp <n>`: guard that the score is centipawns, not `mate`.
        assert_eq!(
            after("score"),
            "cp",
            "gated fixtures are centipawn scores, got: {line:?}"
        );
        let nodes = after("nodes").parse().expect("nodes is a u64");
        let cp = after("cp").parse().expect("cp is an i64");
        (nodes, cp)
    }

    /// Read the next `bestmove` line and return just the move token (dropping any
    /// ` ponder …` suffix).
    fn read_bestmove(&mut self) -> String {
        let line = self
            .read_until(|l| l.starts_with("bestmove "))
            .expect("a bestmove must arrive");
        line.strip_prefix("bestmove ")
            .unwrap()
            .split_whitespace()
            .next()
            .expect("bestmove token")
            .to_string()
    }

    fn quit(mut self) {
        self.send("quit");
        let status = self.child.wait().expect("wait engine");
        assert!(status.success(), "engine exited non-zero: {status:?}");
    }
}

#[test]
#[cfg_attr(miri, ignore)]
fn go_depth_2_and_3_match_fixtures_via_binary() {
    let Some(mut eng) = Engine::start() else {
        return;
    };

    // Pin the single-worker search: the default is 4 workers, which share and
    // pollute the TT once helpers really search, so any fixture
    // (nodes / cp / bestmove) assertion must run on one worker.
    eng.send("setoption name Threads value 1");
    // `PvInterval 0` retains per-iteration `info` output regardless of wall-clock
    // timing (the default 300 ms would suppress intermediate lines on a fast
    // search and print them on a slow one). The reads target the final depth.
    eng.send("setoption name PvInterval value 0");

    // depth 2: position startpos moves 7g7f, go depth 2. Assert the info line's
    // nodes and cp score as well as the bestmove.
    let d2 = load_fixture("search-depth2/startpos-7g7f.json");
    eng.send("usinewgame");
    eng.send(&format!("position startpos moves {}", d2.moves.join(" ")));
    eng.send("go depth 2");
    let (d2_nodes, d2_cp) = eng.read_info_nodes_cp(2);
    assert_eq!(
        d2_nodes, d2.nodes,
        "go depth 2 nodes must match the fixture"
    );
    assert_eq!(
        d2_cp, d2.score.cp,
        "go depth 2 cp score must match the fixture"
    );
    assert_eq!(
        eng.read_bestmove(),
        d2.bestmove,
        "go depth 2 bestmove must match the reference fixture"
    );

    // depth 3: position startpos, go depth 3.
    let d3 = load_fixture("search/startpos.json");
    eng.send("usinewgame");
    eng.send("position startpos");
    eng.send("go depth 3");
    let (d3_nodes, d3_cp) = eng.read_info_nodes_cp(3);
    assert_eq!(
        d3_nodes, d3.nodes,
        "go depth 3 nodes must match the fixture"
    );
    assert_eq!(
        d3_cp, d3.score.cp,
        "go depth 3 cp score must match the fixture"
    );
    assert_eq!(
        eng.read_bestmove(),
        d3.bestmove,
        "go depth 3 bestmove must match the reference fixture"
    );

    eng.quit();
}

#[test]
#[cfg_attr(miri, ignore)]
fn threads_cycle_single_thread_matches_fixture_multi_thread_is_legal() {
    // The Lazy-SMP re-scope of the Threads-cycle gate. Cycling `setoption name
    // Threads value N` resizes the worker pool; on the `Threads=1` leg the same
    // `position` + `go depth 2` must still reproduce the reference fixture
    // exactly (nodes / cp / bestmove), but once helpers really search the
    // `Threads>1` legs are nondeterministic (the workers share and pollute the
    // TT), so those legs assert only exactly one legal bestmove and clean
    // termination — a leaked helper would hang the `quit` `wait` below.
    let Some(mut eng) = Engine::start() else {
        return;
    };

    // `PvInterval 0` keeps per-iteration `info` output deterministic across the
    // Threads legs (see `go_depth_2_and_3_match_fixtures_via_binary`).
    eng.send("setoption name PvInterval value 0");
    let d2 = load_fixture("search-depth2/startpos-7g7f.json");
    for threads in [1u32, 4, 2] {
        eng.send(&format!("setoption name Threads value {threads}"));
        eng.send("usinewgame");
        eng.send(&format!("position startpos moves {}", d2.moves.join(" ")));
        eng.send("go depth 2");
        let best = eng.read_bestmove();
        if threads == 1 {
            // Single worker: bit-identical to the reference fixture. The info
            // line is read before the bestmove, so re-drive that leg to read it.
            eng.send("usinewgame");
            eng.send(&format!("position startpos moves {}", d2.moves.join(" ")));
            eng.send("go depth 2");
            let (nodes, cp) = eng.read_info_nodes_cp(2);
            assert_eq!(nodes, d2.nodes, "Threads=1: node count must match fixture");
            assert_eq!(cp, d2.score.cp, "Threads=1: cp score must match fixture");
            assert_eq!(
                eng.read_bestmove(),
                d2.bestmove,
                "Threads=1: bestmove must match fixture"
            );
        } else {
            // Multi worker: assert legality, not equality.
            assert_legal_move_after(&d2.moves, &best);
        }
    }

    eng.quit();
}

/// Multi-thread session smoke tests, all at `Threads=2`.
/// Each drives the spawned binary; each asserts exactly one *legal* bestmove and
/// prompt termination (the search results are nondeterministic under Lazy SMP).
#[test]
#[cfg_attr(miri, ignore)]
fn threads2_go_movetime_and_depth_and_infinite_and_fischer() {
    let Some(mut eng) = Engine::start() else {
        return;
    };
    eng.send("setoption name Threads value 2");

    // The deadline is polled only at ~512-node `check_time` checkpoints on the
    // main worker; with two workers contending for cores in an *unoptimised test
    // build*, a single checkpoint can be several seconds. These bounds are
    // therefore deliberately loose — they prove the search self-terminates near
    // its budget (never running to the depth-245 ceiling, which would take far
    // longer than the bound) and returns exactly one legal bestmove per move.
    let bound = Duration::from_secs(10);

    // (a) go movetime 300 → one legal bestmove within a generous bound.
    eng.send("usinewgame");
    eng.send("position startpos");
    let t = Instant::now();
    eng.send("go movetime 300");
    let best = eng.read_bestmove();
    assert!(
        t.elapsed() < bound,
        "Threads=2 go movetime 300 took too long"
    );
    assert_legal_move_after(&[], &best);

    // (c) go depth 3 → a legal bestmove (result nondeterministic under SMP).
    eng.send("usinewgame");
    eng.send("position startpos");
    eng.send("go depth 3");
    let best = eng.read_bestmove();
    assert_legal_move_after(&[], &best);

    // (d) go infinite + stop → one prompt bestmove.
    eng.send("usinewgame");
    eng.send("position startpos");
    eng.send("go infinite");
    std::thread::sleep(Duration::from_millis(250));
    let t = Instant::now();
    eng.send("stop");
    let best = eng.read_bestmove();
    assert!(
        t.elapsed() < bound,
        "Threads=2 bestmove after stop took too long"
    );
    assert_legal_move_after(&[], &best);

    // (b) Fischer mini-game → every move meets its deadline with one bestmove.
    const CLOCK: u64 = 300;
    const INC: u64 = 200;
    eng.send("usinewgame");
    let mut moves: Vec<String> = Vec::new();
    for ply in 0..6 {
        if moves.is_empty() {
            eng.send("position startpos");
        } else {
            eng.send(&format!("position startpos moves {}", moves.join(" ")));
        }
        let t = Instant::now();
        eng.send(&format!(
            "go btime {CLOCK} wtime {CLOCK} binc {INC} winc {INC}"
        ));
        let best = eng.read_bestmove();
        assert!(t.elapsed() < bound, "Threads=2 ply {ply}: missed deadline");
        if best == "resign" || best == "win" {
            break;
        }
        assert_legal_move_after(&moves, &best);
        moves.push(best);
    }
    assert!(
        moves.len() >= 2,
        "Threads=2 mini-game should play several moves, got {moves:?}"
    );

    eng.quit();
}

/// Assert `best` is a legal move in the position reached by applying `setup`
/// (USI move strings) from startpos. `resign` / `win` are *rejected* here: every
/// call site breaks out of its loop on a terminal reply before reaching this
/// guard, so a `resign` / `win` arriving here is an unexpected result and fails
/// the assertion below.
fn assert_legal_move_after(setup: &[String], best: &str) {
    assert!(
        !best.is_empty() && best != "resign" && best != "win",
        "expected a real move, got {best:?}"
    );
    let mut pos = attic_state::parse_sfen(attic_state::STARTPOS_SFEN).expect("startpos SFEN");
    for m in setup {
        let mv = attic_state::parse_usi_move(m, &pos).expect("legal setup move");
        pos.do_move(mv);
    }
    let mv = attic_state::parse_usi_move(best, &pos)
        .unwrap_or_else(|_| panic!("bestmove {best:?} is not a well-formed USI move"));
    let mut legal = Vec::new();
    pos.generate_legal_all(&mut legal);
    assert!(legal.contains(&mv), "bestmove {best:?} is not legal");
}

#[test]
#[cfg_attr(miri, ignore)]
fn go_infinite_then_stop_yields_one_prompt_bestmove() {
    let Some(mut eng) = Engine::start() else {
        return;
    };

    // Single-worker timing test: pin Threads=1 so the ~512-node `check_time`
    // cadence is not slowed by helper CPU contention. The
    // multi-thread timing coverage lives in `threads2_*`.
    eng.send("setoption name Threads value 1");
    eng.send("usinewgame");
    eng.send("position startpos");
    eng.send("go infinite");

    // Let the search run, then ask it to stop and time how long bestmove takes.
    std::thread::sleep(Duration::from_millis(250));
    let t = Instant::now();
    eng.send("stop");
    let best = eng.read_bestmove();
    let elapsed = t.elapsed();

    assert!(
        !best.is_empty() && best != "resign",
        "go infinite on startpos must return a real move, got {best:?}"
    );
    // Bounded by the ~512-node check granularity, not by an iteration boundary.
    // Generous for a debug build under load.
    assert!(
        elapsed < Duration::from_secs(3),
        "bestmove after stop took too long: {elapsed:?}"
    );

    eng.quit();
}

#[test]
#[cfg_attr(miri, ignore)]
fn go_movetime_returns_within_a_generous_bound() {
    let Some(mut eng) = Engine::start() else {
        return;
    };

    // Single-worker timing test (see the note in the go-infinite test above).
    eng.send("setoption name Threads value 1");
    eng.send("usinewgame");
    eng.send("position startpos");
    let t = Instant::now();
    eng.send("go movetime 300");
    let best = eng.read_bestmove();
    let elapsed = t.elapsed();

    assert!(
        !best.is_empty() && best != "resign",
        "go movetime on startpos must return a real move, got {best:?}"
    );
    // 300 ms budget; allow ample slack for process scheduling in a debug build.
    assert!(
        elapsed < Duration::from_secs(3),
        "go movetime 300 took too long: {elapsed:?}"
    );

    eng.quit();
}

#[test]
#[cfg_attr(miri, ignore)]
fn fischer_mini_game_makes_every_deadline_with_one_bestmove_each() {
    let Some(mut eng) = Engine::start() else {
        return;
    };

    // Small Fischer budgets: each move's hard deadline is remaining + increment,
    // so a bestmove must arrive well within that. We replay the engine's own
    // choices to walk a short game.
    const BTIME: u64 = 300;
    const WTIME: u64 = 300;
    const INC: u64 = 200;
    // A generous ceiling on the per-move wall clock. The engine's hard deadline
    // is (clock + increment - margin) ≈ 460 ms, but the deadline is polled only
    // at ~512-node `check_time` checkpoints; in an *unoptimised test build* a
    // checkpoint is ~0.5 s of compute (~1000 nodes/s), so a move can overshoot
    // the nominal budget by up to one checkpoint. In a release build a checkpoint
    // is sub-millisecond and the budget is met to the millisecond. This bound is
    // therefore deliberately loose — it proves the engine self-terminates near
    // its budget (never running to the depth ceiling) and always returns exactly
    // one bestmove, which is what "no missed deadline" means for a debug run.
    let per_move_bound = Duration::from_secs(3);

    // Single-worker timing test (see the note in the go-infinite test above).
    eng.send("setoption name Threads value 1");
    eng.send("usinewgame");
    let mut moves: Vec<String> = Vec::new();

    for ply in 0..6 {
        if moves.is_empty() {
            eng.send("position startpos");
        } else {
            eng.send(&format!("position startpos moves {}", moves.join(" ")));
        }
        let t = Instant::now();
        eng.send(&format!(
            "go btime {BTIME} wtime {WTIME} binc {INC} winc {INC}"
        ));
        let best = eng.read_bestmove();
        let elapsed = t.elapsed();

        assert!(
            elapsed < per_move_bound,
            "ply {ply}: missed deadline ({elapsed:?} >= {per_move_bound:?})"
        );

        if best == "resign" || best == "win" {
            break; // a terminal result ends the mini-game early.
        }
        moves.push(best);
    }

    // At least a couple of real plies were played under the clock.
    assert!(
        moves.len() >= 2,
        "expected the mini-game to play several moves, got {moves:?}"
    );

    eng.quit();
}

#[test]
#[cfg_attr(miri, ignore)]
fn byoyomi_mini_game_makes_every_deadline_with_one_bestmove_each() {
    let Some(mut eng) = Engine::start() else {
        return;
    };

    // A byoyomi game with the main clock exhausted (`btime 0 wtime 0`): every move
    // has only the byoyomi period. This is the reference "final push" shape
    // (`timeman.cpp`) — with `time[us] < byoyomi * 1.2` the manager spends
    // the byoyomi. The nominal per-move deadline is `time + byoyomi == 1000 ms`.
    //
    // The per-move wall bound is deliberately loose for the same reason as the
    // Fischer test above: in an *unoptimised test build* the deadline is polled
    // only at ~512-node `check_time` checkpoints (~0.5 s of compute each), so a
    // move can overshoot its nominal budget by roughly one checkpoint. The bound
    // proves the engine self-terminates on the byoyomi clock (never running to the
    // depth ceiling) and always returns exactly one bestmove — "no missed
    // deadline" for a debug run. In a release build the byoyomi is met to the
    // millisecond (the `TimeManagement` maths is unit-tested in `attic_search::timeman`).
    const BYOYOMI: u64 = 1000;
    let per_move_bound = Duration::from_secs(3);

    // Single-worker timing test (see the note in the go-infinite test above).
    eng.send("setoption name Threads value 1");
    eng.send("usinewgame");
    let mut moves: Vec<String> = Vec::new();

    for ply in 0..6 {
        if moves.is_empty() {
            eng.send("position startpos");
        } else {
            eng.send(&format!("position startpos moves {}", moves.join(" ")));
        }
        let t = Instant::now();
        eng.send(&format!("go btime 0 wtime 0 byoyomi {BYOYOMI}"));
        let best = eng.read_bestmove();
        let elapsed = t.elapsed();

        assert!(
            elapsed < per_move_bound,
            "ply {ply}: byoyomi move missed deadline ({elapsed:?} >= {per_move_bound:?})"
        );

        if best == "resign" || best == "win" {
            break; // a terminal result ends the mini-game early.
        }
        moves.push(best);
    }

    assert!(
        moves.len() >= 2,
        "expected the byoyomi mini-game to play several moves, got {moves:?}"
    );

    eng.quit();
}
