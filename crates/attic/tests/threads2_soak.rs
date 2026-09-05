//! Multi-thread self-play soak: several workers share the transposition table
//! through relaxed atomics, and this drives the built `attic` binary at
//! `Threads=2` through a long stream of real self-play games, asserting exactly
//! one legal `bestmove` per `go`, no panic, no hang, and a clean `quit`.
//!
//! `#[ignore]`-gated so the default `cargo test` stays fast. It needs the real
//! network, staged locally and never committed; when that is absent it prints a
//! notice and passes.
//!
//! Run it in a release build, optionally overriding the ~10-minute default
//! duration:
//!
//! ```text
//! SOAK_SECS=120 cargo test --release -p attic --test threads2_soak -- --ignored --nocapture
//! ```

use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError};
use std::thread;
use std::time::{Duration, Instant};

use attic_state::{Move, Position, parse_usi_move};

const DEFAULT_SOAK_SECS: u64 = 600;

/// Per-move watchdog. A single `go` at this time control returns in well under
/// a second, so anything past this is a hang rather than a slow search.
const MOVE_WATCHDOG: Duration = Duration::from_secs(30);

/// Restart a fresh game after this many plies even without a terminal result,
/// so repetition-heavy lines cannot pin one game forever.
const MAX_PLIES_PER_GAME: usize = 256;

const TC_GO: &str = "go btime 300 wtime 300 binc 200 winc 200";

fn eval_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../eval")
}

fn legal_moves(p: &Position) -> Vec<Move> {
    let mut moves = Vec::new();
    p.generate_legal_all(&mut moves);
    moves
}

fn send(stdin: &mut ChildStdin, cmd: &str) {
    stdin.write_all(cmd.as_bytes()).expect("write engine stdin");
    stdin.write_all(b"\n").expect("write newline");
    stdin.flush().expect("flush engine stdin");
}

/// A spawned engine plus a background reader funnelling every stdout line into a
/// channel, so the soak loop can apply a `recv_timeout` watchdog to each move.
struct Session {
    child: Child,
    stdin: ChildStdin,
    lines: Receiver<String>,
}

impl Session {
    fn start() -> Option<Session> {
        let exe = env!("CARGO_BIN_EXE_attic");
        let mut child: Child = Command::new(exe)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("spawn engine");

        let stdin = child.stdin.take().expect("stdin piped");
        let stdout = child.stdout.take().expect("stdout piped");
        let (tx, rx) = mpsc::channel::<String>();
        thread::spawn(move || {
            let mut reader = BufReader::new(stdout);
            let mut line = String::new();
            loop {
                line.clear();
                match reader.read_line(&mut line) {
                    Ok(0) | Err(_) => break, // EOF or read error: drop the sender.
                    Ok(_) => {
                        let trimmed = line.trim_end_matches(['\r', '\n']).to_string();
                        if tx.send(trimmed).is_err() {
                            break;
                        }
                    }
                }
            }
        });

        Some(Session {
            child,
            stdin,
            lines: rx,
        })
    }

    /// Read stdout lines until one satisfies `pred`, applying the per-move
    /// watchdog. Panics on a hang or on child death.
    fn read_until<F: Fn(&str) -> bool>(&self, pred: F) -> String {
        loop {
            match self.lines.recv_timeout(MOVE_WATCHDOG) {
                Ok(l) => {
                    if pred(&l) {
                        return l;
                    }
                }
                Err(RecvTimeoutError::Timeout) => {
                    panic!(
                        "watchdog: engine produced no matching line within {MOVE_WATCHDOG:?} (hang)"
                    )
                }
                Err(RecvTimeoutError::Disconnected) => {
                    panic!("engine stdout closed unexpectedly (crash / early exit)")
                }
            }
        }
    }
}

#[test]
#[ignore = "long-running multi-thread soak; run explicitly"]
fn threads2_self_play_soak_stays_legal_and_stable() {
    let dir = eval_dir();
    if !dir.join("nn.bin").exists() {
        eprintln!(
            "skipping threads2_self_play_soak_stays_legal_and_stable: {} is not present (obtained out-of-band)",
            dir.join("nn.bin").display()
        );
        return;
    }

    let soak_secs = std::env::var("SOAK_SECS")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(DEFAULT_SOAK_SECS);
    let deadline = Instant::now() + Duration::from_secs(soak_secs);

    let mut sess = Session::start().expect("engine session");

    send(&mut sess.stdin, "usi");
    sess.read_until(|l| l == "usiok");
    let eval_dir_arg = dir.to_str().expect("utf-8 eval dir");
    send(
        &mut sess.stdin,
        &format!("setoption name EvalDir value {eval_dir_arg}"),
    );
    send(&mut sess.stdin, "setoption name Threads value 2");
    send(&mut sess.stdin, "isready");
    let ack = sess.read_until(|l| l == "readyok" || l.starts_with("info string eval load failed"));
    assert_eq!(
        ack, "readyok",
        "real network must load at Threads=2; got a load failure instead"
    );

    let mut games = 0usize;
    let mut total_moves = 0usize;
    let start = Instant::now();

    'soak: while Instant::now() < deadline {
        send(&mut sess.stdin, "usinewgame");
        let mut pos = Position::startpos();
        let mut moves_usi: Vec<String> = Vec::new();
        games += 1;

        for ply in 0..MAX_PLIES_PER_GAME {
            if Instant::now() >= deadline {
                break 'soak;
            }
            let position_cmd = if moves_usi.is_empty() {
                "position startpos".to_string()
            } else {
                format!("position startpos moves {}", moves_usi.join(" "))
            };
            send(&mut sess.stdin, &position_cmd);
            send(&mut sess.stdin, TC_GO);

            let bestmove = sess.read_until(|l| l.starts_with("bestmove "));
            let mv_str = bestmove
                .strip_prefix("bestmove ")
                .expect("bestmove prefix")
                .split_whitespace()
                .next()
                .expect("bestmove token")
                .to_string();

            if mv_str == "resign" || mv_str == "win" {
                break;
            }

            let mv = parse_usi_move(&mv_str, &pos).unwrap_or_else(|e| {
                panic!("game {games} ply {ply}: malformed bestmove {mv_str:?}: {e}")
            });
            assert!(
                legal_moves(&pos).contains(&mv),
                "game {games} ply {ply}: bestmove {mv_str} is not legal for the running position"
            );
            pos.do_move(mv);
            moves_usi.push(mv_str);
            total_moves += 1;
        }
    }

    send(&mut sess.stdin, "quit");
    drop(sess.stdin);
    let status = sess.child.wait().expect("wait for engine");
    assert!(status.success(), "engine exited non-zero: {status:?}");

    let elapsed = start.elapsed();
    eprintln!(
        "threads2 soak summary: {games} games, {total_moves} moves, {:.1}s (target {soak_secs}s)",
        elapsed.as_secs_f64()
    );
    assert!(games > 0 && total_moves > 0, "soak played no moves");
}
