//! `go` clock values at the ends of the `i64` range, driven through the built
//! `attic` binary: each must neither crash the engine nor make it move at once.
//! The time manager clamps every clock value to a bound of 1000 days, so a
//! huge clock gives a long but finite budget, which these tests cut short with
//! `stop`, while a huge negative one is read as minus the bound, which gives a
//! short budget the search keeps to on its own.
//!
//! The network file is staged locally and never committed, so when it is absent
//! each test is skipped with a notice.

use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError};
use std::thread;
use std::time::{Duration, Instant};

const MAX: i64 = i64::MAX;
const MIN: i64 = i64::MIN;

/// How long a search on a huge clock must run before it is stopped. Any
/// budget the clamped clock gives is far longer.
const THINK: Duration = Duration::from_millis(500);

/// A loose ceiling on any wait for a reply. A single ~512-node `check_time`
/// checkpoint can take a large part of a second in an unoptimised build.
const BOUND: Duration = Duration::from_secs(10);

fn eval_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../eval")
}

/// A live USI session whose stdout lines arrive on a channel, so a test can
/// tell whether a line came before a deadline.
struct Engine {
    child: Child,
    stdin: ChildStdin,
    lines: Receiver<String>,
}

impl Engine {
    /// Spawn the engine on one worker with the real network loaded. `None`
    /// with a printed notice when the network is absent.
    fn start() -> Option<Self> {
        let dir = eval_dir();
        if !dir.join("nn.bin").exists() {
            eprintln!(
                "skipping usi_extreme_clock: {} is not present (obtained out-of-band)",
                dir.join("nn.bin").display()
            );
            return None;
        }

        let mut child = Command::new(env!("CARGO_BIN_EXE_attic"))
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .expect("spawn engine");
        let stdin = child.stdin.take().expect("stdin piped");
        let stdout = child.stdout.take().expect("stdout piped");
        let (tx, lines) = mpsc::channel();
        thread::spawn(move || {
            for line in BufReader::new(stdout).lines() {
                let Ok(line) = line else { break };
                if tx.send(line.trim_end_matches('\r').to_string()).is_err() {
                    break;
                }
            }
        });

        let mut eng = Engine {
            child,
            stdin,
            lines,
        };
        eng.send("usi");
        eng.expect_line(|l| l == "usiok", BOUND);
        eng.send(&format!(
            "setoption name EvalDir value {}",
            dir.to_str().expect("utf-8 eval dir")
        ));
        eng.send("setoption name Threads value 1");
        eng.send("isready");
        eng.expect_line(|l| l == "readyok", Duration::from_secs(60));
        Some(eng)
    }

    fn send(&mut self, cmd: &str) {
        writeln!(self.stdin, "{cmd}").expect("write engine stdin");
        self.stdin.flush().expect("flush engine stdin");
    }

    /// The first line satisfying `pred` before `deadline`, `None` once the
    /// deadline passes. Panics if the engine's stdout closes.
    fn line_before<F: Fn(&str) -> bool>(&self, pred: F, deadline: Instant) -> Option<String> {
        loop {
            let wait = deadline.saturating_duration_since(Instant::now());
            match self.lines.recv_timeout(wait) {
                Ok(l) if pred(&l) => return Some(l),
                Ok(_) => {}
                Err(RecvTimeoutError::Timeout) => return None,
                Err(RecvTimeoutError::Disconnected) => {
                    panic!("engine stdout closed (crash or early exit)")
                }
            }
        }
    }

    fn expect_line<F: Fn(&str) -> bool>(&self, pred: F, within: Duration) -> String {
        self.line_before(pred, Instant::now() + within)
            .unwrap_or_else(|| panic!("no expected line within {within:?}"))
    }

    fn expect_bestmove(&self, go: &str) -> String {
        let line = self.expect_line(|l| l.starts_with("bestmove "), BOUND);
        let mv = line.split_whitespace().nth(1).unwrap_or("").to_string();
        assert!(
            !mv.is_empty() && mv != "resign" && mv != "win",
            "`{go}` must reply with a real move, got {line:?}"
        );
        mv
    }

    fn position(&mut self, moves: &[&str]) {
        if moves.is_empty() {
            self.send("position startpos");
        } else {
            self.send(&format!("position startpos moves {}", moves.join(" ")));
        }
    }

    fn quit(mut self) {
        self.send("quit");
        let status = self.child.wait().expect("wait engine");
        assert!(status.success(), "engine exited non-zero: {status:?}");
    }
}

#[test]
#[cfg_attr(miri, ignore)]
fn the_largest_clock_values_think_until_stopped() {
    let Some(mut eng) = Engine::start() else {
        return;
    };

    // White to move after `7g7f`, so the `w` tokens are the ones read.
    let cases: &[(&[&str], String)] = &[
        (&[], format!("go btime {MAX} byoyomi 1000")),
        (&["7g7f"], format!("go wtime {MAX} byoyomi 1000")),
        (&[], format!("go btime 60000 binc {MAX}")),
        (&["7g7f"], format!("go wtime 60000 winc {MAX}")),
        (&[], format!("go btime 0 byoyomi {MAX}")),
        (&[], format!("go movetime {MAX}")),
        (&[], format!("go rtime {MAX}")),
        (
            &[],
            format!("go btime {MAX} wtime {MAX} binc {MAX} winc {MAX} byoyomi {MAX}"),
        ),
    ];
    for (moves, go) in cases {
        eng.send("usinewgame");
        eng.position(moves);
        eng.send(go);
        let early = eng.line_before(|l| l.starts_with("bestmove "), Instant::now() + THINK);
        assert_eq!(early, None, "`{go}` must not move at once");
        eng.send("stop");
        eng.expect_bestmove(go);
    }

    eng.quit();
}

#[test]
#[cfg_attr(miri, ignore)]
fn the_most_negative_clock_values_move_on_their_own() {
    let Some(mut eng) = Engine::start() else {
        return;
    };

    // The protocol has no negative clock values. A negative main clock or
    // `movetime` is read as 1 ms; a negative increment, byoyomi or `rtime` is
    // used as given, which leaves a short budget of the minimum think time or
    // less.
    let cases = [
        format!("go btime 60000 binc {MIN}"),
        format!("go btime 60000 byoyomi {MIN}"),
        format!("go btime {MIN} byoyomi 1000"),
        format!("go movetime {MIN}"),
        format!("go rtime {MIN}"),
        format!("go btime {MIN} wtime {MIN} binc {MIN} winc {MIN} byoyomi {MIN}"),
    ];
    for go in &cases {
        eng.send("usinewgame");
        eng.position(&[]);
        eng.send(go);
        eng.expect_bestmove(go);
    }

    eng.quit();
}
