//! Integration test: a real-network self-play session against the built
//! `attic` binary.
//!
//! The SFNN-1536 network is staged locally at
//! `eval/nn.bin` and is never committed. When it is
//! absent (a checkout without it staged) the test prints a notice and passes, so
//! the default `cargo test` run stays green everywhere — the same skip pattern
//! `attic-eval` and `attic-search`'s integration tests use.
//!
//! When present, it spawns the engine, loads the network via `EvalDir`, and
//! drives a ~40-ply self-play loop over one live USI session: `position
//! startpos moves <accumulated>` + `go depth 1`, reusing the session. Every
//! `bestmove` must be legal for the running position, the process must never
//! crash, and it must exit cleanly on `quit`. A `bestmove resign` or
//! `bestmove win` (a mate or a declaration win can happen) ends the loop early.

use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

use attic_state::{Move, Position, parse_usi_move};

const MAX_PLIES: usize = 40;

fn eval_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../eval")
}

/// Read child stdout lines until one satisfies `pred`; returns the matched
/// (trimmed) line, or `None` on EOF.
fn read_until<F: Fn(&str) -> bool>(reader: &mut BufReader<ChildStdout>, pred: F) -> Option<String> {
    let mut line = String::new();
    loop {
        line.clear();
        let n = reader.read_line(&mut line).expect("read engine stdout");
        if n == 0 {
            return None;
        }
        let trimmed = line.trim_end_matches(['\r', '\n']);
        if pred(trimmed) {
            return Some(trimmed.to_string());
        }
    }
}

fn send(stdin: &mut ChildStdin, cmd: &str) {
    stdin.write_all(cmd.as_bytes()).expect("write engine stdin");
    stdin.flush().expect("flush engine stdin");
}

fn legal_moves(p: &Position) -> Vec<Move> {
    let mut moves = Vec::new();
    p.generate_legal_all(&mut moves);
    moves
}

#[test]
#[cfg_attr(miri, ignore)]
fn real_network_self_play_stays_legal_and_exits_cleanly() {
    let dir = eval_dir();
    if !dir.join("nn.bin").exists() {
        eprintln!(
            "skipping real_network_self_play_stays_legal_and_exits_cleanly: {} is not present (staged only on the dev VM)",
            dir.join("nn.bin").display()
        );
        return;
    }

    let exe = env!("CARGO_BIN_EXE_attic");
    let mut child: Child = Command::new(exe)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn engine");

    let mut stdin = child.stdin.take().expect("stdin piped");
    let mut stdout = BufReader::new(child.stdout.take().expect("stdout piped"));

    send(&mut stdin, "usi\n");
    read_until(&mut stdout, |l| l == "usiok").expect("usiok before EOF");
    let eval_dir_arg = dir.to_str().expect("utf-8 eval dir");
    send(
        &mut stdin,
        &format!("setoption name EvalDir value {eval_dir_arg}\n"),
    );
    send(&mut stdin, "isready\n");
    let ack = read_until(&mut stdout, |l| {
        l == "readyok" || l.starts_with("info string eval load failed")
    })
    .expect("readyok or load-failure before EOF");
    assert_eq!(
        ack, "readyok",
        "real network must load; got a load failure instead"
    );

    let mut pos = Position::startpos();
    let mut moves_usi: Vec<String> = Vec::new();
    for ply in 0..MAX_PLIES {
        let position_cmd = if moves_usi.is_empty() {
            "position startpos\n".to_string()
        } else {
            format!("position startpos moves {}\n", moves_usi.join(" "))
        };
        send(&mut stdin, &position_cmd);
        send(&mut stdin, "go depth 1\n");

        let bestmove =
            read_until(&mut stdout, |l| l.starts_with("bestmove ")).expect("bestmove before EOF");
        // `bestmove <move> [ponder <move>]`; only the played move matters here.
        let mv_str = bestmove
            .strip_prefix("bestmove ")
            .expect("bestmove prefix")
            .split_whitespace()
            .next()
            .expect("bestmove token")
            .to_string();

        if mv_str == "resign" || mv_str == "win" {
            // A mate / no-legal-move (resign) or a declaration win ends the game
            // early — both are valid endings.
            break;
        }

        let mv = parse_usi_move(&mv_str, &pos)
            .unwrap_or_else(|e| panic!("ply {ply}: malformed bestmove {mv_str:?}: {e}"));
        assert!(
            legal_moves(&pos).contains(&mv),
            "ply {ply}: bestmove {mv_str} is not legal for the running position"
        );
        pos.do_move(mv);
        moves_usi.push(mv_str);
    }

    send(&mut stdin, "quit\n");
    drop(stdin);
    let status = child.wait().expect("wait for engine");
    assert!(status.success(), "engine exited non-zero: {status:?}");
}
