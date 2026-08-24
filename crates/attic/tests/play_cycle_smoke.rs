//! Cross-crate smoke test: spawn the built `attic` binary, drive a multi-
//! cycle USI session through stdin, capture stdout, assert each `bestmove`
//! line is well-formed and the binary exits 0. Confirms the `main` ↔
//! `UsiDriver` wiring carries the new position/go path end-to-end.

use std::io::Write;
use std::process::{Command, Stdio};

fn is_well_formed_usi_move(s: &str) -> bool {
    if s == "resign" {
        return true;
    }
    let b = s.as_bytes();
    if b.len() == 4 && b[1] == b'*' {
        return matches!(b[0], b'P' | b'L' | b'N' | b'S' | b'G' | b'B' | b'R')
            && (b'1'..=b'9').contains(&b[2])
            && (b'a'..=b'i').contains(&b[3]);
    }
    if b.len() != 4 && b.len() != 5 {
        return false;
    }
    let file_ok = |c: u8| (b'1'..=b'9').contains(&c);
    let rank_ok = |c: u8| (b'a'..=b'i').contains(&c);
    if !(file_ok(b[0]) && rank_ok(b[1]) && file_ok(b[2]) && rank_ok(b[3])) {
        return false;
    }
    if b.len() == 5 && b[4] != b'+' {
        return false;
    }
    true
}

#[test]
#[cfg_attr(miri, ignore)]
fn multi_cycle_play_via_spawned_binary() {
    let exe = env!("CARGO_BIN_EXE_attic");
    let mut child = Command::new(exe)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn engine");

    {
        let stdin = child.stdin.as_mut().expect("stdin piped");
        stdin
            .write_all(
                b"usi\n\
                  isready\n\
                  position startpos\n\
                  go depth 1\n\
                  position startpos moves 7g7f\n\
                  go depth 1\n\
                  quit\n",
            )
            .expect("write stdin");
    }

    let out = child.wait_with_output().expect("wait");
    assert!(
        out.status.success(),
        "engine exited non-zero: {:?}",
        out.status
    );

    let stdout = String::from_utf8(out.stdout).expect("utf-8 stdout");
    assert!(stdout.contains("usiok\n"), "missing usiok in:\n{stdout}");
    // No EvalDir set → default `eval/nn.bin` is absent in the spawned binary's
    // CWD → load fails, no readyok, and each `go` resigns. This confirms the
    // main ↔ UsiDriver wiring survives an unloaded network end-to-end; the
    // positive multi-cycle play path lives in engine/tests/real_network_selfplay.
    assert!(
        stdout.contains("info string eval load failed:"),
        "missing eval-load-failure notice in:\n{stdout}"
    );
    assert!(
        !stdout.contains("readyok"),
        "unexpected readyok in:\n{stdout}"
    );

    let bestmoves: Vec<&str> = stdout
        .lines()
        .filter_map(|l| l.strip_prefix("bestmove "))
        .collect();
    assert_eq!(
        bestmoves,
        vec!["resign", "resign"],
        "expected two resign bestmoves in:\n{stdout}"
    );
    for m in &bestmoves {
        assert!(
            is_well_formed_usi_move(m),
            "malformed bestmove {m:?} in:\n{stdout}",
        );
    }
}
