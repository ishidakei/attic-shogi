//! Smoke test: spawn the built `attic` binary, drive a USI handshake, and
//! assert on stdout.

use std::io::Write;
use std::process::{Command, Stdio};

#[test]
#[cfg_attr(miri, ignore)]
fn handshake_round_trip_via_spawned_binary() {
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
            .write_all(b"usi\nsetoption name USI_Hash value 256\nisready\nquit\n")
            .expect("write stdin");
    }

    let out = child.wait_with_output().expect("wait");
    assert!(
        out.status.success(),
        "engine exited non-zero: {:?}",
        out.status
    );

    let stdout = String::from_utf8(out.stdout).expect("utf-8 stdout");

    assert!(stdout.contains("id name "), "missing id name in:\n{stdout}");
    assert!(
        stdout.contains("id author "),
        "missing id author in:\n{stdout}"
    );
    assert!(
        stdout.contains("option name USI_Hash type spin"),
        "missing USI_Hash option in:\n{stdout}"
    );
    assert!(
        stdout.contains("option name Threads type spin"),
        "missing Threads option in:\n{stdout}"
    );
    assert!(
        stdout.contains("option name MultiPV type spin"),
        "missing MultiPV option in:\n{stdout}"
    );
    assert!(
        stdout.contains("option name EvalDir type string"),
        "missing EvalDir option in:\n{stdout}"
    );
    assert!(
        stdout.contains("option name BookFile type combo default no_book"),
        "missing BookFile combo option in:\n{stdout}"
    );
    assert!(
        stdout.contains("option name USI_OwnBook type check default true"),
        "missing USI_OwnBook option in:\n{stdout}"
    );
    assert!(stdout.contains("usiok\n"), "missing usiok in:\n{stdout}");
    // No `EvalDir` was set, so `isready` looks for the default `eval/nn.bin`,
    // absent in the spawned binary's CWD: a load-failure notice and no
    // `readyok`.
    assert!(
        stdout.contains("info string eval load failed:"),
        "missing eval-load-failure notice in:\n{stdout}"
    );
    assert!(
        !stdout.contains("readyok"),
        "readyok must not appear on a failed load in:\n{stdout}"
    );
    assert!(
        !stdout.contains("rejected"),
        "unexpected rejection in:\n{stdout}"
    );
}
