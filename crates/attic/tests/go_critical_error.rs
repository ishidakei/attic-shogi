//! A malformed `go` argument terminates the engine with the reference's
//! `CRITICAL ERROR` line and exit status 1; the commands after it are never
//! read.

use std::io::Write;
use std::process::{Command, Stdio};

fn run_engine(input: &[u8]) -> std::process::Output {
    let exe = env!("CARGO_BIN_EXE_attic");
    let mut child = Command::new(exe)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn engine");
    // The engine may exit before reading everything; a broken pipe is fine.
    let _ = child.stdin.take().expect("stdin piped").write_all(input);
    child.wait_with_output().expect("wait")
}

#[test]
#[cfg_attr(miri, ignore)]
fn malformed_go_argument_exits_with_critical_error() {
    let out = run_engine(b"position startpos\ngo btime x\nusi\nquit\n");
    assert_eq!(out.status.code(), Some(1), "status: {:?}", out.status);
    assert_eq!(
        String::from_utf8(out.stdout).expect("utf-8 stdout"),
        "info string CRITICAL ERROR: Command `go btime x` failed. \
         Reason: Invalid argument for 'btime'\n\n"
    );
}

#[test]
#[cfg_attr(miri, ignore)]
fn malformed_go_mate_budget_exits_with_critical_error() {
    let out = run_engine(b"go  mate 12abc \nusi\nquit\n");
    assert_eq!(out.status.code(), Some(1), "status: {:?}", out.status);
    assert_eq!(
        String::from_utf8(out.stdout).expect("utf-8 stdout"),
        "info string CRITICAL ERROR: Command `go  mate 12abc ` failed. \
         Reason: Invalid argument for 'mate'\n\n"
    );
}
