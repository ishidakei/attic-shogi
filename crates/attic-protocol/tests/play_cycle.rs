//! Multi-cycle session tests over `UsiDriver` with an in-memory byte sequence.
//!
//! These cover the driver's behaviour when **no** evaluation network is loaded
//! (the default `EvalDir` has no `nn.bin` in the test CWD): the session must
//! survive, `isready` must report the load failure, and every `go` must reply
//! `bestmove resign` with the no-network notice rather than crashing. The
//! positive play path — search-chosen legal moves — is covered by
//! `tests/eval_session.rs` (synthetic network) and `tests/real_network_selfplay`
//! (the real `nn.bin`).

use std::sync::{Arc, Mutex};

use attic_protocol::UsiDriver;

fn drive(input: &str) -> String {
    let output = Arc::new(Mutex::new(Vec::<u8>::new()));
    let driver = UsiDriver::new(input.as_bytes(), Arc::clone(&output));
    driver.run().expect("driver run");
    let bytes = output.lock().expect("output lock").clone();
    String::from_utf8(bytes).expect("utf-8")
}

fn is_well_formed_usi_move(s: &str) -> bool {
    if s == "resign" {
        return true;
    }
    let b = s.as_bytes();
    // Drop: `[PLNSGBR]*<file><rank>` — 4 bytes.
    if b.len() == 4 && b[1] == b'*' {
        return matches!(b[0], b'P' | b'L' | b'N' | b'S' | b'G' | b'B' | b'R')
            && (b'1'..=b'9').contains(&b[2])
            && (b'a'..=b'i').contains(&b[3]);
    }
    // Board move: `<file><rank><file><rank>[+]` — 4 or 5 bytes.
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
fn multi_cycle_without_network_survives_and_resigns() {
    let session = "usi\n\
                   isready\n\
                   position startpos\n\
                   go depth 1\n\
                   position startpos moves 7g7f\n\
                   go depth 1\n\
                   quit\n";
    let out = drive(session);

    // Handshake present; the failed load reports itself and withholds readyok.
    assert!(out.contains("usiok\n"), "missing usiok in:\n{out}");
    assert!(
        out.contains("info string eval load failed:"),
        "missing eval-load-failure notice in:\n{out}"
    );
    assert!(!out.contains("readyok"), "unexpected readyok in:\n{out}");

    // Two `go` cycles, each replies `bestmove resign` with the no-network notice.
    assert_eq!(
        out.matches("info string no eval network loaded; run isready")
            .count(),
        2,
        "expected the no-network notice before each go in:\n{out}"
    );
    let bestmoves: Vec<&str> = out
        .lines()
        .filter_map(|l| l.strip_prefix("bestmove "))
        .collect();
    assert_eq!(bestmoves, vec!["resign", "resign"], "in:\n{out}");
    for m in &bestmoves {
        assert!(is_well_formed_usi_move(m), "malformed bestmove: {m:?}");
    }
}

#[test]
fn position_then_go_after_sfen_without_network_resigns() {
    let sfen = attic_state::STARTPOS_SFEN;
    let session = format!(
        "usi\n\
         isready\n\
         position sfen {sfen}\n\
         go infinite\n\
         stop\n\
         quit\n"
    );
    let out = drive(&session);

    let bestmoves: Vec<&str> = out
        .lines()
        .filter_map(|l| l.strip_prefix("bestmove "))
        .collect();
    assert_eq!(bestmoves, vec!["resign"], "in:\n{out}");
}

#[test]
fn usinewgame_between_cycles_without_network_resigns() {
    let session = "usi\n\
                   isready\n\
                   position startpos moves 7g7f\n\
                   go\n\
                   usinewgame\n\
                   go\n\
                   quit\n";
    let out = drive(session);

    let bestmoves: Vec<&str> = out
        .lines()
        .filter_map(|l| l.strip_prefix("bestmove "))
        .collect();
    assert_eq!(bestmoves, vec!["resign", "resign"], "in:\n{out}");
}
