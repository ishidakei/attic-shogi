//! Multi-cycle session tests for the driver's behaviour when no evaluation
//! network is loaded: the session must survive, `isready` must report the load
//! failure, and every `go` must reply `bestmove resign` with the notice.

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
fn multi_cycle_without_network_survives_and_resigns() {
    let session = "usi\n\
                   isready\n\
                   position startpos\n\
                   go depth 1\n\
                   position startpos moves 7g7f\n\
                   go depth 1\n\
                   quit\n";
    let out = drive(session);

    assert!(out.contains("usiok\n"), "missing usiok in:\n{out}");
    assert!(
        out.contains("info string eval load failed:"),
        "missing eval-load-failure notice in:\n{out}"
    );
    assert!(!out.contains("readyok"), "unexpected readyok in:\n{out}");

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
#[cfg_attr(miri, ignore)]
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
#[cfg_attr(miri, ignore)]
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
