use std::sync::{Arc, Mutex};

use attic_protocol::UsiDriver;

fn drive(input: &str) -> String {
    let output = Arc::new(Mutex::new(Vec::<u8>::new()));
    let driver = UsiDriver::new(input.as_bytes(), Arc::clone(&output));
    driver.run().expect("driver run");
    let bytes = output.lock().expect("output lock").clone();
    String::from_utf8(bytes).expect("utf-8")
}

#[test]
fn full_usi_to_usiok_golden() {
    let out = drive("usi\nquit\n");
    let expected = "\
id name Attic 9.70git\n\
id author Kei Ishida <ishida.kei@gmail.com>\n\
option name USI_Hash type spin default 1024 min 1 max 33554432\n\
option name Threads type spin default 4 min 1 max 1024\n\
option name MultiPV type spin default 1 min 1 max 600\n\
option name EvalDir type string default eval\n\
option name FV_SCALE type spin default 16 min 1 max 128\n\
option name USI_OwnBook type check default true\n\
option name NarrowBook type check default false\n\
option name BookMoves type spin default 16 min 0 max 10000\n\
option name BookIgnoreRate type spin default 0 min 0 max 100\n\
option name BookFile type combo default no_book var no_book var standard_book.ybb var yaneura_book1.ybb var yaneura_book2.ybb var yaneura_book3.ybb var yaneura_book4.ybb var user_book1.ybb var user_book2.ybb var user_book3.ybb var book.ybb\n\
option name BookDir type string default book\n\
option name BookEvalDiff type spin default 30 min 0 max 99999\n\
option name BookEvalBlackLimit type spin default 0 min -99999 max 99999\n\
option name BookEvalWhiteLimit type spin default -140 min -99999 max 99999\n\
option name BookDepthLimit type spin default 16 min 0 max 99999\n\
option name BookOnTheFly type check default false\n\
option name ConsiderBookMoveCount type check default false\n\
option name BookPvMoves type spin default 8 min 1 max 246\n\
option name IgnoreBookPly type check default false\n\
option name FlippedBook type check default true\n\
option name EnteringKingRule type combo default CSARule27 var NoEnteringKing var CSARule24 var CSARule24H var CSARule27 var CSARule27H var TryRule\n\
option name DepthLimit type spin default 0 min 0 max 2147483647\n\
option name NodesLimit type spin default 0 min 0 max 9223372036854775807\n\
option name MaxMovesToDraw type spin default 0 min 0 max 100000\n\
option name PvInterval type spin default 300 min 0 max 100000000\n\
option name ConsiderationMode type check default false\n\
option name OutputFailLHPV type check default true\n\
option name DrawValueBlack type spin default -2 min -30000 max 30000\n\
option name DrawValueWhite type spin default -2 min -30000 max 30000\n\
option name ResignValue type spin default 99999 min 0 max 99999\n\
option name GenerateAllLegalMoves type check default false\n\
option name NetworkDelay type spin default 120 min 0 max 10000\n\
option name NetworkDelay2 type spin default 1120 min 0 max 10000\n\
option name MinimumThinkingTime type spin default 2000 min 1 max 100000\n\
option name SlowMover type spin default 100 min 1 max 1000\n\
option name RoundUpToFullSecond type check default true\n\
option name NumaPolicy type string default auto\n\
option name USI_Ponder type check default false\n\
option name Stochastic_Ponder type check default false\n\
usiok\n";
    assert_eq!(out, expected);
}

#[test]
fn isready_without_network_reports_load_failure() {
    // Default EvalDir (`eval`) has no `nn.bin` in the test CWD, so the load
    // fails: an `info string eval load failed:` notice and no `readyok`, per
    // the isready contract — a failed network load must never be answered with
    // `readyok`. The positive path is covered by
    // tests/eval_session.rs (synthetic network) and tests/real_network_selfplay.
    let out = drive("isready\nquit\n");
    assert!(
        out.contains("info string eval load failed:"),
        "expected eval-load-failure notice, got: {out:?}"
    );
    assert!(
        !out.contains("readyok"),
        "readyok must not appear on a failed load: {out:?}"
    );
}

#[test]
fn unknown_command_emits_info_string() {
    let out = drive("frobnicate\nquit\n");
    assert_eq!(out, "info string unknown command: frobnicate\n");
}

#[test]
fn oversized_line_emits_command_too_long() {
    // 64 KB + 1 byte → TooLong. Followed by a real line so the driver loops past it.
    let mut input = "x".repeat(64 * 1024 + 1);
    input.push('\n');
    input.push_str("quit\n");
    let out = drive(&input);
    assert_eq!(out, "info string command too long\n");
}

#[test]
fn full_handshake_then_setoption_then_quit() {
    let out = drive("usi\nsetoption name USI_Hash value 256\nisready\nquit\n");
    assert!(out.starts_with("id name Attic 9.70git\n"));
    assert!(out.contains("usiok\n"));
    assert!(!out.contains("rejected"));
    // No EvalDir set → default `eval/nn.bin` is absent → load fails, no readyok.
    assert!(out.contains("info string eval load failed:"));
    assert!(!out.contains("readyok"));
}

#[test]
fn usinewgame_emits_nothing() {
    let out = drive("usinewgame\nquit\n");
    assert_eq!(out, "");
}
