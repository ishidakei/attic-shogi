//! Driver-level session tests for the root opening-book integration.
//!
//! Each test drives a full `usi → setoption → isready → position → go` session
//! in-process against a synthetic (all-zero) network and a `.ybb` staged in a
//! temp dir, so they are hermetic. The all-zero network makes any *search*
//! deterministic; a book hit short-circuits the search entirely, so a reply that
//! equals the book's best move (and carries the depth-0 book `info` signature)
//! proves the book was consulted.

mod common;

use attic_state::{format_sfen, parse_usi_move};
use common::{
    StreamHarness, TEST_BOOK_SEED, TempDir, bestmove_lines, drive_with_seed, parse,
    stage_sample_ybb, write_synthetic_nn_bin, write_ybb,
};

const STARTPOS_B: &str = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1";
const STARTPOS_W: &str = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL w - 1";
const STARTPOS_B_PLY2: &str = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 2";
const CHECK_EVASION: &str = "4k4/9/4r4/9/9/9/4K3B/9/9 b RG2gs2n3p 1";
const DROP_HEAVY: &str = "k8/1P7/G8/1N2P4/9/9/9/9/8K b 2PG2pg 1";
const SENNICHITE: &str = "9/4k4/9/9/9/9/9/4K4/9 b 9P9p 1";

/// A session prefix that loads the synthetic net and points the book at `dir`,
/// staging `sample.ybb` as `user_book1.ybb` — one of the advertised `BookFile`
/// choices, all of which are `.ybb` names. `on_the_fly` toggles the
/// `BookOnTheFly` leg. The filter options are opened up so every fixture's
/// unique top move survives with `BookEvalDiff 0`.
fn book_session_prefix(dir: &str, on_the_fly: bool) -> String {
    format!(
        "usi\n\
         setoption name Threads value 1\n\
         setoption name EvalDir value {dir}\n\
         setoption name BookDir value {dir}\n\
         setoption name BookFile value user_book1.ybb\n\
         setoption name BookDepthLimit value 0\n\
         setoption name BookEvalBlackLimit value -99999\n\
         setoption name BookEvalWhiteLimit value -99999\n\
         setoption name BookEvalDiff value 0\n\
         setoption name BookMoves value 10000\n\
         setoption name BookOnTheFly value {on_the_fly}\n\
         isready\n"
    )
}

// -------------------------------------------------------------------------
// No-book behaviour is invisible.
// -------------------------------------------------------------------------

#[test]
fn default_no_book_emits_no_book_output_and_searches() {
    let dir = TempDir::new("nobook");
    write_synthetic_nn_bin(dir.path());
    let evaldir = dir.path().to_str().unwrap();
    // Pure defaults: BookFile=no_book. No book string of any kind, real search.
    let session = format!(
        "usi\n\
         setoption name Threads value 1\n\
         setoption name EvalDir value {evaldir}\n\
         isready\n\
         position startpos\n\
         go depth 1\n\
         quit\n"
    );
    let out = drive_with_seed(&session, TEST_BOOK_SEED);
    // `usi` lists the book *option declarations* (which contain "Book"); what
    // must be absent is any book *activity* — no `info` line mentions the book,
    // and no depth-0 book short-circuit occurs.
    assert!(
        !out.lines()
            .any(|l| l.starts_with("info") && l.to_lowercase().contains("book")),
        "no book info output expected with defaults, got:\n{out}"
    );
    assert!(
        !out.contains("info depth 0 "),
        "no book short-circuit with defaults, got:\n{out}"
    );
    assert!(
        out.lines().any(|l| l.starts_with("info depth 1 ")),
        "a real depth-1 search must run, got:\n{out}"
    );
    assert_eq!(bestmove_lines(&out).len(), 1, "one bestmove:\n{out}");
}

#[test]
fn absent_listed_book_falls_back_to_bookless_without_panic() {
    let dir = TempDir::new("absent");
    write_synthetic_nn_bin(dir.path());
    let d = dir.path().to_str().unwrap();
    // A listed name whose file is absent: an info-string notice, then a normal
    // (bookless) search — never a panic.
    let session = format!(
        "usi\n\
         setoption name Threads value 1\n\
         setoption name EvalDir value {d}\n\
         setoption name BookDir value {d}\n\
         setoption name BookFile value user_book1.ybb\n\
         isready\n\
         position startpos\n\
         go depth 1\n\
         quit\n"
    );
    let out = drive_with_seed(&session, TEST_BOOK_SEED);
    assert!(
        out.contains("info string") && out.to_lowercase().contains("book"),
        "expected a book load-failure notice, got:\n{out}"
    );
    assert!(
        out.lines().any(|l| l.starts_with("info depth 1 ")),
        "engine must still search bookless, got:\n{out}"
    );
    assert_eq!(bestmove_lines(&out).len(), 1);
}

// -------------------------------------------------------------------------
// Book-hit session (both read modes).
// -------------------------------------------------------------------------

fn run_book_hits(on_the_fly: bool) {
    let dir = TempDir::new(if on_the_fly { "hit-otf" } else { "hit-mem" });
    write_synthetic_nn_bin(dir.path());
    stage_sample_ybb(dir.path(), "user_book1.ybb");
    let d = dir.path().to_str().unwrap();

    let cases = [
        (STARTPOS_B, "7g7f"),
        (CHECK_EVASION, "1g5c+"),
        (DROP_HEAVY, "5d5c+"),
        (SENNICHITE, "P*5e"),
    ];
    let mut session = book_session_prefix(d, on_the_fly);
    for (sfen, _) in &cases {
        session.push_str(&format!("position sfen {sfen}\ngo depth 1\n"));
    }
    session.push_str("quit\n");
    let out = drive_with_seed(&session, TEST_BOOK_SEED);

    // An advertised choice names the file directly, so nothing is rewritten:
    // the sibling-fallback info string must NOT appear. (The fallback itself is
    // unchanged and unit-tested in `driver.rs`; it is simply unreachable from a
    // combo that offers only `.ybb` names.)
    assert!(
        !out.contains("book file fallback :"),
        "an advertised `.ybb` choice must load without a fallback rewrite:\n{out}"
    );
    assert!(
        out.contains("book loaded : "),
        "expected the book-loaded info string, got:\n{out}"
    );

    let bestmoves = bestmove_lines(&out);
    assert_eq!(bestmoves.len(), cases.len(), "one bestmove per go:\n{out}");
    for ((sfen, want), got) in cases.iter().zip(&bestmoves) {
        assert_eq!(
            got.split_whitespace().next().unwrap(),
            *want,
            "book move for {sfen} (on_the_fly={on_the_fly}):\n{out}"
        );
    }
    // A book hit reports a depth-0 line; a search never does. No depth-1 search
    // line appears for any of the four in-book positions.
    assert!(
        out.contains("info depth 0 "),
        "book hits must emit a depth-0 info line:\n{out}"
    );
    assert!(
        !out.lines().any(|l| l.starts_with("info depth 1 ")),
        "no normal search should run for in-book positions:\n{out}"
    );
}

#[test]
fn book_hits_in_memory_mode() {
    run_book_hits(false);
}

#[test]
fn book_hits_on_the_fly_mode() {
    run_book_hits(true);
}

// -------------------------------------------------------------------------
// Gating options.
// -------------------------------------------------------------------------

#[test]
fn own_book_off_runs_a_real_search() {
    let dir = TempDir::new("ownbook-off");
    write_synthetic_nn_bin(dir.path());
    stage_sample_ybb(dir.path(), "user_book1.ybb");
    let d = dir.path().to_str().unwrap();
    let mut session = book_session_prefix(d, false);
    // Turn the master gate off, then search an in-book position.
    session.push_str("setoption name USI_OwnBook value false\n");
    session.push_str(&format!("position sfen {STARTPOS_B}\ngo depth 1\nquit\n"));
    let out = drive_with_seed(&session, TEST_BOOK_SEED);
    assert!(
        out.lines().any(|l| l.starts_with("info depth 1 ")),
        "USI_OwnBook=false must run a real search:\n{out}"
    );
    assert!(
        !out.contains("info depth 0 "),
        "no book short-circuit when the book is gated off:\n{out}"
    );
}

#[test]
fn wrong_game_ply_misses_then_ignore_book_ply_hits() {
    let dir = TempDir::new("ignore-ply");
    write_synthetic_nn_bin(dir.path());
    stage_sample_ybb(dir.path(), "user_book1.ybb");
    let d = dir.path().to_str().unwrap();

    // Same board as the in-book startpos but at ply 2 → miss → search runs.
    let mut miss = book_session_prefix(d, false);
    miss.push_str(&format!(
        "position sfen {STARTPOS_B_PLY2}\ngo depth 1\nquit\n"
    ));
    let out = drive_with_seed(&miss, TEST_BOOK_SEED);
    assert!(
        out.lines().any(|l| l.starts_with("info depth 1 ")),
        "ply mismatch must miss and search:\n{out}"
    );

    // IgnoreBookPly is captured at load, so it needs a reload (isready) to take
    // effect — then the same ply-2 position hits the ply-1 entry.
    let mut hit = book_session_prefix(d, false);
    hit.push_str("setoption name IgnoreBookPly value true\n");
    hit.push_str("isready\n");
    hit.push_str(&format!(
        "position sfen {STARTPOS_B_PLY2}\ngo depth 1\nquit\n"
    ));
    let out = drive_with_seed(&hit, TEST_BOOK_SEED);
    let bm = bestmove_lines(&out);
    assert_eq!(
        bm.last().unwrap().split_whitespace().next().unwrap(),
        "7g7f",
        "IgnoreBookPly must let ply-2 hit the ply-1 entry:\n{out}"
    );
    assert!(out.contains("info depth 0 "), "book hit expected:\n{out}");
}

#[test]
fn game_ply_past_book_moves_misses() {
    let dir = TempDir::new("bookmoves");
    write_synthetic_nn_bin(dir.path());
    stage_sample_ybb(dir.path(), "user_book1.ybb");
    let d = dir.path().to_str().unwrap();
    let mut session = book_session_prefix(d, false);
    session.push_str("setoption name BookMoves value 0\n"); // ply 1 > 0 → miss.
    session.push_str(&format!("position sfen {STARTPOS_B}\ngo depth 1\nquit\n"));
    let out = drive_with_seed(&session, TEST_BOOK_SEED);
    assert!(
        out.lines().any(|l| l.starts_with("info depth 1 ")),
        "game_ply past BookMoves must miss and search:\n{out}"
    );
}

// -------------------------------------------------------------------------
// FlippedBook (session level; unit-level flip_move is in attic-state).
// -------------------------------------------------------------------------

#[test]
fn flipped_book_hits_the_rotated_startpos() {
    let dir = TempDir::new("flipped");
    write_synthetic_nn_bin(dir.path());
    stage_sample_ybb(dir.path(), "user_book1.ybb");
    let d = dir.path().to_str().unwrap();

    // Startpos is 180°-symmetric under color swap: the White-to-move startpos
    // packs (after flipping) to the in-book Black key, so 7g7f → 3c3d.
    let mut on = book_session_prefix(d, false);
    on.push_str("setoption name FlippedBook value true\n");
    on.push_str(&format!("position sfen {STARTPOS_W}\ngo depth 1\nquit\n"));
    let out = drive_with_seed(&on, TEST_BOOK_SEED);
    assert_eq!(
        bestmove_lines(&out)
            .last()
            .unwrap()
            .split_whitespace()
            .next()
            .unwrap(),
        "3c3d",
        "FlippedBook on must hit the rotated position:\n{out}"
    );

    // With FlippedBook off the rotated position misses and searches.
    let mut off = book_session_prefix(d, false);
    off.push_str("setoption name FlippedBook value false\n");
    off.push_str(&format!("position sfen {STARTPOS_W}\ngo depth 1\nquit\n"));
    let out = drive_with_seed(&off, TEST_BOOK_SEED);
    assert!(
        out.lines().any(|l| l.starts_with("info depth 1 ")),
        "FlippedBook off must miss the rotated position and search:\n{out}"
    );
}

// -------------------------------------------------------------------------
// Ponder fallback.
// -------------------------------------------------------------------------

#[test]
fn ponder_emitted_when_child_is_in_book_and_omitted_at_a_leaf() {
    let dir = TempDir::new("ponder");
    write_synthetic_nn_bin(dir.path());
    let d = dir.path().to_str().unwrap();

    // Post-7g7f position, for the chained entry.
    let after = {
        let mut p = parse(STARTPOS_B);
        p.do_move(parse_usi_move("7g7f", &p).unwrap());
        p
    };
    let after_sfen = format_sfen(&after);

    // A chained book: startpos → 7g7f, and its child → 3c3d.
    write_ybb(
        &dir.path().join("user_book1.ybb"),
        &[
            (STARTPOS_B, vec![("7g7f", 100, 20)]),
            (after_sfen.as_str(), vec![("3c3d", 80, 18)]),
        ],
    );
    let mut chained = book_session_prefix(d, false);
    chained.push_str(&format!("position sfen {STARTPOS_B}\ngo depth 1\nquit\n"));
    let out = drive_with_seed(&chained, TEST_BOOK_SEED);
    assert_eq!(
        bestmove_lines(&out).last().unwrap().trim(),
        "7g7f ponder 3c3d",
        "child in book → ponder from its first move:\n{out}"
    );

    // A leaf book (startpos only): no child entry → no ponder.
    write_ybb(
        &dir.path().join("user_book1.ybb"),
        &[(STARTPOS_B, vec![("7g7f", 100, 20)])],
    );
    let mut leaf = book_session_prefix(d, false);
    // A reload picks up the rewritten file (same name → force via BookOnTheFly
    // toggle so the signature changes and the book is re-read).
    leaf.push_str("setoption name BookOnTheFly value true\nisready\n");
    leaf.push_str(&format!("position sfen {STARTPOS_B}\ngo depth 1\nquit\n"));
    let out = drive_with_seed(&leaf, TEST_BOOK_SEED);
    let bm = bestmove_lines(&out);
    assert_eq!(
        bm.last().unwrap().trim(),
        "7g7f",
        "leaf position → bestmove with no ponder:\n{out}"
    );
}

// -------------------------------------------------------------------------
// Ponder / infinite discipline (no bestmove until stop / ponderhit).
// -------------------------------------------------------------------------

#[test]
fn go_infinite_book_hit_holds_bestmove_until_stop() {
    let dir = TempDir::new("hold-inf");
    write_synthetic_nn_bin(dir.path());
    stage_sample_ybb(dir.path(), "user_book1.ybb");
    let d = dir.path().to_str().unwrap();

    let h = StreamHarness::start_with_seed(Some(TEST_BOOK_SEED));
    for line in book_session_prefix(d, false).lines() {
        h.send(line);
    }
    assert!(h.wait_until(30000, |o| o.contains("readyok")), "readyok");
    h.send(&format!("position sfen {STARTPOS_B}"));
    h.send("go infinite");

    // The book probe emits its multipv info lines immediately, then holds.
    assert!(
        h.wait_until(2000, |o| o.contains("multipv 1")),
        "book info lines must appear:\n{}",
        h.output()
    );
    // No bestmove yet — the reply is held for stop.
    std::thread::sleep(std::time::Duration::from_millis(120));
    assert!(
        !h.output().contains("bestmove"),
        "bestmove must be withheld until stop:\n{}",
        h.output()
    );

    h.send("stop");
    assert!(
        h.wait_until(2000, |o| o.contains("bestmove")),
        "stop must release the reply:\n{}",
        h.output()
    );
    let out = h.quit_join();
    assert!(out.contains("info depth 0 "), "book depth-0 line:\n{out}");
    let bm = bestmove_lines(&out);
    assert_eq!(bm.len(), 1);
    assert_eq!(bm[0].split_whitespace().next().unwrap(), "7g7f");
}

#[test]
fn go_ponder_book_hit_holds_bestmove_until_ponderhit() {
    let dir = TempDir::new("hold-ponder");
    write_synthetic_nn_bin(dir.path());
    stage_sample_ybb(dir.path(), "user_book1.ybb");
    let d = dir.path().to_str().unwrap();

    let h = StreamHarness::start_with_seed(Some(TEST_BOOK_SEED));
    for line in book_session_prefix(d, false).lines() {
        h.send(line);
    }
    assert!(h.wait_until(30000, |o| o.contains("readyok")), "readyok");
    h.send(&format!("position sfen {STARTPOS_B}"));
    h.send("go ponder");

    assert!(
        h.wait_until(2000, |o| o.contains("multipv 1")),
        "book info lines must appear:\n{}",
        h.output()
    );
    std::thread::sleep(std::time::Duration::from_millis(120));
    assert!(
        !h.output().contains("bestmove"),
        "bestmove must be withheld until ponderhit:\n{}",
        h.output()
    );

    h.send("ponderhit");
    assert!(
        h.wait_until(2000, |o| o.contains("bestmove")),
        "ponderhit must release the reply:\n{}",
        h.output()
    );
    let out = h.quit_join();
    let bm = bestmove_lines(&out);
    assert_eq!(bm.len(), 1);
    assert_eq!(bm[0].split_whitespace().next().unwrap(), "7g7f");
}

// -------------------------------------------------------------------------
// Multiple Book — the numbered priority series.
// -------------------------------------------------------------------------

/// [`book_session_prefix`] with `FlippedBook` off: the startpos is 180°-symmetric
/// under a color swap, so with the flip enabled a Black-startpos entry in an
/// upper book would answer a White-startpos probe and mask which book was
/// actually consulted.
fn multi_book_prefix(dir: &str, on_the_fly: bool) -> String {
    let mut s = book_session_prefix(dir, on_the_fly);
    s.push_str("setoption name FlippedBook value false\n");
    s
}

/// How many books the session reported loading.
fn loaded_count(out: &str) -> usize {
    out.lines().filter(|l| l.contains("book loaded : ")).count()
}

fn run_priority_series_first_hit(on_the_fly: bool) {
    let dir = TempDir::new(if on_the_fly {
        "series-otf"
    } else {
        "series-mem"
    });
    write_synthetic_nn_bin(dir.path());
    let d = dir.path().to_str().unwrap();

    // Priority 0 answers the Black startpos; the base answers it differently and
    // is the only book holding the White startpos.
    write_ybb(
        &dir.path().join("user_book1-000.ybb"),
        &[(STARTPOS_B, vec![("7g7f", 100, 20)])],
    );
    write_ybb(
        &dir.path().join("user_book1.ybb"),
        &[
            (STARTPOS_B, vec![("2g2f", 100, 20)]),
            (STARTPOS_W, vec![("8c8d", 100, 20)]),
        ],
    );

    let mut session = multi_book_prefix(d, on_the_fly);
    session.push_str(&format!("position sfen {STARTPOS_B}\ngo depth 1\n"));
    session.push_str(&format!("position sfen {STARTPOS_W}\ngo depth 1\n"));
    session.push_str("quit\n");
    let out = drive_with_seed(&session, TEST_BOOK_SEED);

    assert_eq!(loaded_count(&out), 2, "both books must load:\n{out}");
    let bm = bestmove_lines(&out);
    assert_eq!(bm.len(), 2, "one bestmove per go:\n{out}");
    assert_eq!(
        bm[0].split_whitespace().next().unwrap(),
        "7g7f",
        "priority book 0 wins a position both books hold (on_the_fly={on_the_fly}):\n{out}"
    );
    assert_eq!(
        bm[1].split_whitespace().next().unwrap(),
        "8c8d",
        "a miss in book 0 falls through to the base book:\n{out}"
    );
    assert!(
        !out.lines().any(|l| l.starts_with("info depth 1 ")),
        "both positions are in book, so no search should run:\n{out}"
    );
}

#[test]
fn priority_series_first_hit_in_memory_mode() {
    run_priority_series_first_hit(false);
}

#[test]
fn priority_series_first_hit_on_the_fly_mode() {
    run_priority_series_first_hit(true);
}

#[test]
fn a_gap_ends_the_series() {
    let dir = TempDir::new("series-gap");
    write_synthetic_nn_bin(dir.path());
    let d = dir.path().to_str().unwrap();

    write_ybb(
        &dir.path().join("user_book1-000.ybb"),
        &[(STARTPOS_B, vec![("7g7f", 100, 20)])],
    );
    // `-001` is absent; `-003` exists but must never be reached.
    write_ybb(
        &dir.path().join("user_book1-003.ybb"),
        &[(STARTPOS_W, vec![("3c3d", 100, 20)])],
    );
    write_ybb(
        &dir.path().join("user_book1.ybb"),
        &[(STARTPOS_W, vec![("8c8d", 100, 20)])],
    );

    let mut session = multi_book_prefix(d, false);
    session.push_str(&format!("position sfen {STARTPOS_B}\ngo depth 1\n"));
    session.push_str(&format!("position sfen {STARTPOS_W}\ngo depth 1\n"));
    session.push_str("quit\n");
    let out = drive_with_seed(&session, TEST_BOOK_SEED);

    assert_eq!(
        loaded_count(&out),
        2,
        "only `-000` and the base load; `-003` is past the gap:\n{out}"
    );
    let bm = bestmove_lines(&out);
    assert_eq!(bm[0].split_whitespace().next().unwrap(), "7g7f");
    assert_eq!(
        bm[1].split_whitespace().next().unwrap(),
        "8c8d",
        "`-003` must not answer — it is past the gap:\n{out}"
    );
}

#[test]
fn duplicate_extension_prefers_the_primary_and_a_db_slot_fails_loudly() {
    let dir = TempDir::new("series-dup");
    write_synthetic_nn_bin(dir.path());
    let d = dir.path().to_str().unwrap();

    // `BookFile user_book1.ybb` makes `.ybb` the PRIMARY extension, so:
    //   `-000` present under both extensions -> the `.ybb` wins, with the pin's
    //          verbatim duplicate notice;
    //   `-001` present only as a `.db`       -> the secondary extension is picked,
    //          and this port cannot read it, so it reports loudly instead of
    //          skipping silently. The series still continues past it.
    let dup_slot = dir.path().join("user_book1-000.ybb");
    write_ybb(&dup_slot, &[(STARTPOS_B, vec![("7g7f", 100, 20)])]);
    std::fs::write(
        dir.path().join("user_book1-000.db"),
        b"#YANEURAOU-DB2016 1.00\n",
    )
    .expect("write .db slot");
    let db_slot = dir.path().join("user_book1-001.db");
    std::fs::write(&db_slot, b"#YANEURAOU-DB2016 1.00\n").expect("write .db slot");
    write_ybb(
        &dir.path().join("user_book1.ybb"),
        &[(STARTPOS_B, vec![("2g2f", 100, 20)])],
    );

    let mut session = multi_book_prefix(d, false);
    session.push_str(&format!("position sfen {STARTPOS_B}\ngo depth 1\nquit\n"));
    let out = drive_with_seed(&session, TEST_BOOK_SEED);

    assert!(
        out.contains(&format!(
            "info string priority book file exists twice. use : {}\n",
            dup_slot.display()
        )),
        "expected the pin's verbatim duplicate notice, got:\n{out}"
    );
    assert!(
        out.contains(&format!(
            "info string unsupported book format : {}\n",
            db_slot.display()
        )),
        "a `.db` priority slot must fail loudly, not be skipped silently:\n{out}"
    );
    assert_eq!(
        loaded_count(&out),
        2,
        "the `-000.ybb` and the base load; the `-001.db` is unusable:\n{out}"
    );
    assert_eq!(
        bestmove_lines(&out)[0].split_whitespace().next().unwrap(),
        "7g7f",
        "the `-000.ybb` wins the duplicate slot and answers first:\n{out}"
    );
}

#[test]
fn no_book_loads_nothing_even_with_numbered_files_present() {
    let dir = TempDir::new("series-nobook");
    write_synthetic_nn_bin(dir.path());
    let d = dir.path().to_str().unwrap();
    // A stray numbered file for the sentinel: the `no_book` stem is empty, so no
    // series is enumerated and nothing is loaded.
    write_ybb(
        &dir.path().join("no_book-000.ybb"),
        &[(STARTPOS_B, vec![("7g7f", 100, 20)])],
    );
    let session = format!(
        "usi\n\
         setoption name Threads value 1\n\
         setoption name EvalDir value {d}\n\
         setoption name BookDir value {d}\n\
         isready\n\
         position sfen {STARTPOS_B}\n\
         go depth 1\n\
         quit\n"
    );
    let out = drive_with_seed(&session, TEST_BOOK_SEED);
    assert_eq!(loaded_count(&out), 0, "no_book loads no books:\n{out}");
    assert!(
        !out.contains("info string book"),
        "no book chatter at all:\n{out}"
    );
    assert!(
        out.lines().any(|l| l.starts_with("info depth 1 ")),
        "a real search must run:\n{out}"
    );
}

#[test]
fn reread_is_skipped_until_the_capture_triple_changes() {
    let dir = TempDir::new("series-reread");
    write_synthetic_nn_bin(dir.path());
    let d = dir.path().to_str().unwrap();
    write_ybb(
        &dir.path().join("user_book1.ybb"),
        &[(STARTPOS_B, vec![("2g2f", 100, 20)])],
    );

    // An unchanged triple: the second `isready` re-reads nothing.
    let mut same = multi_book_prefix(d, false);
    same.push_str("isready\nquit\n");
    let out = drive_with_seed(&same, TEST_BOOK_SEED);
    assert_eq!(
        loaded_count(&out),
        1,
        "an unchanged (names, BookOnTheFly, IgnoreBookPly) triple must not reload:\n{out}"
    );

    // Flipping BookOnTheFly re-reads; flipping IgnoreBookPly re-reads again.
    let mut flips = multi_book_prefix(d, false);
    flips.push_str("setoption name BookOnTheFly value true\nisready\n");
    flips.push_str("setoption name IgnoreBookPly value true\nisready\n");
    flips.push_str("quit\n");
    let out = drive_with_seed(&flips, TEST_BOOK_SEED);
    assert_eq!(
        loaded_count(&out),
        3,
        "each capture flip forces a re-read:\n{out}"
    );

    // A numbered file appearing between two `isready`s changes the resolved name
    // list — also a re-read, and the new book takes priority.
    let h = StreamHarness::start_with_seed(Some(TEST_BOOK_SEED));
    for line in multi_book_prefix(d, false).lines() {
        h.send(line);
    }
    assert!(h.wait_until(30000, |o| o.contains("readyok")), "readyok");
    write_ybb(
        &dir.path().join("user_book1-000.ybb"),
        &[(STARTPOS_B, vec![("7g7f", 100, 20)])],
    );
    h.send("isready");
    assert!(
        h.wait_until(30000, |o| o.matches("readyok").count() == 2),
        "second readyok:\n{}",
        h.output()
    );
    h.send(&format!("position sfen {STARTPOS_B}"));
    h.send("go depth 1");
    assert!(
        h.wait_until(30000, |o| o.contains("bestmove")),
        "bestmove:\n{}",
        h.output()
    );
    let out = h.quit_join();
    assert_eq!(
        loaded_count(&out),
        3,
        "a new numbered file changes the name list → re-read (1 + 2):\n{out}"
    );
    assert_eq!(
        bestmove_lines(&out)
            .last()
            .unwrap()
            .split_whitespace()
            .next()
            .unwrap(),
        "7g7f",
        "the freshly appeared priority book answers:\n{out}"
    );
}
