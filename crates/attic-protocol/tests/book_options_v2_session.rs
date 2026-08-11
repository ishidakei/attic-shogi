//! Driver-level session tests for the `BOOK_OPTIONS=V2` engine-option profile.
//!
//! The profile file (`engine_option_profile.txt`) is written into an isolated
//! temp directory and injected via `UsiDriver::with_option_profile`, so no test
//! ever depends on the process working directory containing (or not containing)
//! a stray profile file.
//!
//! Gates: the V2 handshake surface (`book.cpp`), and the four
//! probe-side divergences (`book.cpp`)
//! — of which the two side-to-move-dependent option NAMES are
//! observable from a session, via both the filtering they perform and the
//! `info string` that names the option used.

mod common;

use common::{
    TEST_BOOK_SEED, TempDir, bestmove_lines, drive, drive_with_profile, write_option_profile,
    write_synthetic_nn_bin, write_ybb,
};

const STARTPOS_B: &str = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1";
const STARTPOS_W: &str = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL w - 1";

/// Stage a two-record book (one Black root, one White root) with identical
/// value / depth spreads, so the only thing that can distinguish the Black from
/// the White filter options is the root side to move.
fn stage_two_sided_book(dir: &std::path::Path) {
    write_ybb(
        &dir.join("user_book1.ybb"),
        &[
            (
                STARTPOS_B,
                vec![("7g7f", 42, 20), ("2g2f", -13, 18), ("6i7h", 5, 16)],
            ),
            (
                STARTPOS_W,
                vec![("3c3d", 42, 20), ("8c8d", -13, 18), ("4a3b", 5, 16)],
            ),
        ],
    );
}

/// A V2 session prefix: synthetic net + the two-sided book. No book-filter
/// option is set, so every filter runs at its V2 default.
fn v2_prefix(dir: &str) -> String {
    format!(
        "usi\n\
         setoption name Threads value 1\n\
         setoption name EvalDir value {dir}\n\
         setoption name BookDir value {dir}\n\
         setoption name BookFile value user_book1.ybb\n\
         isready\n"
    )
}

// -------------------------------------------------------------------------
// The handshake surface.
// -------------------------------------------------------------------------

#[test]
fn v2_profile_handshake_surface() {
    let dir = TempDir::new("v2-handshake");
    let profile = write_option_profile(dir.path(), "# large-book profile\nBOOK_OPTIONS = V2\n");
    let out = drive_with_profile("usi\nquit\n", TEST_BOOK_SEED, &profile);

    // The split filters, with the pin's defaults and ranges.
    for line in [
        "option name BookEvalBlackDiff type spin default 0 min 0 max 99999",
        "option name BookEvalWhiteDiff type spin default 0 min 0 max 99999",
        "option name BookDepthBlackLimit type spin default 0 min 0 max 99999",
        "option name BookDepthWhiteLimit type spin default 5 min 0 max 99999",
    ] {
        assert!(
            out.lines().any(|l| l == line),
            "missing V2 option line `{line}`:\n{out}"
        );
    }

    // The options V2 retires.
    for name in [
        "NarrowBook",
        "ConsiderBookMoveCount",
        "BookEvalDiff",
        "BookDepthLimit",
    ] {
        assert!(
            !out.lines()
                .any(|l| l.starts_with(&format!("option name {name} "))),
            "{name} must be absent under V2:\n{out}"
        );
    }

    // Shifted defaults.
    for line in [
        "option name BookMoves type spin default 200 min 0 max 10000",
        "option name BookOnTheFly type check default true",
        "option name IgnoreBookPly type check default true",
    ] {
        assert!(
            out.lines().any(|l| l == line),
            "missing V2 default line `{line}`:\n{out}"
        );
    }

    // `BookFile` keeps the port's deliberate `no_book` default in BOTH profiles
    // (the pin would default to `user_book1.db` here) — books stay opt-in.
    assert!(
        out.lines().any(|l| l.starts_with(
            "option name BookFile type combo default no_book var no_book var standard_book.ybb "
        )),
        "BookFile must still default to no_book under V2:\n{out}"
    );

    // Every non-book option is profile-independent: dropping the book group from
    // both transcripts leaves them identical.
    let strip_book = |s: &str| -> Vec<String> {
        s.lines()
            .filter(|l| !l.contains("Book") && !l.contains("book"))
            .map(str::to_string)
            .collect()
    };
    let v1 = drive("usi\nquit\n");
    assert_eq!(
        strip_book(&out),
        strip_book(&v1),
        "only the book group may differ between profiles"
    );
}

#[test]
fn absent_profile_file_keeps_the_v1_surface() {
    let dir = TempDir::new("v2-absent");
    // Deliberately do NOT write the profile file.
    let missing = dir.path().join("engine_option_profile.txt");
    let out = drive_with_profile("usi\nquit\n", TEST_BOOK_SEED, &missing);
    assert_eq!(
        out,
        drive("usi\nquit\n"),
        "a missing profile file must reproduce the default (V1) handshake byte-for-byte"
    );
}

#[test]
fn a_v1_profile_file_keeps_the_v1_surface() {
    let dir = TempDir::new("v2-explicit-v1");
    let profile = write_option_profile(dir.path(), "BOOK_OPTIONS = V1\n");
    let out = drive_with_profile("usi\nquit\n", TEST_BOOK_SEED, &profile);
    assert_eq!(out, drive("usi\nquit\n"));
}

// -------------------------------------------------------------------------
// The eval-diff option name follows the root side to move.
// -------------------------------------------------------------------------

#[test]
fn v2_black_root_filters_with_the_black_eval_diff() {
    let dir = TempDir::new("v2-eval-black");
    write_synthetic_nn_bin(dir.path());
    stage_two_sided_book(dir.path());
    let profile = write_option_profile(dir.path(), "BOOK_OPTIONS_V2\n");
    let d = dir.path().to_str().unwrap();

    // Defaults: BookEvalBlackDiff 0 → only the top-valued move survives.
    let session = format!(
        "{}position sfen {STARTPOS_B}\ngo depth 1\nquit\n",
        v2_prefix(d)
    );
    let out = drive_with_profile(&session, TEST_BOOK_SEED, &profile);
    assert_eq!(
        bestmove_lines(&out)[0].split_whitespace().next().unwrap(),
        "7g7f",
        "BookEvalBlackDiff 0 must isolate the best move:\n{out}"
    );
    assert!(
        out.contains(
            "info string BookEvalBlackDiff = 0 , BookEvalBlackLimit = 0 , 3 moves to 1 moves."
        ),
        "the filter notice must name the Black option:\n{out}"
    );

    // Widening the WHITE gap changes nothing at a Black root.
    let session = format!(
        "{}setoption name BookEvalWhiteDiff value 99999\n\
         position sfen {STARTPOS_B}\ngo depth 1\nquit\n",
        v2_prefix(d)
    );
    let out = drive_with_profile(&session, TEST_BOOK_SEED, &profile);
    assert_eq!(
        bestmove_lines(&out)[0].split_whitespace().next().unwrap(),
        "7g7f",
        "BookEvalWhiteDiff is inert at a Black root:\n{out}"
    );

    // Widening the BLACK gap does: the eval floor (BookEvalBlackLimit 0) now
    // decides, so 6i7h (value 5) survives alongside 7g7f and -13 is dropped.
    let session = format!(
        "{}setoption name BookEvalBlackDiff value 99999\n\
         position sfen {STARTPOS_B}\ngo depth 1\nquit\n",
        v2_prefix(d)
    );
    let out = drive_with_profile(&session, TEST_BOOK_SEED, &profile);
    assert!(
        out.contains(
            "info string BookEvalBlackDiff = 99999 , BookEvalBlackLimit = 0 , 3 moves to 2 moves."
        ),
        "a wide Black gap must leave the eval floor in charge:\n{out}"
    );
    let best = bestmove_lines(&out)[0].split_whitespace().next().unwrap();
    assert!(
        ["7g7f", "6i7h"].contains(&best),
        "unexpected book move {best}:\n{out}"
    );
}

#[test]
fn v2_white_root_filters_with_the_white_eval_diff() {
    let dir = TempDir::new("v2-eval-white");
    write_synthetic_nn_bin(dir.path());
    stage_two_sided_book(dir.path());
    let profile = write_option_profile(dir.path(), "BOOK_OPTIONS_V2\n");
    let d = dir.path().to_str().unwrap();

    // Defaults: BookEvalWhiteDiff 0 → only the top-valued move survives, and the
    // notice names the White pair (the floor is BookEvalWhiteLimit = -140).
    let session = format!(
        "{}position sfen {STARTPOS_W}\ngo depth 1\nquit\n",
        v2_prefix(d)
    );
    let out = drive_with_profile(&session, TEST_BOOK_SEED, &profile);
    assert_eq!(
        bestmove_lines(&out)[0].split_whitespace().next().unwrap(),
        "3c3d",
        "BookEvalWhiteDiff 0 must isolate the best move:\n{out}"
    );
    assert!(
        out.contains(
            "info string BookEvalWhiteDiff = 0 , BookEvalWhiteLimit = -140 , 3 moves to 1 moves."
        ),
        "the filter notice must name the White option:\n{out}"
    );

    // The BLACK gap is inert at a White root; the WHITE one is not.
    let session = format!(
        "{}setoption name BookEvalBlackDiff value 99999\n\
         position sfen {STARTPOS_W}\ngo depth 1\nquit\n",
        v2_prefix(d)
    );
    let out = drive_with_profile(&session, TEST_BOOK_SEED, &profile);
    assert_eq!(
        bestmove_lines(&out)[0].split_whitespace().next().unwrap(),
        "3c3d",
        "BookEvalBlackDiff is inert at a White root:\n{out}"
    );

    let session = format!(
        "{}setoption name BookEvalWhiteDiff value 99999\n\
         position sfen {STARTPOS_W}\ngo depth 1\nquit\n",
        v2_prefix(d)
    );
    let out = drive_with_profile(&session, TEST_BOOK_SEED, &profile);
    // Floor -140 keeps all three moves, so nothing is filtered and no notice is
    // emitted; any of the three may be selected.
    let best = bestmove_lines(&out)[0].split_whitespace().next().unwrap();
    assert!(
        ["3c3d", "8c8d", "4a3b"].contains(&best),
        "unexpected book move {best}:\n{out}"
    );
    assert!(
        !out.contains("BookEvalWhiteDiff = 99999"),
        "nothing was filtered, so no notice is due:\n{out}"
    );
}

// -------------------------------------------------------------------------
// The depth-floor option name follows the root side to move.
// -------------------------------------------------------------------------

#[test]
fn v2_depth_floor_follows_the_root_side_to_move() {
    let dir = TempDir::new("v2-depth");
    write_synthetic_nn_bin(dir.path());
    stage_two_sided_book(dir.path());
    let profile = write_option_profile(dir.path(), "BOOK_OPTIONS_V2\n");
    let d = dir.path().to_str().unwrap();

    // Black root, Black floor above the best move's depth (20) → the whole entry
    // is skipped and a real search runs.
    let session = format!(
        "{}setoption name BookDepthBlackLimit value 25\n\
         position sfen {STARTPOS_B}\ngo depth 1\nquit\n",
        v2_prefix(d)
    );
    let out = drive_with_profile(&session, TEST_BOOK_SEED, &profile);
    assert!(
        out.contains("info string BookDepthBlackLimit is lower than the depth of this node."),
        "the skip notice must name the Black option:\n{out}"
    );
    assert!(
        out.lines().any(|l| l.starts_with("info depth 1 ")),
        "a skipped book entry must fall through to a real search:\n{out}"
    );

    // The White floor is inert at a Black root: the book still hits.
    let session = format!(
        "{}setoption name BookDepthWhiteLimit value 25\n\
         position sfen {STARTPOS_B}\ngo depth 1\nquit\n",
        v2_prefix(d)
    );
    let out = drive_with_profile(&session, TEST_BOOK_SEED, &profile);
    assert_eq!(
        bestmove_lines(&out)[0].split_whitespace().next().unwrap(),
        "7g7f",
        "BookDepthWhiteLimit is inert at a Black root:\n{out}"
    );

    // …and the mirror image at a White root.
    let session = format!(
        "{}setoption name BookDepthWhiteLimit value 25\n\
         position sfen {STARTPOS_W}\ngo depth 1\nquit\n",
        v2_prefix(d)
    );
    let out = drive_with_profile(&session, TEST_BOOK_SEED, &profile);
    assert!(
        out.contains("info string BookDepthWhiteLimit is lower than the depth of this node."),
        "the skip notice must name the White option:\n{out}"
    );

    let session = format!(
        "{}setoption name BookDepthBlackLimit value 25\n\
         position sfen {STARTPOS_W}\ngo depth 1\nquit\n",
        v2_prefix(d)
    );
    let out = drive_with_profile(&session, TEST_BOOK_SEED, &profile);
    assert_eq!(
        bestmove_lines(&out)[0].split_whitespace().next().unwrap(),
        "3c3d",
        "BookDepthBlackLimit is inert at a White root:\n{out}"
    );
}

// -------------------------------------------------------------------------
// V1 behaviour is untouched when no profile file is present.
// -------------------------------------------------------------------------

#[test]
fn v1_still_uses_the_single_eval_diff_and_depth_limit() {
    let dir = TempDir::new("v1-unchanged");
    write_synthetic_nn_bin(dir.path());
    stage_two_sided_book(dir.path());
    let d = dir.path().to_str().unwrap();
    let session = format!(
        "usi\n\
         setoption name Threads value 1\n\
         setoption name EvalDir value {d}\n\
         setoption name BookDir value {d}\n\
         setoption name BookFile value user_book1.ybb\n\
         setoption name BookDepthLimit value 0\n\
         setoption name BookEvalDiff value 0\n\
         isready\n\
         position sfen {STARTPOS_B}\ngo depth 1\nquit\n"
    );
    let out = drive(&session);
    assert_eq!(
        bestmove_lines(&out)[0].split_whitespace().next().unwrap(),
        "7g7f"
    );
    assert!(
        out.contains("info string BookEvalDiff = 0 , BookEvalBlackLimit = 0 , 3 moves to 1 moves."),
        "V1 must still report the single BookEvalDiff option:\n{out}"
    );
}
