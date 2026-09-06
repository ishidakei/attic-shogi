//! Session tests for the wall-clock fields of the PV `info` line: `nps` and
//! `time`, which are derived rather than reported and so have to agree with the
//! `nodes` on the same line.
//!
//! Both run against a synthetic all-zero network staged in a temp dir; the book
//! test additionally stages a `.ybb`, whose hit short-circuits the search and
//! takes the separate depth-0 output path.

mod common;

use common::{TEST_BOOK_SEED, TempDir, drive_with_seed, stage_sample_ybb, write_synthetic_nn_bin};

const STARTPOS_B: &str = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1";

/// The token following `key` in a whitespace-tokenised `info` line.
fn field_after<'a>(line: &'a str, key: &str) -> Option<&'a str> {
    let mut toks = line.split_whitespace();
    toks.find(|t| *t == key)?;
    toks.next()
}

fn u64_field(line: &str, key: &str) -> u64 {
    field_after(line, key)
        .unwrap_or_else(|| panic!("`{key}` missing in info line: {line:?}"))
        .parse()
        .unwrap_or_else(|_| panic!("`{key}` is not an integer in info line: {line:?}"))
}

#[test]
#[cfg_attr(miri, ignore)]
fn searched_pv_line_reports_nps_consistent_with_its_nodes_and_time() {
    let dir = TempDir::new("pv-timing");
    write_synthetic_nn_bin(dir.path());
    let evaldir = dir.path().to_str().unwrap();
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

    let line = out
        .lines()
        .find(|l| l.starts_with("info depth 1 "))
        .unwrap_or_else(|| panic!("missing depth-1 info line in:\n{out}"));

    let nodes = u64_field(line, "nodes");
    let nps = u64_field(line, "nps");
    let time = u64_field(line, "time");
    assert!(time >= 1, "time is floored at 1, got {time} in: {line:?}");
    assert_eq!(
        nps,
        nodes * 1000 / time,
        "nps must be nodes*1000/time in: {line:?}"
    );
    assert!(
        line.contains(" pv "),
        "a depth-1 line carries a PV: {line:?}"
    );
}

#[test]
#[cfg_attr(miri, ignore)]
fn book_hit_line_reports_zero_nodes_and_nps_with_a_floored_time() {
    let dir = TempDir::new("pv-timing-book");
    write_synthetic_nn_bin(dir.path());
    stage_sample_ybb(dir.path(), "user_book1.ybb");
    let d = dir.path().to_str().unwrap();
    let session = format!(
        "usi\n\
         setoption name Threads value 1\n\
         setoption name EvalDir value {d}\n\
         setoption name BookDir value {d}\n\
         setoption name BookFile value user_book1.ybb\n\
         setoption name BookDepthLimit value 0\n\
         setoption name BookEvalBlackLimit value -99999\n\
         setoption name BookEvalDiff value 0\n\
         setoption name BookMoves value 10000\n\
         isready\n\
         position sfen {STARTPOS_B}\n\
         go depth 1\n\
         quit\n"
    );
    let out = drive_with_seed(&session, TEST_BOOK_SEED);

    let line = out
        .lines()
        .find(|l| l.starts_with("info depth 0 "))
        .unwrap_or_else(|| panic!("missing book depth-0 info line in:\n{out}"));

    assert_eq!(u64_field(line, "nodes"), 0, "book hit searches nothing");
    assert_eq!(u64_field(line, "nps"), 0, "no nodes means no nps");
    let time = u64_field(line, "time");
    assert!(time >= 1, "time is floored at 1, got {time} in: {line:?}");
}
