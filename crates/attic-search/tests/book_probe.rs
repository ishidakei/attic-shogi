//! Parity gate for the `.ybb` opening-book reader (`attic-storage::book`).
//!
//! Ground truth is a hand-authored `tests/fixtures/book/book.db`, converted to
//! `sample.ybb` by `cargo xtask capture-book` (which packs each position with
//! the workspace PackedSfen encoder — the same encoder pinned bit-for-bit
//! against the reference's cshogi vectors in `attic-state`). The expected move
//! lists live in `expected.json`, derived from the same `.db`.
//!
//! This test recomputes each query key from the SFEN via `sfen_pack`, so it
//! exercises the full encoder → binary-search → decode path in both read modes,
//! and confirms the two modes agree.

use std::path::PathBuf;

use attic_state::{parse_sfen, sfen_pack};
use attic_storage::{Book, BookMove};
use serde::Deserialize;

fn fixture(rel: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join(rel)
}

#[derive(Deserialize)]
struct Expected {
    with_depth: bool,
    positions: Vec<ExpectedPos>,
}

#[derive(Deserialize)]
struct ExpectedPos {
    sfen: String,
    ply: u16,
    moves: Vec<ExpectedMove>,
}

#[derive(Deserialize)]
struct ExpectedMove {
    #[allow(dead_code)]
    usi: String,
    move16: u16,
    value: i16,
    depth: u16,
    count: u16,
}

fn load_expected() -> Expected {
    let text = std::fs::read_to_string(fixture("tests/fixtures/book/expected.json"))
        .expect("read expected.json");
    serde_json::from_str(&text).expect("parse expected.json")
}

fn packed_of(sfen: &str) -> [u8; 32] {
    sfen_pack(&parse_sfen(sfen).expect("valid sfen"))
}

fn to_book_moves(moves: &[ExpectedMove]) -> Vec<BookMove> {
    moves
        .iter()
        .map(|m| BookMove {
            move16: m.move16,
            value: m.value,
            depth: m.depth,
            count: m.count,
        })
        .collect()
}

const STARTPOS: &str = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1";

#[test]
#[cfg_attr(miri, ignore)]
fn probe_matches_expected_in_both_modes() {
    let path = fixture("tests/fixtures/book/sample.ybb");
    let memory = Book::open_in_memory(&path).expect("open in-memory");
    let on_the_fly = Book::open_on_the_fly(&path).expect("open on-the-fly");

    let expected = load_expected();
    assert_eq!(memory.has_move_depth(), expected.with_depth);
    assert_eq!(on_the_fly.has_move_depth(), expected.with_depth);
    assert_eq!(memory.record_count(), expected.positions.len() as u64);

    for pos in &expected.positions {
        let packed = packed_of(&pos.sfen);
        let want = to_book_moves(&pos.moves);

        let got_memory = memory
            .probe(&packed, pos.ply, false)
            .expect("no i/o error")
            .unwrap_or_else(|| panic!("in-memory miss for {}", pos.sfen));
        let got_otf = on_the_fly
            .probe(&packed, pos.ply, false)
            .expect("no i/o error")
            .unwrap_or_else(|| panic!("on-the-fly miss for {}", pos.sfen));

        assert_eq!(got_memory, want, "in-memory moves for {}", pos.sfen);
        assert_eq!(got_otf, want, "on-the-fly moves for {}", pos.sfen);
        // The two modes must be byte-for-byte identical.
        assert_eq!(got_memory, got_otf, "modes disagree for {}", pos.sfen);
    }
}

#[test]
#[cfg_attr(miri, ignore)]
fn probe_misses_in_both_modes() {
    let path = fixture("tests/fixtures/book/sample.ybb");
    let memory = Book::open_in_memory(&path).expect("open in-memory");
    let on_the_fly = Book::open_on_the_fly(&path).expect("open on-the-fly");

    // Near-miss: same board as the in-book startpos, but the wrong game ply.
    // PackedSfen carries no ply, so the key matches; the index ply is an
    // exact-equality post-filter, so this misses when ply is enforced and hits
    // when it is ignored.
    let packed_start = packed_of(STARTPOS);
    assert!(memory.probe(&packed_start, 2, false).unwrap().is_none());
    assert!(on_the_fly.probe(&packed_start, 2, false).unwrap().is_none());
    assert!(memory.probe(&packed_start, 2, true).unwrap().is_some());
    assert!(on_the_fly.probe(&packed_start, 2, true).unwrap().is_some());

    // Different side to move → different packed key → miss even ignoring ply.
    let white = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL w - 1";
    let packed_white = packed_of(white);
    assert!(memory.probe(&packed_white, 1, false).unwrap().is_none());
    assert!(on_the_fly.probe(&packed_white, 1, false).unwrap().is_none());
    assert!(memory.probe(&packed_white, 1, true).unwrap().is_none());

    // A legal but simply off-book position (startpos after 7g7f).
    let off_book = "lnsgkgsnl/1r5b1/ppppppppp/9/9/2P6/PP1PPPPPP/1B5R1/LNSGKGSNL w - 2";
    let packed_off = packed_of(off_book);
    assert!(memory.probe(&packed_off, 2, false).unwrap().is_none());
    assert!(on_the_fly.probe(&packed_off, 2, false).unwrap().is_none());
}

#[test]
#[cfg_attr(miri, ignore)]
fn truncated_or_corrupt_files_never_panic() {
    let path = fixture("tests/fixtures/book/sample.ybb");
    let full = std::fs::read(&path).expect("read sample.ybb");
    let expected = load_expected();

    // Query keys for every in-book position, so a surviving region still gets
    // exercised by the binary search.
    let keys: Vec<([u8; 32], u16)> = expected
        .positions
        .iter()
        .map(|p| (packed_of(&p.sfen), p.ply))
        .collect();

    // Region boundaries: header end (32) and index end (32 + 4*44 = 208), plus a
    // spread of other lengths and deterministic "random" truncations.
    let mut lengths: Vec<usize> = vec![0, 1, 16, 31, 32, 33, 48, 100, 175, 207, 208, 209, 254, 255];
    for cut in [1usize, 3, 5, 7, 11, 13, 29, 47, 101, 199] {
        if cut < full.len() {
            lengths.push(full.len() - cut);
        }
    }
    lengths.retain(|&n| n <= full.len());
    lengths.sort_unstable();
    lengths.dedup();

    let tmp_dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR"));

    let probe_all = |book: &Book| {
        for (packed, ply) in &keys {
            // The contract: never panic. Any Ok/Err with any Some/None is fine.
            let _ = book.probe(packed, *ply, false);
            let _ = book.probe(packed, *ply, true);
        }
    };

    for (i, &len) in lengths.iter().enumerate() {
        let buf = full[..len].to_vec();

        // In-memory mode: open may reject (header/index truncated) or accept.
        if let Ok(book) = Book::from_memory(buf.clone()) {
            probe_all(&book);
        }

        // On-the-fly mode: same, via a real file.
        let file = tmp_dir.join(format!("trunc_{i}.ybb"));
        std::fs::write(&file, &buf).expect("write truncated fixture");
        if let Ok(book) = Book::open_on_the_fly(&file) {
            probe_all(&book);
        }
    }

    // Single-byte corruptions across the header, index, and moves regions.
    for (i, &pos) in [0usize, 8, 16, 20, 24, 40, 60, 100, 208, 230, 255]
        .iter()
        .enumerate()
    {
        if pos >= full.len() {
            continue;
        }
        let mut buf = full.clone();
        buf[pos] ^= 0xFF;

        if let Ok(book) = Book::from_memory(buf.clone()) {
            probe_all(&book);
        }
        let file = tmp_dir.join(format!("corrupt_{i}.ybb"));
        std::fs::write(&file, &buf).expect("write corrupt fixture");
        if let Ok(book) = Book::open_on_the_fly(&file) {
            probe_all(&book);
        }
    }
}
