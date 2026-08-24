//! Every `BookFile` choice the handshake advertises must be selectable AND
//! loadable.
//!
//! The pin's choice list is `no_book` plus eight `.db` names plus `book.bin`;
//! this engine reads `.ybb` books only, so it deliberately diverges and offers
//! the `.ybb` spelling of the same stems. This test is the guard on that
//! divergence: it reads the combo line out of the real `usi` handshake (not a
//! hard-coded copy of the list), then drives one full session per choice with
//! the committed `sample.ybb` staged under `BookDir` as exactly that name, and
//! requires a book load plus a book hit through the unchanged name-resolution
//! and priority-enumeration machinery.

mod common;

use common::{
    TEST_BOOK_SEED, TempDir, bestmove_lines, drive, drive_with_seed, stage_sample_ybb,
    write_synthetic_nn_bin,
};

const STARTPOS_B: &str = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1";

/// The `BookFile` choices as advertised by the `usi` handshake, in order.
fn advertised_choices() -> Vec<String> {
    let out = drive("usi\nquit\n");
    let line = out
        .lines()
        .find(|l| l.starts_with("option name BookFile type combo "))
        .unwrap_or_else(|| panic!("no BookFile combo line in the handshake:\n{out}"))
        .to_string();
    let choices: Vec<String> = line
        .split(" var ")
        .skip(1)
        .map(str::to_string)
        .collect::<Vec<_>>();
    assert!(!choices.is_empty(), "combo line has no `var`: {line}");
    choices
}

/// A session that points `BookDir`/`EvalDir` at `dir`, selects `choice`, and
/// relaxes the book filters so the fixture's top move survives.
fn session_for(dir: &str, choice: &str) -> String {
    format!(
        "usi\n\
         setoption name Threads value 1\n\
         setoption name EvalDir value {dir}\n\
         setoption name BookDir value {dir}\n\
         setoption name BookFile value {choice}\n\
         setoption name BookDepthLimit value 0\n\
         setoption name BookEvalBlackLimit value -99999\n\
         setoption name BookEvalWhiteLimit value -99999\n\
         setoption name BookEvalDiff value 0\n\
         setoption name BookMoves value 10000\n\
         isready\n\
         position sfen {STARTPOS_B}\n\
         go depth 1\n\
         quit\n"
    )
}

#[test]
#[cfg_attr(miri, ignore)]
fn the_advertised_choice_list_is_the_expected_ybb_set() {
    assert_eq!(
        advertised_choices(),
        vec![
            "no_book",
            "standard_book.ybb",
            "yaneura_book1.ybb",
            "yaneura_book2.ybb",
            "yaneura_book3.ybb",
            "yaneura_book4.ybb",
            "user_book1.ybb",
            "user_book2.ybb",
            "user_book3.ybb",
            "book.ybb",
        ]
    );
}

#[test]
#[cfg_attr(miri, ignore)]
fn every_advertised_choice_is_accepted_by_setoption() {
    // A combo rejects any value outside its list with an `option … rejected:`
    // info string. No advertised choice may take that path.
    for (i, choice) in advertised_choices().iter().enumerate() {
        let dir = TempDir::new(&format!("choice-accept-{i}"));
        let d = dir.path().to_str().unwrap();
        let out = drive(&format!(
            "usi\nsetoption name BookDir value {d}\nsetoption name BookFile value {choice}\nquit\n"
        ));
        assert!(
            !out.contains("rejected"),
            "advertised choice `{choice}` was rejected:\n{out}"
        );
    }
}

#[test]
#[cfg_attr(miri, ignore)]
fn every_advertised_book_choice_loads_from_book_dir() {
    for (i, choice) in advertised_choices().iter().enumerate() {
        let dir = TempDir::new(&format!("choice-load-{i}"));
        write_synthetic_nn_bin(dir.path());
        let d = dir.path().to_str().unwrap();

        if choice == "no_book" {
            // The sentinel: bookless by construction, so nothing loads and a
            // real search runs. Even a stray file named `no_book` is irrelevant
            // — the name has no book extension, hence no priority series.
            let out = drive_with_seed(&session_for(d, choice), TEST_BOOK_SEED);
            assert!(
                !out.contains("book loaded : "),
                "`no_book` must load nothing:\n{out}"
            );
            assert!(
                out.lines().any(|l| l.starts_with("info depth 1 ")),
                "`no_book` must run a real search:\n{out}"
            );
            continue;
        }

        // The committed fixture, staged under the advertised name verbatim.
        stage_sample_ybb(dir.path(), choice);
        let out = drive_with_seed(&session_for(d, choice), TEST_BOOK_SEED);

        assert!(
            out.contains("book loaded : 4 positions"),
            "`{choice}` must load the staged fixture:\n{out}"
        );
        assert!(
            !out.contains("info string book load failed"),
            "`{choice}` reported a load failure:\n{out}"
        );
        assert!(
            !out.contains("info string unsupported book format"),
            "`{choice}` was rejected as an unsupported format:\n{out}"
        );
        assert!(
            out.contains("info depth 0 "),
            "`{choice}` must short-circuit the search with a book hit:\n{out}"
        );
        let bm = bestmove_lines(&out);
        assert_eq!(bm.len(), 1, "one bestmove for `{choice}`:\n{out}");
        assert_eq!(
            bm[0].split_whitespace().next().unwrap(),
            "7g7f",
            "`{choice}` must answer with the fixture's book move:\n{out}"
        );
    }
}

#[test]
#[cfg_attr(miri, ignore)]
fn every_advertised_book_choice_drives_its_priority_series() {
    // The advertised names must work with the Multiple Book enumeration too:
    // a `<stem>-000.ybb` alongside the base is picked up first, and the base
    // still answers what `-000` misses.
    for (i, choice) in advertised_choices().iter().enumerate() {
        if choice == "no_book" {
            continue;
        }
        let dir = TempDir::new(&format!("choice-series-{i}"));
        write_synthetic_nn_bin(dir.path());
        let d = dir.path().to_str().unwrap();

        let stem = choice.strip_suffix(".ybb").expect("a `.ybb` choice");
        stage_sample_ybb(dir.path(), &format!("{stem}-000.ybb"));
        stage_sample_ybb(dir.path(), choice);

        let out = drive_with_seed(&session_for(d, choice), TEST_BOOK_SEED);
        assert!(
            out.matches("book loaded : 4 positions").count() == 2,
            "`{choice}`: both the `-000` slot and the base must load:\n{out}"
        );
        assert_eq!(
            bestmove_lines(&out)[0].split_whitespace().next().unwrap(),
            "7g7f",
            "`{choice}` series must still answer from the book:\n{out}"
        );
    }
}
