//! Parser for `engine_option_profile.txt`, the pre-handshake selector that
//! decides *which* engine-option group the engine grows.
//!
//! Ported from the reference `OptionsMap::read_engine_option_profile`
//! (`source/usioption.cpp`). The reference reads the
//! file from the current directory in `USIEngine::set_engine`, BEFORE
//! `add_options` builds the option map and therefore before the `usi` reply
//! (`usi.cpp`) — so the read must print nothing at all, not even an
//! `info string` (unlike `engine_options.txt`, which is read at `isready` and
//! does announce itself).
//!
//! Only one knob exists today: the book-option profile version
//! ([`BookOptionsVersion`]). V1 is the historical surface and the default; V2
//! swaps in the black/white-split book filters (see
//! [`crate::options::declarations`] and `attic_search::BookConfig`).
//!
//! The filename is taken as a parameter (as in the reference) so tests can point
//! at an isolated temp directory instead of relying on the process working
//! directory.

use std::path::Path;

/// The profile filename the production call site reads, resolved against the
/// process's current directory (`usi.cpp`).
pub const ENGINE_OPTION_PROFILE_FILE: &str = "engine_option_profile.txt";

/// Which book-option group to register.
///
/// Mirrors `OptionsMap::book_options_version` (`usioption.h`), whose default
/// is `1`; a missing / unreadable profile file leaves it at V1.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum BookOptionsVersion {
    /// The historical surface: `NarrowBook`, `BookEvalDiff`, `BookDepthLimit`,
    /// `ConsiderBookMoveCount`.
    #[default]
    V1,
    /// `BOOK_OPTIONS=V2`: black/white-split eval-diff and depth limits, and the
    /// large-book defaults (`BookOnTheFly` / `IgnoreBookPly` on, `BookMoves`
    /// 200).
    V2,
}

impl BookOptionsVersion {
    /// The reference accessor `OptionsMap::book_options_v2()`
    /// (`usioption.h`).
    pub fn is_v2(self) -> bool {
        matches!(self, BookOptionsVersion::V2)
    }
}

/// The characters the reference `StringExtension::trim` strips — and it strips
/// them from the END only (`misc.cpp`, `is_space` at `misc.cpp`).
const TRAILING_SPACE: [char; 4] = ['\r', '\n', ' ', '\t'];

/// Parse the contents of a profile file, mirroring the reference scan loop
/// (`usioption.cpp`): trailing-trim each line; skip empty lines, lines
/// starting with `#`, and lines starting with `//`; replace `=` with a space;
/// take the first whitespace token as the key. `BOOK_OPTIONS_V2` (any case) sets
/// V2; `BOOK_OPTIONS` takes its value from the next token (`V2` / `2` → V2,
/// `V1` / `1` → V1, anything else leaves the version untouched). Unknown keys
/// are ignored, and the last recognised line wins.
pub fn parse_engine_option_profile(contents: &str) -> BookOptionsVersion {
    let mut version = BookOptionsVersion::V1;

    for raw in contents.lines() {
        let line = raw.trim_end_matches(TRAILING_SPACE);

        if line.is_empty() || line.starts_with('#') || line.starts_with("//") {
            continue;
        }

        // `'=' → ' '` first, then whitespace tokenisation (`Parser::LineScanner`,
        // whose `get_text` yields `""` past end-of-line).
        let replaced = line.replace('=', " ");
        let mut tokens = replaced.split_whitespace();
        let key = tokens.next().unwrap_or("");

        if key.eq_ignore_ascii_case("BOOK_OPTIONS_V2") {
            version = BookOptionsVersion::V2;
            continue;
        }

        if key.eq_ignore_ascii_case("BOOK_OPTIONS") {
            let value = tokens.next().unwrap_or("");
            if value.eq_ignore_ascii_case("V2") || value == "2" {
                version = BookOptionsVersion::V2;
            } else if value.eq_ignore_ascii_case("V1") || value == "1" {
                version = BookOptionsVersion::V1;
            }
        }
    }

    version
}

/// Read `path` and parse it as a profile file. A missing / unreadable / non-UTF-8
/// file is the V1 default, silently — the reference `Open(...).is_not_ok()`
/// early-return (`usioption.cpp`). Nothing is printed on any path.
pub fn read_engine_option_profile(path: &Path) -> BookOptionsVersion {
    match std::fs::read_to_string(path) {
        Ok(text) => parse_engine_option_profile(&text),
        Err(_) => BookOptionsVersion::V1,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_or_absent_input_is_v1() {
        assert_eq!(parse_engine_option_profile(""), BookOptionsVersion::V1);
        assert_eq!(
            parse_engine_option_profile("\n\n   \n\t\n"),
            BookOptionsVersion::V1
        );
        // A path that cannot exist reads as V1 without erroring.
        assert_eq!(
            read_engine_option_profile(Path::new(
                "/nonexistent-dir-for-tests/engine_option_profile.txt"
            )),
            BookOptionsVersion::V1
        );
    }

    #[test]
    fn bare_v2_key_selects_v2() {
        assert_eq!(
            parse_engine_option_profile("BOOK_OPTIONS_V2\n"),
            BookOptionsVersion::V2
        );
        // No trailing newline, and a trailing-whitespace tail, are equivalent.
        assert_eq!(
            parse_engine_option_profile("BOOK_OPTIONS_V2"),
            BookOptionsVersion::V2
        );
        assert_eq!(
            parse_engine_option_profile("BOOK_OPTIONS_V2  \t\r\n"),
            BookOptionsVersion::V2
        );
    }

    #[test]
    fn book_options_key_takes_its_value() {
        for line in [
            "BOOK_OPTIONS = V2",
            "BOOK_OPTIONS=V2",
            "BOOK_OPTIONS V2",
            "BOOK_OPTIONS = 2",
            "BOOK_OPTIONS=2",
        ] {
            assert_eq!(
                parse_engine_option_profile(line),
                BookOptionsVersion::V2,
                "line {line:?}"
            );
        }
        for line in ["BOOK_OPTIONS = V1", "BOOK_OPTIONS=1", "BOOK_OPTIONS 1"] {
            assert_eq!(
                parse_engine_option_profile(line),
                BookOptionsVersion::V1,
                "line {line:?}"
            );
        }
    }

    #[test]
    fn keys_and_values_are_case_insensitive() {
        assert_eq!(
            parse_engine_option_profile("book_options_v2\n"),
            BookOptionsVersion::V2
        );
        assert_eq!(
            parse_engine_option_profile("Book_Options = v2\n"),
            BookOptionsVersion::V2
        );
        assert_eq!(
            parse_engine_option_profile("BOOK_OPTIONS_V2\nbook_options = v1\n"),
            BookOptionsVersion::V1
        );
    }

    #[test]
    fn comments_and_blank_lines_are_skipped() {
        let text = "\
# BOOK_OPTIONS_V2 in a hash comment\n\
// BOOK_OPTIONS_V2 in a slash comment\n\
\n\
   \n\
";
        assert_eq!(parse_engine_option_profile(text), BookOptionsVersion::V1);

        // A real directive surrounded by comments still applies.
        let text = "# header\nBOOK_OPTIONS_V2\n// trailer\n";
        assert_eq!(parse_engine_option_profile(text), BookOptionsVersion::V2);
    }

    #[test]
    fn unknown_keys_and_values_are_ignored() {
        assert_eq!(
            parse_engine_option_profile("SOME_OTHER_KEY = V2\nBOOK_OPTIONS_V3\n"),
            BookOptionsVersion::V1
        );
        // An unrecognised value leaves the current version untouched (the
        // reference sets neither branch).
        assert_eq!(
            parse_engine_option_profile("BOOK_OPTIONS_V2\nBOOK_OPTIONS = V9\n"),
            BookOptionsVersion::V2
        );
        assert_eq!(
            parse_engine_option_profile("BOOK_OPTIONS\n"),
            BookOptionsVersion::V1
        );
    }

    #[test]
    fn last_recognised_line_wins() {
        assert_eq!(
            parse_engine_option_profile("BOOK_OPTIONS = V1\nBOOK_OPTIONS = V2\n"),
            BookOptionsVersion::V2
        );
        assert_eq!(
            parse_engine_option_profile("BOOK_OPTIONS = V2\nBOOK_OPTIONS = V1\n"),
            BookOptionsVersion::V1
        );
    }

    #[test]
    fn reads_a_real_file() {
        let dir =
            std::env::temp_dir().join(format!("engine-option-profile-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("create temp dir");
        let path = dir.join(ENGINE_OPTION_PROFILE_FILE);
        std::fs::write(&path, "# profile\nBOOK_OPTIONS = V2\n").expect("write profile");
        assert_eq!(read_engine_option_profile(&path), BookOptionsVersion::V2);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
