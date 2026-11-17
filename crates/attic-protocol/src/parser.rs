/// Input-validation limit: lines longer than this become
/// `Command::TooLong` and are not parsed further.
pub const MAX_LINE_BYTES: usize = 64 * 1024;

/// `position` command's first argument: either the implicit start position or
/// an explicit four-token SFEN string preserved verbatim for re-parsing.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PositionSfen {
    StartPos,
    Sfen(String),
}

/// All USI `go` sub-tokens captured verbatim, including the ones the driver
/// does not act on, so the parse is lossless. Each field has the integer type
/// the reference engine reads it into, so a value it accepts (a negative
/// increment, a negative `nodes` wrapping to a huge count) is accepted here
/// with the same meaning. A zero is kept as given: the reference reads a zero
/// limit as no limit, and the driver applies that only after it has seeded
/// `depth` and `nodes` from their options, which an explicit zero overrides.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct GoLimits {
    pub depth: Option<i32>,
    pub nodes: Option<u64>,
    /// Never negative: a negative `movetime` is clamped to 1.
    pub movetime: Option<i64>,
    /// Never negative: a negative `wtime` is clamped to 1.
    pub wtime: Option<i64>,
    /// Never negative: a negative `btime` is clamped to 1.
    pub btime: Option<i64>,
    pub winc: Option<i64>,
    pub binc: Option<i64>,
    pub byoyomi: Option<i64>,
    pub infinite: bool,
    /// `go ponder` — think on the predicted position; hold the reply until
    /// `ponderhit` or `stop`.
    pub ponder: bool,
    /// `go mate [ms|infinite]` — mate-search mode. In USI (unlike UCI) the token
    /// after `mate` is a TIME BUDGET in milliseconds, not a move count;
    /// `infinite` or a bare `mate` means unlimited. `None` means this is not a
    /// mate search; `Some(ms)` carries the budget, with the sentinel
    /// [`MATE_UNLIMITED_MS`] standing for unlimited (the reference's
    /// `INT32_MAX`).
    pub mate: Option<i32>,
    /// `go rtime <ms>` — a randomised minimum-thinking-time budget used for
    /// self-play variety. `init_` seeds all three time bounds to `rtime` (plus a
    /// decaying random bump) and returns early. `None` means no `rtime`.
    pub rtime: Option<i64>,
}

/// The `go mate` unlimited-budget sentinel (`limits.mate = INT32_MAX`):
/// `go mate infinite` and a bare `go mate` both map here.
pub const MATE_UNLIMITED_MS: i32 = i32::MAX;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Command {
    Usi,
    IsReady,
    SetOption {
        name: String,
        value: String,
    },
    UsiNewGame,
    Position {
        sfen: PositionSfen,
        moves: Vec<String>,
    },
    Go(GoLimits),
    Stop,
    /// `gameover [win|lose|draw]` — the game ended. The optional result token is
    /// ignored; the command is treated exactly like `stop`: over a shogi GUI, an
    /// opponent resign during `go ponder` arrives as `gameover` without a
    /// preceding `stop`, so it must release a held reply.
    GameOver,
    /// `ponderhit` — the opponent played the pondered move; commit the search.
    PonderHit,
    /// `bench [ttSizeMB] [threads] [limit] [default|current|<fenFile>] [limitType]`
    /// — the reproducible NPS benchmark. The raw trailing tokens are carried
    /// verbatim; [`crate::bench::parse_bench`] gives them meaning (defaults, limit
    /// type, position source).
    Bench(Vec<String>),
    /// A command the reference engine refuses by terminating: it prints an
    /// `info string CRITICAL ERROR` line quoting `command` and `reason`, then
    /// exits with status 1. `command` is the input line as read, without its
    /// trailing newline.
    CriticalError {
        command: String,
        reason: String,
    },
    Quit,
    Unknown(String),
    TooLong,
}

pub fn parse_line(input: &str) -> Command {
    if input.len() > MAX_LINE_BYTES {
        return Command::TooLong;
    }
    let trimmed = input.trim_matches(|c: char| c == '\r' || c == '\n' || c.is_whitespace());
    if trimmed.is_empty() {
        return Command::Unknown(String::new());
    }
    let mut parts = trimmed.split_whitespace();
    let head = parts.next().unwrap_or("");
    match head {
        "usi" => Command::Usi,
        "isready" => Command::IsReady,
        "usinewgame" => Command::UsiNewGame,
        "quit" => Command::Quit,
        "setoption" => parse_setoption(parts),
        "position" => parse_position(trimmed, parts),
        "go" => parse_go(input.strip_suffix('\n').unwrap_or(input), trimmed),
        "stop" => Command::Stop,
        // `gameover [result]`: the trailing win/lose/draw token is optional and
        // ignored — the command is handled identically to `stop`.
        "gameover" => Command::GameOver,
        "ponderhit" => Command::PonderHit,
        // The trailing `bench` tokens are preserved verbatim for the semantic
        // parse in `crate::bench` (which fills defaults and validates them).
        "bench" => Command::Bench(parts.map(str::to_string).collect()),
        _ => Command::Unknown(trimmed.to_string()),
    }
}

fn parse_position<'a>(line: &str, parts: impl Iterator<Item = &'a str>) -> Command {
    let tokens: Vec<&str> = parts.collect();
    let Some((&kind, rest)) = tokens.split_first() else {
        return Command::Unknown(line.to_string());
    };
    let (sfen, after_sfen) = match kind {
        "startpos" => (PositionSfen::StartPos, rest),
        "sfen" => {
            // The four SFEN tokens are: board, side-to-move, hands, ply. We pass
            // the joined string to `attic_state::parse_sfen` in the driver and
            // surface any per-field error from there.
            if rest.len() < 4 {
                return Command::Unknown(line.to_string());
            }
            let sfen_str = rest[..4].join(" ");
            (PositionSfen::Sfen(sfen_str), &rest[4..])
        }
        _ => return Command::Unknown(line.to_string()),
    };
    let moves = match after_sfen {
        [] => Vec::new(),
        ["moves", rest @ ..] => rest.iter().map(|s| (*s).to_string()).collect(),
        _ => return Command::Unknown(line.to_string()),
    };
    Command::Position { sfen, moves }
}

/// Parse the arguments of `go` as the reference engine reads them from a C++
/// input stream, so the same lines are accepted, clamped or refused.
///
/// A number is read as the longest prefix of the next word that forms an
/// integer of the field's type; whatever follows the digits is read as the
/// next word (`go btime 1.5` is `btime 1`, then a word `.5`). A missing,
/// non-numeric or out-of-range number is a [`Command::CriticalError`], as is a
/// `go mate` budget that is not a whole integer word.
///
/// A word that is not a key is skipped and the line is still searched, as in
/// the reference: `go nodes 12abc` is `nodes 12`, `go btime 1.5` is `btime 1`.
/// `searchmoves`, which this port does not support, makes the line
/// [`Command::Unknown`] at once, since the reference takes every remaining
/// word as a move. A `perft` with a nonzero depth, which the reference answers
/// with a move count instead of a search, makes the line
/// [`Command::Unknown`] too, but only once the rest of the line has been read,
/// so a malformed number after it is still refused.
fn parse_go(raw: &str, trimmed: &str) -> Command {
    let mut args = ArgStream {
        rest: &trimmed["go".len()..],
    };
    let critical = |key: &str| Command::CriticalError {
        command: raw.to_string(),
        reason: format!("Invalid argument for '{key}'"),
    };
    let mut limits = GoLimits::default();
    let mut perft = false;
    while let Some(key) = args.word() {
        match key {
            "searchmoves" => return Command::Unknown(trimmed.to_string()),
            "infinite" => limits.infinite = true,
            "ponder" => limits.ponder = true,
            // `go mate [ms|infinite]`: the word after `mate` is a millisecond
            // time budget; `infinite`, or nothing following, means unlimited.
            "mate" => match args.word() {
                None | Some("infinite") => limits.mate = Some(MATE_UNLIMITED_MS),
                Some(word) => {
                    let mut budget = ArgStream { rest: word };
                    match budget.integer() {
                        Some(v) if budget.rest.is_empty() => limits.mate = Some(v),
                        _ => return critical("mate"),
                    }
                }
            },
            "depth" => match args.integer() {
                Some(v) => limits.depth = Some(v),
                None => return critical(key),
            },
            "nodes" => match args.integer() {
                Some(v) => limits.nodes = Some(v),
                None => return critical(key),
            },
            // `go perft` is a reference engine feature this port lacks. A
            // later `perft` overwrites the depth, and a zero one is not a
            // perft, so the reference searches.
            "perft" => match args.integer::<i32>() {
                Some(depth) => perft = depth != 0,
                None => return critical(key),
            },
            "movetime" | "wtime" | "btime" | "winc" | "binc" | "byoyomi" | "rtime" => {
                let Some(v) = args.integer::<i64>() else {
                    return critical(key);
                };
                let field = match key {
                    "movetime" => &mut limits.movetime,
                    "wtime" => &mut limits.wtime,
                    "btime" => &mut limits.btime,
                    "winc" => &mut limits.winc,
                    "binc" => &mut limits.binc,
                    "byoyomi" => &mut limits.byoyomi,
                    _ => &mut limits.rtime,
                };
                *field = Some(v);
            }
            _ => {}
        }
    }
    if perft {
        return Command::Unknown(trimmed.to_string());
    }
    for t in [&mut limits.wtime, &mut limits.btime, &mut limits.movetime]
        .into_iter()
        .flatten()
    {
        if *t < 0 {
            *t = 1;
        }
    }
    Command::Go(limits)
}

/// The whitespace a C++ input stream skips in the "C" locale.
fn is_stream_space(c: char) -> bool {
    matches!(c, ' ' | '\t' | '\n' | '\x0b' | '\x0c' | '\r')
}

/// A cursor over the arguments of a command line that reads words and
/// integers the way a C++ `std::istream` extracts them.
struct ArgStream<'a> {
    rest: &'a str,
}

impl<'a> ArgStream<'a> {
    /// The next whitespace-delimited word, or `None` at the end of the line.
    fn word(&mut self) -> Option<&'a str> {
        let s = self.rest.trim_start_matches(is_stream_space);
        let end = s.find(is_stream_space).unwrap_or(s.len());
        self.rest = &s[end..];
        (end > 0).then(|| &s[..end])
    }

    /// The integer at the cursor: an optional sign and at least one decimal
    /// digit, consumed up to the first non-digit. `None` when there is no
    /// digit or the value does not fit `T`. For an unsigned `T` a minus sign
    /// negates modulo 2^N, as the stream does.
    fn integer<T: StreamInteger>(&mut self) -> Option<T> {
        let s = self.rest.trim_start_matches(is_stream_space);
        let (negative, body) = match s.as_bytes().first() {
            Some(b'-') => (true, &s[1..]),
            Some(b'+') => (false, &s[1..]),
            _ => (false, s),
        };
        let digits = body.bytes().take_while(u8::is_ascii_digit).count();
        self.rest = &body[digits..];
        if digits == 0 {
            return None;
        }
        let magnitude = body[..digits].bytes().try_fold(0u64, |acc, b| {
            acc.checked_mul(10)?.checked_add(u64::from(b - b'0'))
        })?;
        T::from_sign_magnitude(negative, magnitude)
    }
}

trait StreamInteger: Sized {
    fn from_sign_magnitude(negative: bool, magnitude: u64) -> Option<Self>;
}

impl StreamInteger for u64 {
    fn from_sign_magnitude(negative: bool, magnitude: u64) -> Option<Self> {
        Some(if negative {
            magnitude.wrapping_neg()
        } else {
            magnitude
        })
    }
}

macro_rules! signed_stream_integer {
    ($($t:ty),*) => {$(
        impl StreamInteger for $t {
            fn from_sign_magnitude(negative: bool, magnitude: u64) -> Option<Self> {
                let v = i128::from(magnitude);
                <$t>::try_from(if negative { -v } else { v }).ok()
            }
        }
    )*};
}
signed_stream_integer!(i32, i64);

fn parse_setoption<'a>(parts: impl Iterator<Item = &'a str>) -> Command {
    // USI: setoption name <NAME...> [value <VALUE...>]
    // Read as the reference reads it: the first token is skipped whatever it
    // is, the name is every token up to the first `value`, and the value is
    // everything after it, each joined back with single spaces. A line with no
    // name therefore sets the empty name, which is reported as unknown.
    let mut parts = parts.skip(1);
    let name: Vec<&str> = parts.by_ref().take_while(|&t| t != "value").collect();
    let value: Vec<&str> = parts.collect();
    Command::SetOption {
        name: name.join(" "),
        value: value.join(" "),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_usi() {
        assert_eq!(parse_line("usi"), Command::Usi);
        assert_eq!(parse_line("usi\n"), Command::Usi);
        assert_eq!(parse_line("usi\r\n"), Command::Usi);
        assert_eq!(parse_line("  usi  "), Command::Usi);
    }

    #[test]
    fn parses_isready() {
        assert_eq!(parse_line("isready"), Command::IsReady);
    }

    #[test]
    fn parses_usinewgame() {
        assert_eq!(parse_line("usinewgame"), Command::UsiNewGame);
    }

    #[test]
    fn parses_quit() {
        assert_eq!(parse_line("quit"), Command::Quit);
    }

    #[test]
    fn parses_setoption_with_value() {
        assert_eq!(
            parse_line("setoption name USI_Hash value 1024"),
            Command::SetOption {
                name: "USI_Hash".to_string(),
                value: "1024".to_string(),
            }
        );
    }

    #[test]
    fn parses_setoption_with_empty_value() {
        assert_eq!(
            parse_line("setoption name EvalDir value"),
            Command::SetOption {
                name: "EvalDir".to_string(),
                value: String::new(),
            }
        );
    }

    #[test]
    fn parses_setoption_with_no_value_keyword() {
        assert_eq!(
            parse_line("setoption name UsiNewGameThing"),
            Command::SetOption {
                name: "UsiNewGameThing".to_string(),
                value: String::new(),
            }
        );
    }

    #[test]
    fn parses_setoption_name_up_to_the_first_value_word() {
        assert_eq!(
            parse_line("setoption name No  Such Option value 1 value 2"),
            Command::SetOption {
                name: "No Such Option".to_string(),
                value: "1 value 2".to_string(),
            }
        );
    }

    #[test]
    fn parses_setoption_without_a_name_as_the_empty_name() {
        for line in ["setoption", "setoption name", "setoption name value 5"] {
            assert_eq!(
                parse_line(line),
                Command::SetOption {
                    name: String::new(),
                    value: if line.ends_with('5') { "5" } else { "" }.to_string(),
                },
                "{line}"
            );
        }
    }

    #[test]
    fn parses_setoption_skipping_the_first_word_whatever_it_is() {
        assert_eq!(
            parse_line("setoption nmae Threads value 2"),
            Command::SetOption {
                name: "Threads".to_string(),
                value: "2".to_string(),
            }
        );
    }

    #[test]
    fn parses_setoption_with_multi_word_value() {
        assert_eq!(
            parse_line("setoption name EvalDir value /srv/eval dir/sub"),
            Command::SetOption {
                name: "EvalDir".to_string(),
                value: "/srv/eval dir/sub".to_string(),
            }
        );
    }

    #[test]
    fn unknown_command_preserves_trimmed_line() {
        assert_eq!(
            parse_line("frobnicate the gizmo"),
            Command::Unknown("frobnicate the gizmo".to_string())
        );
    }

    #[test]
    fn parses_position_startpos() {
        assert_eq!(
            parse_line("position startpos"),
            Command::Position {
                sfen: PositionSfen::StartPos,
                moves: Vec::new(),
            }
        );
    }

    #[test]
    fn parses_position_startpos_with_moves() {
        assert_eq!(
            parse_line("position startpos moves 7g7f 8c8d"),
            Command::Position {
                sfen: PositionSfen::StartPos,
                moves: vec!["7g7f".to_string(), "8c8d".to_string()],
            }
        );
    }

    #[test]
    fn parses_position_sfen_no_moves() {
        let sfen = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1";
        assert_eq!(
            parse_line(&format!("position sfen {sfen}")),
            Command::Position {
                sfen: PositionSfen::Sfen(sfen.to_string()),
                moves: Vec::new(),
            }
        );
    }

    #[test]
    fn parses_position_sfen_with_moves() {
        let sfen = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1";
        assert_eq!(
            parse_line(&format!("position sfen {sfen} moves 7g7f")),
            Command::Position {
                sfen: PositionSfen::Sfen(sfen.to_string()),
                moves: vec!["7g7f".to_string()],
            }
        );
    }

    #[test]
    fn position_without_kind_token_is_unknown() {
        assert_eq!(
            parse_line("position"),
            Command::Unknown("position".to_string())
        );
        assert_eq!(
            parse_line("position something"),
            Command::Unknown("position something".to_string())
        );
    }

    #[test]
    fn position_sfen_short_field_count_is_unknown() {
        // Only three tokens (missing ply) → cannot form a valid SFEN.
        assert_eq!(
            parse_line("position sfen a b c"),
            Command::Unknown("position sfen a b c".to_string())
        );
    }

    #[test]
    fn parses_bare_go() {
        assert_eq!(parse_line("go"), Command::Go(GoLimits::default()));
    }

    #[test]
    fn parses_go_depth() {
        let expected = GoLimits {
            depth: Some(8),
            ..Default::default()
        };
        assert_eq!(parse_line("go depth 8"), Command::Go(expected));
    }

    #[test]
    fn parses_go_nodes_movetime_combined() {
        let expected = GoLimits {
            nodes: Some(1000),
            movetime: Some(250),
            ..Default::default()
        };
        assert_eq!(
            parse_line("go nodes 1000 movetime 250"),
            Command::Go(expected)
        );
    }

    #[test]
    fn parses_go_infinite() {
        let expected = GoLimits {
            infinite: true,
            ..Default::default()
        };
        assert_eq!(parse_line("go infinite"), Command::Go(expected));
    }

    #[test]
    fn parses_go_time_controls() {
        let expected = GoLimits {
            wtime: Some(60000),
            btime: Some(60000),
            byoyomi: Some(5000),
            ..Default::default()
        };
        assert_eq!(
            parse_line("go wtime 60000 btime 60000 byoyomi 5000"),
            Command::Go(expected)
        );
    }

    #[test]
    fn parses_go_winc_binc() {
        let expected = GoLimits {
            winc: Some(1000),
            binc: Some(2000),
            ..Default::default()
        };
        assert_eq!(parse_line("go winc 1000 binc 2000"), Command::Go(expected));
    }

    #[test]
    fn go_searchmoves_is_unknown() {
        assert_eq!(
            parse_line("go searchmoves 7g7f"),
            Command::Unknown("go searchmoves 7g7f".to_string())
        );
    }

    fn critical(command: &str, key: &str) -> Command {
        Command::CriticalError {
            command: command.to_string(),
            reason: format!("Invalid argument for '{key}'"),
        }
    }

    #[test]
    fn go_with_missing_value_is_critical() {
        for key in [
            "depth", "nodes", "movetime", "wtime", "btime", "winc", "binc", "byoyomi", "rtime",
            "perft",
        ] {
            let line = format!("go {key}");
            assert_eq!(parse_line(&line), critical(&line, key));
        }
    }

    #[test]
    fn go_with_non_integer_value_is_critical() {
        for (line, key) in [
            ("go nodes not-a-number", "nodes"),
            ("go btime x", "btime"),
            ("go byoyomi depth 3", "byoyomi"),
            ("go depth -", "depth"),
            ("go wtime +", "wtime"),
            ("go movetime .5", "movetime"),
            ("go perft x", "perft"),
        ] {
            assert_eq!(parse_line(line), critical(line, key));
        }
    }

    #[test]
    fn go_with_out_of_range_value_is_critical() {
        for (line, key) in [
            ("go depth 2147483648", "depth"),
            ("go depth -2147483649", "depth"),
            ("go btime 9223372036854775808", "btime"),
            ("go nodes 18446744073709551616", "nodes"),
            ("go nodes -18446744073709551616", "nodes"),
            ("go mate 2147483648", "mate"),
        ] {
            assert_eq!(parse_line(line), critical(line, key));
        }
    }

    #[test]
    fn go_critical_error_quotes_the_line_as_read() {
        assert_eq!(
            parse_line("go  wtime  abc \n"),
            critical("go  wtime  abc ", "wtime")
        );
        assert_eq!(
            parse_line("go btime x\r\n"),
            critical("go btime x\r", "btime")
        );
    }

    #[test]
    fn go_malformed_value_after_skipped_word_is_critical() {
        assert_eq!(
            parse_line("go movestogo btime x"),
            critical("go movestogo btime x", "btime")
        );
        assert_eq!(
            parse_line("go perft 3 btime x"),
            critical("go perft 3 btime x", "btime")
        );
    }

    #[test]
    fn go_skips_a_word_after_a_number() {
        let expected = GoLimits {
            nodes: Some(12),
            ..Default::default()
        };
        assert_eq!(parse_line("go nodes 12abc"), Command::Go(expected));
    }

    #[test]
    fn go_skips_a_word_between_two_keys() {
        let expected = GoLimits {
            depth: Some(2),
            nodes: Some(1000),
            ..Default::default()
        };
        assert_eq!(
            parse_line("go depth 2 abc nodes 1000"),
            Command::Go(expected.clone())
        );
        assert_eq!(
            parse_line("go movestogo 40 depth 2 nodes 1000"),
            Command::Go(expected)
        );
    }

    #[test]
    fn go_reads_a_fraction_as_its_integer_part() {
        let expected = GoLimits {
            btime: Some(1),
            ..Default::default()
        };
        assert_eq!(parse_line("go btime 1.5"), Command::Go(expected));
    }

    #[test]
    fn go_keeps_zero_limits_as_given() {
        let expected = GoLimits {
            depth: Some(0),
            nodes: Some(0),
            movetime: Some(0),
            wtime: Some(0),
            btime: Some(0),
            winc: Some(0),
            binc: Some(0),
            byoyomi: Some(0),
            mate: Some(0),
            rtime: Some(0),
            ..Default::default()
        };
        assert_eq!(
            parse_line(
                "go depth 0 nodes 0 movetime 0 wtime 0 btime 0 winc 0 binc 0 byoyomi 0 mate 0 rtime 0"
            ),
            Command::Go(expected)
        );
        let depth = GoLimits {
            depth: Some(0),
            ..Default::default()
        };
        assert_eq!(parse_line("go depth 0"), Command::Go(depth));
        let movetime = GoLimits {
            movetime: Some(0),
            ..Default::default()
        };
        assert_eq!(parse_line("go movetime 0"), Command::Go(movetime));
    }

    #[test]
    fn go_keeps_a_negative_depth() {
        let expected = GoLimits {
            depth: Some(-1),
            ..Default::default()
        };
        assert_eq!(parse_line("go depth -1"), Command::Go(expected));
    }

    #[test]
    fn go_perft_zero_is_a_search() {
        let expected = GoLimits {
            depth: Some(3),
            ..Default::default()
        };
        assert_eq!(parse_line("go perft 0 depth 3"), Command::Go(expected));
        assert_eq!(
            parse_line("go perft 3 perft 0 depth 3"),
            Command::Go(GoLimits {
                depth: Some(3),
                ..Default::default()
            })
        );
    }

    #[test]
    fn go_malformed_value_after_searchmoves_is_unknown() {
        assert_eq!(
            parse_line("go searchmoves 7g7f btime x"),
            Command::Unknown("go searchmoves 7g7f btime x".to_string())
        );
    }

    #[test]
    fn go_negative_times_clamp_to_one() {
        let expected = GoLimits {
            btime: Some(1),
            wtime: Some(1),
            movetime: Some(1),
            ..Default::default()
        };
        assert_eq!(
            parse_line("go btime -5 wtime -7 movetime -1000"),
            Command::Go(expected)
        );
    }

    #[test]
    fn go_zero_times_are_not_clamped() {
        let expected = GoLimits {
            btime: Some(0),
            wtime: Some(0),
            movetime: Some(0),
            ..Default::default()
        };
        assert_eq!(
            parse_line("go btime 0 wtime 0 movetime 0"),
            Command::Go(expected)
        );
    }

    #[test]
    fn go_other_negative_values_are_kept() {
        let expected = GoLimits {
            depth: Some(-3),
            winc: Some(-5),
            binc: Some(-6),
            byoyomi: Some(-100),
            rtime: Some(-7),
            mate: Some(-5),
            ..Default::default()
        };
        assert_eq!(
            parse_line("go depth -3 winc -5 binc -6 byoyomi -100 rtime -7 mate -5"),
            Command::Go(expected)
        );
    }

    #[test]
    fn go_negative_nodes_wraps() {
        let expected = GoLimits {
            nodes: Some(u64::MAX - 4),
            ..Default::default()
        };
        assert_eq!(parse_line("go nodes -5"), Command::Go(expected));
    }

    #[test]
    fn go_reads_the_numeric_prefix_and_the_rest_as_the_next_word() {
        let expected = GoLimits {
            btime: Some(100),
            depth: Some(3),
            ..Default::default()
        };
        assert_eq!(parse_line("go btime +100depth 3"), Command::Go(expected));
    }

    #[test]
    fn go_perft_is_unknown() {
        assert_eq!(
            parse_line("go perft 3"),
            Command::Unknown("go perft 3".to_string())
        );
        assert_eq!(
            parse_line("go perft 0 perft -2"),
            Command::Unknown("go perft 0 perft -2".to_string())
        );
    }

    #[test]
    fn parses_stop() {
        assert_eq!(parse_line("stop"), Command::Stop);
        assert_eq!(parse_line("stop\n"), Command::Stop);
    }

    #[test]
    fn parses_gameover_with_and_without_result() {
        assert_eq!(parse_line("gameover"), Command::GameOver);
        assert_eq!(parse_line("gameover win"), Command::GameOver);
        assert_eq!(parse_line("gameover lose"), Command::GameOver);
        assert_eq!(parse_line("gameover draw"), Command::GameOver);
        assert_eq!(parse_line("gameover\n"), Command::GameOver);
    }

    #[test]
    fn parses_bare_bench() {
        assert_eq!(parse_line("bench"), Command::Bench(Vec::new()));
    }

    #[test]
    fn parses_bench_with_all_tokens() {
        assert_eq!(
            parse_line("bench 16 1 6 default depth"),
            Command::Bench(
                ["16", "1", "6", "default", "depth"]
                    .iter()
                    .map(|s| s.to_string())
                    .collect()
            )
        );
    }

    #[test]
    fn parses_ponderhit() {
        assert_eq!(parse_line("ponderhit"), Command::PonderHit);
        assert_eq!(parse_line("ponderhit\n"), Command::PonderHit);
    }

    #[test]
    fn parses_go_ponder() {
        let expected = GoLimits {
            ponder: true,
            ..Default::default()
        };
        assert_eq!(parse_line("go ponder"), Command::Go(expected));
    }

    #[test]
    fn parses_go_ponder_with_time() {
        let expected = GoLimits {
            ponder: true,
            btime: Some(1000),
            wtime: Some(1000),
            ..Default::default()
        };
        assert_eq!(
            parse_line("go ponder btime 1000 wtime 1000"),
            Command::Go(expected)
        );
    }

    #[test]
    fn parses_go_mate_with_budget() {
        let expected = GoLimits {
            mate: Some(5000),
            ..Default::default()
        };
        assert_eq!(parse_line("go mate 5000"), Command::Go(expected));
    }

    #[test]
    fn parses_go_mate_bare_is_unlimited() {
        let expected = GoLimits {
            mate: Some(MATE_UNLIMITED_MS),
            ..Default::default()
        };
        assert_eq!(parse_line("go mate"), Command::Go(expected));
    }

    #[test]
    fn parses_go_mate_infinite_is_unlimited() {
        let expected = GoLimits {
            mate: Some(MATE_UNLIMITED_MS),
            ..Default::default()
        };
        assert_eq!(parse_line("go mate infinite"), Command::Go(expected));
    }

    #[test]
    fn go_mate_budget_must_be_a_whole_integer_word() {
        for line in ["go mate soon", "go mate 12abc", "go mate 1.5", "go mate -"] {
            assert_eq!(parse_line(line), critical(line, "mate"));
        }
        let expected = GoLimits {
            mate: Some(5),
            ..Default::default()
        };
        assert_eq!(parse_line("go mate +5"), Command::Go(expected));
    }

    #[test]
    fn empty_line_is_unknown_empty() {
        assert_eq!(parse_line(""), Command::Unknown(String::new()));
        assert_eq!(parse_line("   \n"), Command::Unknown(String::new()));
    }

    #[test]
    fn oversized_line_returns_too_long() {
        let line = "x".repeat(MAX_LINE_BYTES + 1);
        assert_eq!(parse_line(&line), Command::TooLong);
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn line_at_max_size_is_parsed_normally() {
        // 64 KB exactly — still parses (becomes Unknown since it's not a command).
        let line = "x".repeat(MAX_LINE_BYTES);
        assert!(matches!(parse_line(&line), Command::Unknown(_)));
    }
}
