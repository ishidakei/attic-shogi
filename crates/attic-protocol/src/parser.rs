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
/// does not act on, so the parse is lossless.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct GoLimits {
    pub depth: Option<u32>,
    pub nodes: Option<u64>,
    pub movetime: Option<u64>,
    pub wtime: Option<u64>,
    pub btime: Option<u64>,
    pub winc: Option<u64>,
    pub binc: Option<u64>,
    pub byoyomi: Option<u64>,
    pub infinite: bool,
    /// `go ponder` — think on the predicted position; hold the reply until
    /// `ponderhit` or `stop`.
    pub ponder: bool,
    /// `go mate [ms|infinite]` — mate-search mode (`usi.cpp`). In USI
    /// (unlike UCI) the token after `mate` is a TIME BUDGET in milliseconds, not
    /// a move count; `infinite` or a bare `mate` means unlimited. `None` means
    /// this is not a mate search; `Some(ms)` carries the budget, with the
    /// sentinel [`MATE_UNLIMITED_MS`] standing for unlimited (the pin's
    /// `INT32_MAX`).
    pub mate: Option<u64>,
    /// `go rtime <ms>` — a randomised minimum-thinking-time budget used for
    /// self-play variety (`timeman.cpp`). `init_` seeds all three time
    /// bounds to `rtime` (plus a decaying random bump) and returns early. `None`
    /// means no `rtime`.
    pub rtime: Option<u64>,
}

/// The `go mate` unlimited-budget sentinel (`limits.mate = INT32_MAX`,
/// `usi.cpp`): `go mate infinite` and a bare `go mate` both map here.
pub const MATE_UNLIMITED_MS: u64 = i32::MAX as u64;

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
    /// ignored; the command is treated exactly like `stop` (`usi.cpp`):
    /// over a shogi GUI, an opponent resign during `go ponder` arrives as
    /// `gameover` without a preceding `stop`, so it must release a held reply.
    GameOver,
    /// `ponderhit` — the opponent played the pondered move; commit the search.
    PonderHit,
    /// `bench [ttSizeMB] [threads] [limit] [default|current|<fenFile>] [limitType]`
    /// — the reproducible NPS benchmark (`benchmark.cpp` / `usi.cpp`). The raw
    /// trailing tokens are carried verbatim; [`crate::bench::parse_bench`] gives
    /// them meaning (defaults, limit type, position source).
    Bench(Vec<String>),
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
        "go" => parse_go(trimmed, parts),
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

fn parse_go<'a>(line: &str, parts: impl Iterator<Item = &'a str>) -> Command {
    let tokens: Vec<&str> = parts.collect();
    let mut limits = GoLimits::default();
    let mut i = 0;
    while i < tokens.len() {
        let key = tokens[i];
        match key {
            "infinite" => {
                limits.infinite = true;
                i += 1;
            }
            "ponder" => {
                limits.ponder = true;
                i += 1;
            }
            // `go mate [ms|infinite]` (`usi.cpp`): the token after `mate`
            // is a millisecond time budget; `infinite`, or nothing following,
            // means unlimited. Anything else that is not a valid `u64` is an
            // error (the pin's `stoi` would throw).
            "mate" => match tokens.get(i + 1) {
                None => {
                    limits.mate = Some(MATE_UNLIMITED_MS);
                    i += 1;
                }
                Some(&"infinite") => {
                    limits.mate = Some(MATE_UNLIMITED_MS);
                    i += 2;
                }
                Some(value) => {
                    let Ok(v) = value.parse::<u64>() else {
                        return Command::Unknown(line.to_string());
                    };
                    limits.mate = Some(v);
                    i += 2;
                }
            },
            "depth" | "nodes" | "movetime" | "wtime" | "btime" | "winc" | "binc" | "byoyomi"
            | "rtime" => {
                let Some(value) = tokens.get(i + 1) else {
                    return Command::Unknown(line.to_string());
                };
                if key == "depth" {
                    match value.parse::<u32>() {
                        Ok(v) => limits.depth = Some(v),
                        Err(_) => return Command::Unknown(line.to_string()),
                    }
                } else {
                    let Ok(v) = value.parse::<u64>() else {
                        return Command::Unknown(line.to_string());
                    };
                    match key {
                        "nodes" => limits.nodes = Some(v),
                        "movetime" => limits.movetime = Some(v),
                        "wtime" => limits.wtime = Some(v),
                        "btime" => limits.btime = Some(v),
                        "winc" => limits.winc = Some(v),
                        "binc" => limits.binc = Some(v),
                        "byoyomi" => limits.byoyomi = Some(v),
                        "rtime" => limits.rtime = Some(v),
                        _ => unreachable!("matched key {key} but no branch"),
                    }
                }
                i += 2;
            }
            _ => return Command::Unknown(line.to_string()),
        }
    }
    Command::Go(limits)
}

fn parse_setoption<'a>(parts: impl Iterator<Item = &'a str>) -> Command {
    // USI: setoption name <NAME> [value <VALUE...>]
    // Per the protocol, NAME is a single token (option names contain no spaces),
    // and everything after `value` is the value (joined back with single spaces).
    let tokens: Vec<&str> = parts.collect();
    let mut iter = tokens.iter();
    let Some(&kw) = iter.next() else {
        return Command::Unknown(format!("setoption {}", tokens.join(" ")));
    };
    if kw != "name" {
        return Command::Unknown(format!("setoption {}", tokens.join(" ")));
    }
    let Some(&name) = iter.next() else {
        return Command::Unknown(format!("setoption {}", tokens.join(" ")));
    };
    let rest: Vec<&str> = iter.copied().collect();
    let value = match rest.as_slice() {
        [] => String::new(),
        ["value"] => String::new(),
        ["value", rest @ ..] => rest.join(" "),
        _ => return Command::Unknown(format!("setoption {}", tokens.join(" "))),
    };
    Command::SetOption {
        name: name.to_string(),
        value,
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
    fn go_with_unknown_subtoken_is_unknown() {
        assert_eq!(
            parse_line("go searchmoves 7g7f"),
            Command::Unknown("go searchmoves 7g7f".to_string())
        );
    }

    #[test]
    fn go_with_missing_value_is_unknown() {
        assert_eq!(
            parse_line("go depth"),
            Command::Unknown("go depth".to_string())
        );
    }

    #[test]
    fn go_with_non_integer_value_is_unknown() {
        assert_eq!(
            parse_line("go nodes not-a-number"),
            Command::Unknown("go nodes not-a-number".to_string())
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
    fn go_mate_non_integer_budget_is_unknown() {
        assert_eq!(
            parse_line("go mate soon"),
            Command::Unknown("go mate soon".to_string())
        );
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
