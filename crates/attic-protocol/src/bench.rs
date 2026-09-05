//! The `bench` command's argument parsing and default position set — a port of
//! the reference `setup_bench` (`benchmark.cpp`, the non-Stockfish
//! branch) and its `Defaults` list (`benchmark.cpp`).
//!
//! Reference syntax (`benchmark.cpp`):
//!
//! ```text
//! bench [ttSizeMB] [threads] [limit] [default|current|<fenFile>] [limitType]
//! ```
//!
//! The defaults mirrored here are the reference's non-Stockfish ones
//! (`benchmark.cpp`): `ttSize=1024`, `threads=1`, `limit=15000`,
//! `fenFile=default`, `limitType=movetime` — the code, not the stale comment
//! example beside it, is the ground truth.
//!
//! The reference `limitType` also accepts `perft` and `eval`; those are outside
//! this NPS bench's scope and parse to a [`BenchParseError`].

use std::fs;

use attic_state::format_sfen;

use crate::parser::GoLimits;

/// The reference `Defaults` position list (`benchmark.cpp`), transcribed
/// verbatim. Used when the position source is `default` or omitted.
pub const BENCH_DEFAULT_POSITIONS: [&str; 4] = [
    // 初期局面に近い曲面。
    "lnsgkgsnl/1r7/p1ppp1bpp/1p3pp2/7P1/2P6/PP1PPPP1P/1B3S1R1/LNSGKG1NL b - 9",
    // 読めば読むほど後手悪いような局面
    "l4S2l/4g1gs1/5p1p1/pr2N1pkp/4Gn3/PP3PPPP/2GPP4/1K7/L3r+s2L w BS2N5Pb 1",
    // 57同銀は詰み、みたいな。読めば読むほど先手が悪いことがわかってくる局面。
    "6n1l/2+S1k4/2lp4p/1np1B2b1/3PP4/1N1S3rP/1P2+pPP+p1/1p1G5/3KG2r1 b GSN2L4Pgs2p 1",
    // 指し手生成祭りの局面 cf. http://d.hatena.ne.jp/ak11/20110508/p1
    "l6nl/5+P1gk/2np1S3/p1p4Pp/3P2Sp1/1PPb2P1P/P5GS1/R8/LN4bKL w RGgsn5p 1",
];

/// The reference non-Stockfish defaults (`benchmark.cpp`).
const DEFAULT_TT_MB: &str = "1024";
const DEFAULT_THREADS: &str = "1";
const DEFAULT_LIMIT: &str = "15000";
const DEFAULT_FEN_SOURCE: &str = "default";
const DEFAULT_LIMIT_TYPE: &str = "movetime";

/// A `bench` argument-parse failure, surfaced by the driver as an `info string`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BenchParseError(pub String);

impl std::fmt::Display for BenchParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// A fully-resolved `bench` invocation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BenchConfig {
    /// The `USI_Hash` value in MiB to `setoption`.
    pub tt_mb: i64,
    pub threads: i64,
    /// The per-position search limit, consumed exactly as a normal `go` would.
    pub limits: GoLimits,
    /// The positions to search, as SFEN strings.
    pub fens: Vec<String>,
}

/// Parse the `bench` argument tokens into a [`BenchConfig`], filling missing
/// trailing arguments with the reference defaults. `current_sfen` is used only
/// when the position source is `current`.
pub fn parse_bench(tokens: &[String], current_sfen: &str) -> Result<BenchConfig, BenchParseError> {
    let arg = |i: usize, default: &str| -> String {
        tokens
            .get(i)
            .map(String::as_str)
            .unwrap_or(default)
            .to_string()
    };
    let tt_arg = arg(0, DEFAULT_TT_MB);
    let threads_arg = arg(1, DEFAULT_THREADS);
    let limit_arg = arg(2, DEFAULT_LIMIT);
    let fen_source = arg(3, DEFAULT_FEN_SOURCE);
    let limit_type = arg(4, DEFAULT_LIMIT_TYPE);

    let tt_mb: i64 = tt_arg
        .parse()
        .map_err(|_| BenchParseError(format!("invalid ttSizeMB `{tt_arg}`")))?;
    let threads: i64 = threads_arg
        .parse()
        .map_err(|_| BenchParseError(format!("invalid threads `{threads_arg}`")))?;
    let limit: u64 = limit_arg
        .parse()
        .map_err(|_| BenchParseError(format!("invalid limit `{limit_arg}`")))?;

    let mut limits = GoLimits::default();
    match limit_type.as_str() {
        "depth" => {
            let d = u32::try_from(limit)
                .map_err(|_| BenchParseError(format!("depth limit out of range `{limit}`")))?;
            limits.depth = Some(d);
        }
        "nodes" => limits.nodes = Some(limit),
        "movetime" => limits.movetime = Some(limit),
        other => {
            return Err(BenchParseError(format!(
                "unsupported limit type `{other}` (supported: depth, nodes, movetime)"
            )));
        }
    }

    let fens = match fen_source.as_str() {
        "default" => BENCH_DEFAULT_POSITIONS
            .iter()
            .map(|s| s.to_string())
            .collect(),
        "current" => vec![current_sfen.to_string()],
        path => {
            let text = fs::read_to_string(path)
                .map_err(|e| BenchParseError(format!("unable to open file `{path}`: {e}")))?;
            let fens: Vec<String> = text
                .lines()
                .map(str::trim)
                .filter(|l| !l.is_empty())
                .map(str::to_string)
                .collect();
            if fens.is_empty() {
                return Err(BenchParseError(format!("no positions in file `{path}`")));
            }
            fens
        }
    };

    Ok(BenchConfig {
        tt_mb,
        threads,
        limits,
        fens,
    })
}

/// The SFEN of a position, for the `current` position source.
pub fn current_sfen(pos: &attic_state::Position) -> String {
    format_sfen(pos)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_when_no_args() {
        let cfg = parse_bench(&[], "startsfen").expect("defaults parse");
        assert_eq!(cfg.tt_mb, 1024);
        assert_eq!(cfg.threads, 1);
        assert_eq!(cfg.limits.movetime, Some(15000));
        assert_eq!(cfg.limits.depth, None);
        assert_eq!(cfg.fens.len(), 4);
        assert_eq!(cfg.fens[0], BENCH_DEFAULT_POSITIONS[0]);
    }

    #[test]
    fn depth_limit_type() {
        let tokens = ["16", "1", "6", "default", "depth"].map(String::from);
        let cfg = parse_bench(&tokens, "x").expect("parse");
        assert_eq!(cfg.tt_mb, 16);
        assert_eq!(cfg.limits.depth, Some(6));
        assert_eq!(cfg.limits.movetime, None);
    }

    #[test]
    fn nodes_limit_type() {
        let tokens = ["16", "1", "100000", "default", "nodes"].map(String::from);
        let cfg = parse_bench(&tokens, "x").expect("parse");
        assert_eq!(cfg.limits.nodes, Some(100000));
    }

    #[test]
    fn current_source_uses_given_sfen() {
        let tokens = ["16", "1", "4", "current", "depth"].map(String::from);
        let cfg = parse_bench(&tokens, "my-sfen").expect("parse");
        assert_eq!(cfg.fens, vec!["my-sfen".to_string()]);
    }

    #[test]
    fn garbage_tt_size_errors() {
        let tokens = ["notanumber"].map(String::from);
        assert!(parse_bench(&tokens, "x").is_err());
    }

    #[test]
    fn unsupported_limit_type_errors() {
        let tokens = ["16", "1", "5", "default", "perft"].map(String::from);
        let err = parse_bench(&tokens, "x").expect_err("perft unsupported");
        assert!(err.0.contains("perft"), "message names the type: {err}");
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn missing_file_errors() {
        let tokens = ["16", "1", "5", "/no/such/bench/file", "depth"].map(String::from);
        assert!(parse_bench(&tokens, "x").is_err());
    }
}
