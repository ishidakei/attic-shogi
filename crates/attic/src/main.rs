use std::io::{BufReader, stdin, stdout};
use std::process::ExitCode;
use std::sync::{Arc, Mutex};

use attic::perft;
use attic_protocol::UsiDriver;
use attic_state::{Position, parse_sfen, parse_usi_move};

const USAGE: &str = "\
usage:
  engine                                              # run USI event loop on stdin/stdout
  engine perft startpos <depth>
  engine perft sfen <SFEN-LITERAL> <depth>
  engine perft sfen <SFEN-LITERAL> moves <m1> [<m2> ...] <depth>
";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match dispatch(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(msg) => {
            eprintln!("error: {msg}");
            eprint!("{USAGE}");
            ExitCode::from(2)
        }
    }
}

fn dispatch(args: &[String]) -> Result<(), String> {
    if args.is_empty() {
        // The reader stays on the main thread; the writer is shared with the
        // search worker, so it must be `Send + 'static` — use the owned `Stdout`
        // handle (a `StdoutLock` is not `Send`) behind an `Arc<Mutex<_>>`.
        let writer = Arc::new(Mutex::new(stdout()));
        return UsiDriver::new(BufReader::new(stdin()), writer)
            .run()
            .map_err(|e| format!("usi driver i/o error: {e}"));
    }
    let mut it = args.iter().map(String::as_str);
    let cmd = it.next().ok_or("missing subcommand")?;
    match cmd {
        "perft" => perft_cmd(&it.collect::<Vec<_>>()),
        other => Err(format!("unknown subcommand `{other}`")),
    }
}

fn perft_cmd(args: &[&str]) -> Result<(), String> {
    let (mut pos, depth) = match args {
        ["startpos", depth] => (Position::startpos(), parse_depth(depth)?),
        ["sfen", sfen, depth] => {
            let pos = parse_sfen(sfen).map_err(|e| format!("invalid sfen: {e:?}"))?;
            (pos, parse_depth(depth)?)
        }
        ["sfen", sfen, "moves", rest @ ..] => {
            let (depth_str, moves) = rest
                .split_last()
                .ok_or("perft sfen … moves expects at least one move and a depth")?;
            let depth = parse_depth(depth_str)?;
            let mut pos = parse_sfen(sfen).map_err(|e| format!("invalid sfen: {e:?}"))?;
            for m in moves {
                let parsed =
                    parse_usi_move(m, &pos).map_err(|e| format!("invalid usi move {m:?}: {e}"))?;
                pos.do_move(parsed);
            }
            (pos, depth)
        }
        _ => {
            return Err("perft expects `startpos <depth>`, `sfen <SFEN> <depth>`, or `sfen <SFEN> moves <m1> ... <depth>`".into());
        }
    };
    let nodes = perft::perft(&mut pos, depth);
    println!("{nodes}");
    Ok(())
}

fn parse_depth(s: &str) -> Result<u32, String> {
    s.parse::<u32>()
        .map_err(|_| format!("depth must be a non-negative integer, got `{s}`"))
}
