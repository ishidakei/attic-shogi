//! Aggregate nodes-per-second at `Threads=1` and `Threads=2` over three
//! positions. This prints a table and never asserts a scaling factor: on a host
//! whose two logical CPUs are SMT siblings a ratio well under 2× is normal, so
//! a threshold would encode the machine rather than the engine.
//!
//! `#[ignore]`-gated, since it spends about 90 s, and it needs the real network,
//! staged locally and never committed; when that is absent it prints a notice
//! and passes.
//!
//! Run it in a release build:
//!
//! ```text
//! cargo test --release -p attic --test nps_scaling -- --ignored --nocapture
//! ```

use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

const MOVETIME_MS: u64 = 5000;
const RUNS: usize = 3;

/// `(label, SFEN)`: startpos plus two positions from the depth-5 fixtures.
const POSITIONS: &[(&str, &str)] = &[
    (
        "startpos",
        "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1",
    ),
    (
        "mid-game-tactical",
        "l7l/1r1sg2k1/2nppgsp1/p1p3p1p/1p2N4/2P1P1P2/PPSP1PB1P/3GG1SR1/LN2K3L b BNPp 1",
    ),
    ("check-evasion", "4k4/9/4r4/9/9/9/4K3B/9/9 b RG2gs2n3p 1"),
];

fn eval_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../eval")
}

fn send(stdin: &mut ChildStdin, cmd: &str) {
    stdin.write_all(cmd.as_bytes()).expect("write engine stdin");
    stdin.write_all(b"\n").expect("write newline");
    stdin.flush().expect("flush engine stdin");
}

fn read_until<F: Fn(&str) -> bool>(reader: &mut BufReader<ChildStdout>, pred: F) -> Option<String> {
    let mut line = String::new();
    loop {
        line.clear();
        let n = reader.read_line(&mut line).expect("read engine stdout");
        if n == 0 {
            return None;
        }
        let trimmed = line.trim_end_matches(['\r', '\n']);
        if pred(trimmed) {
            return Some(trimmed.to_string());
        }
    }
}

fn parse_nodes(info: &str) -> Option<u64> {
    let mut toks = info.split_whitespace();
    while let Some(t) = toks.next() {
        if t == "nodes" {
            return toks.next().and_then(|n| n.parse::<u64>().ok());
        }
    }
    None
}

/// Run `go movetime 5000` once for `sfen` and return the aggregated node count.
fn measure_nodes(stdin: &mut ChildStdin, reader: &mut BufReader<ChildStdout>, sfen: &str) -> u64 {
    send(stdin, "usinewgame");
    send(stdin, "isready");
    read_until(reader, |l| l == "readyok").expect("readyok");
    send(stdin, &format!("position sfen {sfen}"));
    send(stdin, &format!("go movetime {MOVETIME_MS}"));

    let mut nodes = 0u64;
    let mut line = String::new();
    loop {
        line.clear();
        let n = reader.read_line(&mut line).expect("read engine stdout");
        assert!(n != 0, "engine closed stdout mid-search");
        let trimmed = line.trim_end_matches(['\r', '\n']);
        if trimmed.starts_with("bestmove ") {
            break;
        }
        if trimmed.starts_with("info ")
            && let Some(v) = parse_nodes(trimmed)
        {
            nodes = v;
        }
    }
    nodes
}

fn median3(mut v: [u64; RUNS]) -> u64 {
    v.sort_unstable();
    v[RUNS / 2]
}

fn start_engine(threads: u32) -> (Child, ChildStdin, BufReader<ChildStdout>) {
    let dir = eval_dir();
    let exe = env!("CARGO_BIN_EXE_attic");
    let mut child: Child = Command::new(exe)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn engine");
    let mut stdin = child.stdin.take().expect("stdin piped");
    let mut reader = BufReader::new(child.stdout.take().expect("stdout piped"));

    send(&mut stdin, "usi");
    read_until(&mut reader, |l| l == "usiok").expect("usiok");
    let eval_dir_arg = dir.to_str().expect("utf-8 eval dir");
    send(
        &mut stdin,
        &format!("setoption name EvalDir value {eval_dir_arg}"),
    );
    send(
        &mut stdin,
        &format!("setoption name Threads value {threads}"),
    );
    send(&mut stdin, "isready");
    let ack = read_until(&mut reader, |l| {
        l == "readyok" || l.starts_with("info string eval load failed")
    })
    .expect("readyok or load failure");
    assert_eq!(
        ack, "readyok",
        "real network must load at Threads={threads}"
    );
    (child, stdin, reader)
}

#[test]
#[ignore = "spawns the engine and searches about 90 s; run explicitly"]
fn nps_threads1_vs_threads2() {
    let dir = eval_dir();
    if !dir.join("nn.bin").exists() {
        eprintln!(
            "skipping nps_threads1_vs_threads2: {} is not present (obtained out-of-band)",
            dir.join("nn.bin").display()
        );
        return;
    }

    let mut medians: Vec<(&str, f64, f64)> = Vec::new();

    for &(label, sfen) in POSITIONS {
        let nps_for = |threads: u32| -> f64 {
            let (mut child, mut stdin, mut reader) = start_engine(threads);
            let mut runs = [0u64; RUNS];
            for r in runs.iter_mut() {
                *r = measure_nodes(&mut stdin, &mut reader, sfen);
            }
            send(&mut stdin, "quit");
            drop(stdin);
            let _ = child.wait();
            median3(runs) as f64 / (MOVETIME_MS as f64 / 1000.0)
        };
        let nps1 = nps_for(1);
        let nps2 = nps_for(2);
        medians.push((label, nps1, nps2));
    }

    eprintln!("\nNPS scaling (median of {RUNS} runs, go movetime {MOVETIME_MS}):");
    eprintln!(
        "{:<20} {:>14} {:>14} {:>8}",
        "position", "Threads=1 NPS", "Threads=2 NPS", "ratio"
    );
    for (label, nps1, nps2) in &medians {
        let ratio = if *nps1 > 0.0 { nps2 / nps1 } else { 0.0 };
        eprintln!("{label:<20} {nps1:>14.0} {nps2:>14.0} {ratio:>8.2}");
    }

    // Measurement only: never a scaling factor.
    assert!(
        medians.iter().all(|(_, n1, n2)| *n1 > 0.0 && *n2 > 0.0),
        "every measurement must produce a positive node count"
    );
}
