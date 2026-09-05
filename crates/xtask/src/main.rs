use std::fs;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, Command as ProcessCommand, ExitCode, Stdio};

use clap::{Parser, Subcommand, ValueEnum};

use attic_state::packed_sfen::sfen_pack;
use attic_state::{Position, parse_sfen, parse_usi_move};

#[derive(Parser, Debug)]
#[command(
    name = "xtask",
    about = "Repo-integrated automation for the workspace.",
    long_about = "Run repo-integrated automation as `cargo xtask <subcommand>`. \
                  Subcommands wrap build orchestration, fixture capture, and other \
                  developer tooling so they live in the project's primary language."
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Print a greeting, proving the xtask wiring works end-to-end.
    Hello,
    /// Build the YaneuraOu reference binary used as parity ground truth.
    BuildReference(BuildReferenceArgs),
    /// Capture perft node counts from the reference binary into a JSON fixture.
    CapturePerft(CapturePerftArgs),
    /// Capture the static NNUE evaluation for a position into a JSON fixture.
    CaptureEval(CaptureEvalArgs),
    /// Capture a fixed-depth search result from the reference binary into a JSON fixture.
    CaptureSearch(CaptureSearchArgs),
    /// Convert a hand-authored `.db` text book into a `.ybb` binary book plus an
    /// expected-results JSON, using the workspace PackedSfen encoder.
    CaptureBook(CaptureBookArgs),
}

#[derive(clap::Args, Debug)]
struct BuildReferenceArgs {
    /// Upstream Makefile target. `tournament` is the production parity binary;
    /// `evallearn` is the variant required by `capture-perft` (registers the
    /// `SkipLoadingEval` USI option, which the `tournament` build strips).
    #[arg(long, value_enum, default_value_t = MakeTarget::Tournament)]
    make_target: MakeTarget,
    /// CPU target passed to the upstream Makefile.
    #[arg(long, default_value = "AVX2")]
    target_cpu: String,
    /// C++ compiler the upstream Makefile invokes.
    #[arg(long, default_value = "clang++")]
    compiler: String,
    /// Python interpreter used by the upstream NNUE arch generator.
    #[arg(long, default_value = "python3")]
    python: String,
    /// Skip `make clean`. The default rebuilds from a clean tree: the upstream
    /// targets reuse object files compiled under different CPPFLAGS, which
    /// silently mixes configurations.
    #[arg(long)]
    skip_clean: bool,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
enum MakeTarget {
    Tournament,
    Evallearn,
    Normal,
}

impl MakeTarget {
    fn as_str(self) -> &'static str {
        match self {
            MakeTarget::Tournament => "tournament",
            MakeTarget::Evallearn => "evallearn",
            MakeTarget::Normal => "normal",
        }
    }
}

#[derive(clap::Args, Debug)]
struct CapturePerftArgs {
    /// Path to the reference binary. Build with `cargo xtask build-reference`.
    #[arg(long, default_value = REFERENCE_BINARY_DEFAULT)]
    binary: PathBuf,
    /// Path to the evaluation network file the reference loads on `isready`.
    /// It is obtained out-of-band and never committed, and the engine refuses to
    /// proceed past `isready` without it.
    #[arg(long, default_value = EVAL_FILE_DEFAULT)]
    eval_file: PathBuf,
    #[arg(long, default_value_t = 5)]
    max_depth: u32,
    /// SFEN of the position from which to capture perft.
    #[arg(long, default_value = STARTPOS_SFEN)]
    sfen: String,
    /// Optional space-separated USI moves to play after `position sfen <SFEN>`
    /// before each `go perft <D>`.
    #[arg(long, default_value = "")]
    moves: String,
    /// Where to write the JSON fixture; a relative path resolves against the
    /// workspace root.
    #[arg(long, default_value = "tests/fixtures/perft/startpos.json")]
    fixture: PathBuf,
}

#[derive(clap::Args, Debug)]
struct CaptureEvalArgs {
    /// Path to the reference binary, built at the default `tournament` target.
    #[arg(long, default_value = REFERENCE_BINARY_DEFAULT)]
    binary: PathBuf,
    /// Path to the evaluation network file the reference loads on `isready`.
    #[arg(long, default_value = EVAL_FILE_DEFAULT)]
    eval_file: PathBuf,
    #[arg(long, default_value = STARTPOS_SFEN)]
    sfen: String,
    /// Optional space-separated USI moves to play after `position sfen <SFEN>`
    /// before sending the static-eval command.
    #[arg(long, default_value = "")]
    moves: String,
    /// Where to write the JSON fixture; a relative path resolves against the
    /// workspace root.
    #[arg(long, default_value = "tests/fixtures/eval/startpos.json")]
    fixture: PathBuf,
}

#[derive(clap::Args, Debug)]
struct CaptureSearchArgs {
    /// Path to the reference binary, built at the default `tournament` target.
    #[arg(long, default_value = REFERENCE_BINARY_DEFAULT)]
    binary: PathBuf,
    /// Path to the evaluation network file the reference loads on `isready`.
    #[arg(long, default_value = EVAL_FILE_DEFAULT)]
    eval_file: PathBuf,
    #[arg(long, default_value = STARTPOS_SFEN)]
    sfen: String,
    /// Optional space-separated USI moves to play after `position sfen <SFEN>`
    /// before sending `go depth`.
    #[arg(long, default_value = "")]
    moves: String,
    /// Fixed search depth. A single-thread fixed-depth search is reproducible
    /// byte for byte against the same reference build and network file.
    #[arg(long, default_value_t = 3)]
    depth: u32,
    /// Number of search threads. Must be 1 for deterministic fixture capture.
    #[arg(long, default_value_t = 1)]
    threads: u32,
    /// `USI_Hash` in MiB; omit to leave the engine default in place. A given
    /// value is also recorded in the fixture, since the table size changes node
    /// counts and a `nodes` comparison is only meaningful against the same size.
    #[arg(long)]
    hash: Option<u32>,
    /// Where to write the JSON fixture; a relative path resolves against the
    /// workspace root.
    #[arg(long, default_value = "tests/fixtures/search/startpos.json")]
    fixture: PathBuf,
}

#[derive(clap::Args, Debug)]
struct CaptureBookArgs {
    /// Hand-authored `.db` text book to convert.
    #[arg(long, default_value = "tests/fixtures/book/book.db")]
    db: PathBuf,
    /// Where to write the `.ybb` binary book.
    #[arg(long, default_value = "tests/fixtures/book/sample.ybb")]
    ybb: PathBuf,
    /// Where to write the expected-results JSON.
    #[arg(long, default_value = "tests/fixtures/book/expected.json")]
    expected: PathBuf,
    /// Emit a per-move `depth` field, setting `.ybb` flags bit 0 for 6-byte
    /// records.
    #[arg(long, default_value_t = true)]
    with_depth: bool,
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    match cli.command {
        Command::Hello => {
            println!("hello from xtask");
            ExitCode::SUCCESS
        }
        Command::BuildReference(args) => match build_reference(&args) {
            Ok(path) => {
                println!("reference binary: {}", path.display());
                ExitCode::SUCCESS
            }
            Err(e) => {
                eprintln!("xtask build-reference: {e}");
                ExitCode::FAILURE
            }
        },
        Command::CapturePerft(args) => match capture_perft(&args) {
            Ok(path) => {
                println!("perft fixture: {}", path.display());
                ExitCode::SUCCESS
            }
            Err(e) => {
                eprintln!("xtask capture-perft: {e}");
                ExitCode::FAILURE
            }
        },
        Command::CaptureEval(args) => match capture_eval(&args) {
            Ok(path) => {
                println!("eval fixture: {}", path.display());
                ExitCode::SUCCESS
            }
            Err(e) => {
                eprintln!("xtask capture-eval: {e}");
                ExitCode::FAILURE
            }
        },
        Command::CaptureSearch(args) => match capture_search(&args) {
            Ok(path) => {
                println!("search fixture: {}", path.display());
                ExitCode::SUCCESS
            }
            Err(e) => {
                eprintln!("xtask capture-search: {e}");
                ExitCode::FAILURE
            }
        },
        Command::CaptureBook(args) => match capture_book(&args) {
            Ok((ybb, expected)) => {
                println!("ybb book: {}", ybb.display());
                println!("expected results: {}", expected.display());
                ExitCode::SUCCESS
            }
            Err(e) => {
                eprintln!("xtask capture-book: {e}");
                ExitCode::FAILURE
            }
        },
    }
}

const YANEURAOU_EDITION: &str = "YANEURAOU_ENGINE_SFNN1536";
const REFERENCE_SOURCE_DIR: &str = "upstream YaneuraOu/source";
const REFERENCE_BINARY_NAME: &str = "YaneuraOu-by-gcc";
const REFERENCE_BINARY_DEFAULT: &str = "source/YaneuraOu-by-gcc";
const EVAL_FILE_DEFAULT: &str = "eval/nn.bin";
const STARTPOS_SFEN: &str = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1";

fn build_reference(args: &BuildReferenceArgs) -> Result<PathBuf, String> {
    let workspace_root = workspace_root()?;
    let source_dir = workspace_root.join(REFERENCE_SOURCE_DIR);
    if !source_dir.join("Makefile").is_file() {
        return Err(format!(
            "no Makefile under {}: the YaneuraOu sources this subcommand builds are not present there",
            source_dir.display()
        ));
    }

    if !args.skip_clean {
        run_make(&source_dir, args, &["clean"])?;
    }
    run_make(&source_dir, args, &[args.make_target.as_str()])?;

    let binary = source_dir.join(REFERENCE_BINARY_NAME);
    if !binary.is_file() {
        return Err(format!(
            "expected reference binary at {} but it was not produced",
            binary.display()
        ));
    }
    Ok(binary)
}

fn run_make(source_dir: &Path, args: &BuildReferenceArgs, extra: &[&str]) -> Result<(), String> {
    let mut cmd = ProcessCommand::new("make");
    cmd.current_dir(source_dir)
        .arg(format!("YANEURAOU_EDITION={YANEURAOU_EDITION}"))
        .arg(format!("TARGET_CPU={}", args.target_cpu))
        .arg(format!("COMPILER={}", args.compiler))
        .arg(format!("PYTHON={}", args.python))
        .args(extra)
        .stdin(Stdio::null());

    let status = cmd
        .status()
        .map_err(|e| format!("failed to spawn `make {}`: {e}", extra.join(" ")))?;
    if !status.success() {
        return Err(format!(
            "`make {}` exited with status {}",
            extra.join(" "),
            status
        ));
    }
    Ok(())
}

fn workspace_root() -> Result<PathBuf, String> {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    Path::new(manifest_dir)
        .parent()
        .and_then(Path::parent)
        .map(Path::to_path_buf)
        .ok_or_else(|| {
            format!("could not derive workspace root from CARGO_MANIFEST_DIR ({manifest_dir})")
        })
}

fn capture_perft(args: &CapturePerftArgs) -> Result<PathBuf, String> {
    if args.max_depth == 0 {
        return Err("max-depth must be at least 1".to_string());
    }

    let workspace_root = workspace_root()?;
    let binary = resolve_under(&workspace_root, &args.binary);
    if !binary.is_file() {
        return Err(format!(
            "reference binary not found at {}; build it with \
             `cargo xtask build-reference`",
            binary.display()
        ));
    }
    let eval_file = resolve_under(&workspace_root, &args.eval_file);
    if !eval_file.is_file() {
        return Err(format!(
            "evaluation network not found at {}; the evaluation network file is \
             obtained out-of-band and never committed — place a \
             YaneuraOu-compatible nn.bin at that path before running this command",
            eval_file.display()
        ));
    }

    let fixture_path = resolve_under(&workspace_root, &args.fixture);
    let depths: Vec<u32> = (1..=args.max_depth).collect();
    let counts = drive_perft(&binary, &args.sfen, &args.moves, &depths)?;

    let json = render_fixture(&args.sfen, &args.moves, &depths, &counts);
    if let Some(parent) = fixture_path.parent() {
        fs::create_dir_all(parent)
            .map_err(|e| format!("could not create {}: {e}", parent.display()))?;
    }
    fs::write(&fixture_path, &json)
        .map_err(|e| format!("could not write {}: {e}", fixture_path.display()))?;
    Ok(fixture_path)
}

fn resolve_under(root: &Path, p: &Path) -> PathBuf {
    if p.is_absolute() {
        p.to_path_buf()
    } else {
        root.join(p)
    }
}

fn drive_perft(binary: &Path, sfen: &str, moves: &str, depths: &[u32]) -> Result<Vec<u64>, String> {
    let mut child = ProcessCommand::new(binary)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("failed to spawn {}: {e}", binary.display()))?;

    let mut stdin = child
        .stdin
        .take()
        .ok_or_else(|| "failed to capture engine stdin".to_string())?;
    write_perft_script(&mut stdin, sfen, moves, depths)
        .map_err(|e| format!("failed to drive engine stdin: {e}"))?;
    drop(stdin);

    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| "failed to capture engine stdout".to_string())?;
    let counts = match parse_perft_output(BufReader::new(stdout), depths.len()) {
        Ok(counts) => counts,
        Err(parse_err) => {
            let stderr = read_stderr(&mut child);
            wait_child(&mut child);
            return Err(combine_engine_error(&parse_err, &stderr));
        }
    };

    let status = child
        .wait()
        .map_err(|e| format!("failed to wait on engine process: {e}"))?;
    if !status.success() {
        return Err(format!("engine process exited with {status}"));
    }
    Ok(counts)
}

fn write_perft_script<W: Write>(
    out: &mut W,
    sfen: &str,
    moves: &str,
    depths: &[u32],
) -> std::io::Result<()> {
    writeln!(out, "usi")?;
    writeln!(out, "isready")?;
    let trimmed_moves = moves.trim();
    if trimmed_moves.is_empty() {
        writeln!(out, "position sfen {sfen}")?;
    } else {
        writeln!(out, "position sfen {sfen} moves {trimmed_moves}")?;
    }
    for d in depths {
        writeln!(out, "go perft {d}")?;
    }
    writeln!(out, "quit")?;
    Ok(())
}

fn parse_perft_output<R: BufRead>(reader: R, expected: usize) -> Result<Vec<u64>, String> {
    let mut nodes = Vec::with_capacity(expected);
    for line in reader.lines() {
        let line = line.map_err(|e| format!("error reading engine stdout: {e}"))?;
        if let Some(rest) = line.strip_prefix("Nodes searched:") {
            let n: u64 = rest
                .trim()
                .parse()
                .map_err(|e| format!("could not parse `{line}`: {e}"))?;
            nodes.push(n);
        }
    }
    if nodes.len() != expected {
        return Err(format!(
            "engine produced {} `Nodes searched:` line(s); expected {}",
            nodes.len(),
            expected
        ));
    }
    Ok(nodes)
}

fn read_stderr(child: &mut Child) -> String {
    use std::io::Read;
    let mut buf = String::new();
    if let Some(mut s) = child.stderr.take() {
        let _ = s.read_to_string(&mut buf);
    }
    buf
}

fn wait_child(child: &mut Child) {
    let _ = child.wait();
}

fn combine_engine_error(parse_err: &str, stderr: &str) -> String {
    let trimmed = stderr.trim();
    if trimmed.is_empty() {
        parse_err.to_string()
    } else {
        format!("{parse_err}\nengine stderr:\n{trimmed}")
    }
}

fn render_fixture(sfen: &str, moves: &str, depths: &[u32], counts: &[u64]) -> String {
    debug_assert_eq!(depths.len(), counts.len());
    let mut out = String::new();
    out.push_str("{\n");
    out.push_str(&format!("  \"sfen\": \"{sfen}\",\n"));
    let trimmed_moves = moves.trim();
    if !trimmed_moves.is_empty() {
        out.push_str("  \"moves\": [");
        for (i, m) in trimmed_moves.split_whitespace().enumerate() {
            if i > 0 {
                out.push_str(", ");
            }
            out.push_str(&format!("\"{m}\""));
        }
        out.push_str("],\n");
    }
    out.push_str("  \"results\": [\n");
    for (i, (d, n)) in depths.iter().zip(counts.iter()).enumerate() {
        let comma = if i + 1 == depths.len() { "" } else { "," };
        out.push_str(&format!(
            "    {{ \"depth\": {d}, \"expected_nodes\": {n} }}{comma}\n"
        ));
    }
    out.push_str("  ]\n");
    out.push_str("}\n");
    out
}

fn capture_eval(args: &CaptureEvalArgs) -> Result<PathBuf, String> {
    let workspace_root = workspace_root()?;
    let binary = resolve_under(&workspace_root, &args.binary);
    if !binary.is_file() {
        return Err(format!(
            "reference binary not found at {}; build it with \
             `cargo xtask build-reference`",
            binary.display()
        ));
    }
    let eval_file = resolve_under(&workspace_root, &args.eval_file);
    if !eval_file.is_file() {
        return Err(format!(
            "evaluation network not found at {}; the evaluation network file is \
             obtained out-of-band and never committed — place a \
             YaneuraOu-compatible nn.bin at that path before running this command",
            eval_file.display()
        ));
    }

    let fixture_path = resolve_under(&workspace_root, &args.fixture);
    let value = drive_eval(&binary, &args.sfen, &args.moves)?;

    let json = render_eval_fixture(&args.sfen, &args.moves, value);
    if let Some(parent) = fixture_path.parent() {
        fs::create_dir_all(parent)
            .map_err(|e| format!("could not create {}: {e}", parent.display()))?;
    }
    fs::write(&fixture_path, &json)
        .map_err(|e| format!("could not write {}: {e}", fixture_path.display()))?;
    Ok(fixture_path)
}

fn drive_eval(binary: &Path, sfen: &str, moves: &str) -> Result<i32, String> {
    let mut child = ProcessCommand::new(binary)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("failed to spawn {}: {e}", binary.display()))?;

    let mut stdin = child
        .stdin
        .take()
        .ok_or_else(|| "failed to capture engine stdin".to_string())?;
    write_eval_script(&mut stdin, sfen, moves)
        .map_err(|e| format!("failed to drive engine stdin: {e}"))?;
    drop(stdin);

    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| "failed to capture engine stdout".to_string())?;
    let value = match parse_eval_output(BufReader::new(stdout)) {
        Ok(v) => v,
        Err(parse_err) => {
            let stderr = read_stderr(&mut child);
            wait_child(&mut child);
            return Err(combine_engine_error(&parse_err, &stderr));
        }
    };

    let status = child
        .wait()
        .map_err(|e| format!("failed to wait on engine process: {e}"))?;
    if !status.success() {
        return Err(format!("engine process exited with {status}"));
    }
    Ok(value)
}

fn write_eval_script<W: Write>(out: &mut W, sfen: &str, moves: &str) -> std::io::Result<()> {
    writeln!(out, "usi")?;
    writeln!(out, "isready")?;
    let trimmed_moves = moves.trim();
    if trimmed_moves.is_empty() {
        writeln!(out, "position sfen {sfen}")?;
    } else {
        writeln!(out, "position sfen {sfen} moves {trimmed_moves}")?;
    }
    // The `e` command calls `engine.evaluate()` and prints a single
    // `eval = <integer>` line. `eval` is a different, unrelated stub.
    writeln!(out, "e")?;
    writeln!(out, "quit")?;
    Ok(())
}

fn parse_eval_output<R: BufRead>(reader: R) -> Result<i32, String> {
    for line in reader.lines() {
        let line = line.map_err(|e| format!("error reading engine stdout: {e}"))?;
        if let Some(rest) = line.strip_prefix("eval = ") {
            let v: i32 = rest
                .trim()
                .parse()
                .map_err(|e| format!("could not parse `{line}`: {e}"))?;
            return Ok(v);
        }
    }
    Err("engine produced no `eval = ` line".to_string())
}

fn render_eval_fixture(sfen: &str, moves: &str, eval: i32) -> String {
    let mut out = String::new();
    out.push_str("{\n");
    out.push_str(&format!("  \"sfen\": \"{sfen}\",\n"));
    let trimmed_moves = moves.trim();
    if !trimmed_moves.is_empty() {
        out.push_str("  \"moves\": [");
        for (i, m) in trimmed_moves.split_whitespace().enumerate() {
            if i > 0 {
                out.push_str(", ");
            }
            out.push_str(&format!("\"{m}\""));
        }
        out.push_str("],\n");
    }
    out.push_str(&format!("  \"eval\": {eval}\n"));
    out.push_str("}\n");
    out
}

#[derive(Debug, Clone)]
enum SearchScore {
    Cp(i32),
    Mate(i32),
}

#[derive(Debug, Clone)]
struct SearchResult {
    depth: u32,
    bestmove: String,
    score: SearchScore,
    nodes: u64,
    pv: Vec<String>,
    /// The `FV_SCALE` the reference actually searched with, when it announced an
    /// override. `eval/eval_options.txt` is a local, uncommitted knob, so the
    /// engine's own `info string` announcement is the only reliable source.
    /// `None` means the engine default was in force.
    fv_scale: Option<i32>,
}

struct ParsedInfo {
    depth: u32,
    score: SearchScore,
    nodes: u64,
    pv: Vec<String>,
}

fn capture_search(args: &CaptureSearchArgs) -> Result<PathBuf, String> {
    if args.depth == 0 {
        return Err("depth must be at least 1".to_string());
    }

    let workspace_root = workspace_root()?;
    let binary = resolve_under(&workspace_root, &args.binary);
    if !binary.is_file() {
        return Err(format!(
            "reference binary not found at {}; build it with \
             `cargo xtask build-reference`",
            binary.display()
        ));
    }
    let eval_file = resolve_under(&workspace_root, &args.eval_file);
    if !eval_file.is_file() {
        return Err(format!(
            "evaluation network not found at {}; the evaluation network file is \
             obtained out-of-band and never committed — place a \
             YaneuraOu-compatible nn.bin at that path before running this command",
            eval_file.display()
        ));
    }

    let fixture_path = resolve_under(&workspace_root, &args.fixture);
    let result = drive_search(
        &binary,
        &args.sfen,
        &args.moves,
        args.depth,
        args.threads,
        args.hash,
    )?;

    let json = render_search_fixture(&args.sfen, &args.moves, args.hash, &result);
    if let Some(parent) = fixture_path.parent() {
        fs::create_dir_all(parent)
            .map_err(|e| format!("could not create {}: {e}", parent.display()))?;
    }
    fs::write(&fixture_path, &json)
        .map_err(|e| format!("could not write {}: {e}", fixture_path.display()))?;
    Ok(fixture_path)
}

fn drive_search(
    binary: &Path,
    sfen: &str,
    moves: &str,
    depth: u32,
    threads: u32,
    hash: Option<u32>,
) -> Result<SearchResult, String> {
    let mut child = ProcessCommand::new(binary)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("failed to spawn {}: {e}", binary.display()))?;

    let mut stdin = child
        .stdin
        .take()
        .ok_or_else(|| "failed to capture engine stdin".to_string())?;

    // stdin must stay open until bestmove is read: `go depth N` searches
    // asynchronously, and an EOF reaches the engine's command loop as a
    // synthetic "quit" (misc.cpp) that aborts the search after the
    // first completed depth.
    write_search_script(&mut stdin, sfen, moves, depth, threads, hash)
        .map_err(|e| format!("failed to drive engine stdin: {e}"))?;

    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| "failed to capture engine stdout".to_string())?;

    let result = match parse_search_output(BufReader::new(stdout)) {
        Ok(r) => r,
        Err(parse_err) => {
            let stderr = read_stderr(&mut child);
            drop(stdin);
            wait_child(&mut child);
            return Err(combine_engine_error(&parse_err, &stderr));
        }
    };

    // The EOF this sends is the engine's "quit".
    drop(stdin);

    let status = child
        .wait()
        .map_err(|e| format!("failed to wait on engine process: {e}"))?;
    if !status.success() {
        return Err(format!("engine process exited with {status}"));
    }
    Ok(result)
}

fn write_search_script<W: Write>(
    out: &mut W,
    sfen: &str,
    moves: &str,
    depth: u32,
    threads: u32,
    hash: Option<u32>,
) -> std::io::Result<()> {
    writeln!(out, "usi")?;
    writeln!(out, "setoption name Threads value {threads}")?;
    // `no_book` is the sentinel recognised by book.cpp; without it a book
    // hit would replace the search.
    writeln!(out, "setoption name BookFile value no_book")?;
    if let Some(mb) = hash {
        writeln!(out, "setoption name USI_Hash value {mb}")?;
    }
    writeln!(out, "isready")?;
    // `usinewgame` clears the transposition table, so the search always starts
    // from a clean hash state.
    writeln!(out, "usinewgame")?;
    let trimmed_moves = moves.trim();
    if trimmed_moves.is_empty() {
        writeln!(out, "position sfen {sfen}")?;
    } else {
        writeln!(out, "position sfen {sfen} moves {trimmed_moves}")?;
    }
    writeln!(out, "go depth {depth}")?;
    Ok(())
}

fn parse_info_line(line: &str) -> Option<ParsedInfo> {
    let mut tokens = line.split_ascii_whitespace();
    if tokens.next()? != "info" {
        return None;
    }

    let mut depth: Option<u32> = None;
    let mut score: Option<SearchScore> = None;
    let mut nodes: Option<u64> = None;
    let mut pv: Vec<String> = Vec::new();
    let mut is_bound = false;

    while let Some(tok) = tokens.next() {
        match tok {
            "depth" => {
                depth = tokens.next().and_then(|s| s.parse().ok());
            }
            "score" => {
                let kind = tokens.next()?;
                let val: i32 = tokens.next()?.parse().ok()?;
                score = Some(match kind {
                    "cp" => SearchScore::Cp(val),
                    "mate" => SearchScore::Mate(val),
                    _ => return None,
                });
            }
            "nodes" => {
                nodes = tokens.next().and_then(|s| s.parse().ok());
            }
            "pv" => {
                pv = tokens.map(|s| s.to_string()).collect();
                break;
            }
            "lowerbound" | "upperbound" => {
                is_bound = true;
            }
            _ => {}
        }
    }

    if is_bound {
        // Aspiration-window fail-high / fail-low lines are not exact scores.
        return None;
    }

    Some(ParsedInfo {
        depth: depth?,
        score: score?,
        nodes: nodes?,
        pv,
    })
}

/// Extract `N` from
/// `info string engine option override. name = FV_SCALE , value = N`, the line
/// the reference prints at `isready` for every override it applies.
fn parse_fv_scale_override(line: &str) -> Option<i32> {
    let rest = line.strip_prefix("info string engine option override.")?;
    let mut tokens = rest.split_ascii_whitespace();
    if tokens.next()? != "name" || tokens.next()? != "=" || tokens.next()? != "FV_SCALE" {
        return None;
    }
    if tokens.next()? != "," || tokens.next()? != "value" || tokens.next()? != "=" {
        return None;
    }
    tokens.next()?.parse().ok()
}

fn parse_search_output<R: BufRead>(reader: R) -> Result<SearchResult, String> {
    let mut last_info: Option<ParsedInfo> = None;
    let mut bestmove: Option<String> = None;
    let mut fv_scale: Option<i32> = None;

    for line in reader.lines() {
        let line = line.map_err(|e| format!("error reading engine stdout: {e}"))?;

        if let Some(scale) = parse_fv_scale_override(&line) {
            fv_scale = Some(scale);
            continue;
        }

        if let Some(rest) = line.strip_prefix("bestmove ") {
            let bm = rest
                .split_ascii_whitespace()
                .next()
                .ok_or_else(|| "bestmove line has no move token".to_string())?
                .to_string();
            bestmove = Some(bm);
            break;
        }

        if line.starts_with("info ")
            && let Some(info) = parse_info_line(&line)
        {
            last_info = Some(info);
        }
    }

    let bestmove = bestmove.ok_or_else(|| "engine produced no `bestmove` line".to_string())?;
    let info = last_info
        .ok_or_else(|| "engine produced no `info depth` line before bestmove".to_string())?;

    Ok(SearchResult {
        depth: info.depth,
        bestmove,
        score: info.score,
        nodes: info.nodes,
        pv: info.pv,
        fv_scale,
    })
}

fn render_search_fixture(
    sfen: &str,
    moves: &str,
    hash: Option<u32>,
    result: &SearchResult,
) -> String {
    let mut out = String::new();
    out.push_str("{\n");
    out.push_str(&format!("  \"sfen\": \"{sfen}\",\n"));
    let trimmed_moves = moves.trim();
    if !trimmed_moves.is_empty() {
        out.push_str("  \"moves\": [");
        for (i, m) in trimmed_moves.split_whitespace().enumerate() {
            if i > 0 {
                out.push_str(", ");
            }
            out.push_str(&format!("\"{m}\""));
        }
        out.push_str("],\n");
    }
    out.push_str(&format!("  \"depth\": {},\n", result.depth));
    // Emitted only for an explicit `--hash`.
    if let Some(mb) = hash {
        out.push_str(&format!("  \"hash_mb\": {mb},\n"));
    }
    // Emitted only when the reference announced an override. A `nodes` /
    // `score` comparison is meaningless unless the Rust side divides by the same
    // number, so record what the capture actually ran with.
    if let Some(scale) = result.fv_scale {
        out.push_str(&format!("  \"fv_scale\": {scale},\n"));
    }
    out.push_str(&format!("  \"bestmove\": \"{}\",\n", result.bestmove));
    match result.score {
        SearchScore::Cp(v) => {
            out.push_str(&format!("  \"score\": {{ \"cp\": {v} }},\n"));
        }
        SearchScore::Mate(v) => {
            out.push_str(&format!("  \"score\": {{ \"mate\": {v} }},\n"));
        }
    }
    out.push_str(&format!("  \"nodes\": {},\n", result.nodes));
    out.push_str("  \"pv\": [");
    for (i, m) in result.pv.iter().enumerate() {
        if i > 0 {
            out.push_str(", ");
        }
        out.push_str(&format!("\"{m}\""));
    }
    out.push_str("]\n");
    out.push_str("}\n");
    out
}

/// `.ybb` magic (`YbbMagic`, `book.cpp`).
const YBB_MAGIC: &[u8; 16] = b"YANE-BINBOOK-V1\0";
/// `.ybb` flags bit 0 — per-move depth present (`YbbFlagMoveDepth`).
const YBB_FLAG_MOVE_DEPTH: u64 = 1;

struct BookMoveSrc {
    usi: String,
    move16: u16,
    value: i16,
    depth: u16,
}

struct BookRecordSrc {
    sfen: String,
    packed: [u8; 32],
    ply: u16,
    moves: Vec<BookMoveSrc>,
}

fn capture_book(args: &CaptureBookArgs) -> Result<(PathBuf, PathBuf), String> {
    let workspace_root = workspace_root()?;
    let db_path = resolve_under(&workspace_root, &args.db);
    let text = fs::read_to_string(&db_path)
        .map_err(|e| format!("could not read {}: {e}", db_path.display()))?;

    let mut records = parse_db(&text)?;
    // The index is binary-searched by packed key, so it must be sorted ascending
    // under byte-wise order.
    records.sort_by_key(|r| r.packed);

    let ybb_bytes = serialize_ybb(&records, args.with_depth);
    let ybb_path = resolve_under(&workspace_root, &args.ybb);
    if let Some(parent) = ybb_path.parent() {
        fs::create_dir_all(parent)
            .map_err(|e| format!("could not create {}: {e}", parent.display()))?;
    }
    fs::write(&ybb_path, &ybb_bytes)
        .map_err(|e| format!("could not write {}: {e}", ybb_path.display()))?;

    let json = render_book_expected(&records, args.with_depth);
    let expected_path = resolve_under(&workspace_root, &args.expected);
    fs::write(&expected_path, &json)
        .map_err(|e| format!("could not write {}: {e}", expected_path.display()))?;

    Ok((ybb_path, expected_path))
}

fn parse_db(text: &str) -> Result<Vec<BookRecordSrc>, String> {
    struct Pending {
        sfen: String,
        pos: Position,
        moves: Vec<BookMoveSrc>,
    }

    let finalize = |p: Pending| BookRecordSrc {
        packed: sfen_pack(&p.pos),
        ply: p.pos.ply(),
        sfen: p.sfen,
        moves: p.moves,
    };

    let mut records = Vec::new();
    let mut current: Option<Pending> = None;

    for raw in text.lines() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') || line.starts_with("//") {
            continue;
        }
        if let Some(sfen) = line.strip_prefix("sfen ") {
            if let Some(prev) = current.take() {
                records.push(finalize(prev));
            }
            let sfen = sfen.trim().to_string();
            let pos = parse_sfen(&sfen).map_err(|e| format!("bad sfen {sfen:?}: {e}"))?;
            current = Some(Pending {
                sfen,
                pos,
                moves: Vec::new(),
            });
        } else {
            let pending = current
                .as_mut()
                .ok_or_else(|| format!("move line {line:?} before any `sfen` line"))?;
            let mut toks = line.split_whitespace();
            let best = toks.next().ok_or_else(|| "empty move line".to_string())?;
            let _ponder = toks.next(); // required slot in the .db grammar; ignored
            let value = parse_opt::<i32>(toks.next(), "value")?.unwrap_or(0);
            let depth = parse_opt::<u16>(toks.next(), "depth")?.unwrap_or(0);
            // The `.ybb` move record has no per-move count field.
            let mv = parse_usi_move(best, &pending.pos)
                .map_err(|e| format!("bad move {best:?} in {}: {e}", pending.sfen))?;
            pending.moves.push(BookMoveSrc {
                usi: best.to_string(),
                move16: mv.move16(),
                value: value as i16,
                depth,
            });
        }
    }
    if let Some(prev) = current.take() {
        records.push(finalize(prev));
    }
    Ok(records)
}

fn parse_opt<T: std::str::FromStr>(tok: Option<&str>, field: &str) -> Result<Option<T>, String>
where
    T::Err: std::fmt::Display,
{
    match tok {
        None => Ok(None),
        Some(s) => s
            .parse::<T>()
            .map(Some)
            .map_err(|e| format!("bad {field} token {s:?}: {e}")),
    }
}

fn serialize_ybb(records: &[BookRecordSrc], with_depth: bool) -> Vec<u8> {
    let flags: u64 = if with_depth { YBB_FLAG_MOVE_DEPTH } else { 0 };

    let mut index = Vec::new();
    let mut moves = Vec::new();
    for r in records {
        let moves_offset = moves.len() as u64;
        index.extend_from_slice(&r.packed);
        index.extend_from_slice(&moves_offset.to_le_bytes());
        index.extend_from_slice(&r.ply.to_le_bytes());
        index.extend_from_slice(&(r.moves.len() as u16).to_le_bytes());
        for m in &r.moves {
            moves.extend_from_slice(&m.move16.to_le_bytes());
            moves.extend_from_slice(&(m.value as u16).to_le_bytes());
            if with_depth {
                moves.extend_from_slice(&m.depth.to_le_bytes());
            }
        }
    }

    let mut out = Vec::with_capacity(32 + index.len() + moves.len());
    out.extend_from_slice(YBB_MAGIC);
    out.extend_from_slice(&(records.len() as u64).to_le_bytes());
    out.extend_from_slice(&flags.to_le_bytes());
    out.extend_from_slice(&index);
    out.extend_from_slice(&moves);
    out
}

fn render_book_expected(records: &[BookRecordSrc], with_depth: bool) -> String {
    let mut out = String::new();
    out.push_str("{\n");
    out.push_str(&format!("  \"with_depth\": {with_depth},\n"));
    out.push_str("  \"positions\": [\n");
    for (ri, r) in records.iter().enumerate() {
        out.push_str("    {\n");
        out.push_str(&format!("      \"sfen\": \"{}\",\n", r.sfen));
        out.push_str(&format!("      \"ply\": {},\n", r.ply));
        out.push_str("      \"moves\": [\n");
        for (mi, m) in r.moves.iter().enumerate() {
            let depth = if with_depth { m.depth } else { 0 };
            let comma = if mi + 1 == r.moves.len() { "" } else { "," };
            out.push_str(&format!(
                "        {{ \"usi\": \"{}\", \"move16\": {}, \"value\": {}, \"depth\": {}, \"count\": 0 }}{comma}\n",
                m.usi, m.move16, m.value, depth
            ));
        }
        out.push_str("      ]\n");
        let comma = if ri + 1 == records.len() { "" } else { "," };
        out.push_str(&format!("    }}{comma}\n"));
    }
    out.push_str("  ]\n");
    out.push_str("}\n");
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_nodes_searched_lines() {
        let stdout = "\
info string something\n\
7g7f: 1\n\
\n\
Nodes searched: 30\n\
info string next\n\
Nodes searched: 900\n\
";
        let counts = parse_perft_output(stdout.as_bytes(), 2).unwrap();
        assert_eq!(counts, vec![30, 900]);
    }

    #[test]
    fn errors_when_count_mismatches_expected() {
        let stdout = "Nodes searched: 30\n";
        let err = parse_perft_output(stdout.as_bytes(), 2).unwrap_err();
        assert!(err.contains("expected 2"), "got: {err}");
    }

    #[test]
    fn renders_canonical_fixture_for_known_counts() {
        let json = render_fixture(STARTPOS_SFEN, "", &[1, 2], &[30, 900]);
        let expected = "{\n  \"sfen\": \"lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1\",\n  \"results\": [\n    { \"depth\": 1, \"expected_nodes\": 30 },\n    { \"depth\": 2, \"expected_nodes\": 900 }\n  ]\n}\n";
        assert_eq!(json, expected);
    }

    #[test]
    fn renders_fixture_for_arbitrary_sfen_and_depth_set() {
        // Synthetic counts: this pins the serialisation, not the engine math.
        let sfen = "k8/1P7/G8/1N2P4/9/9/9/9/8K b 2PG2pg 1";
        let json = render_fixture(sfen, "", &[1, 3], &[42, 12345]);
        let expected = "{\n  \"sfen\": \"k8/1P7/G8/1N2P4/9/9/9/9/8K b 2PG2pg 1\",\n  \"results\": [\n    { \"depth\": 1, \"expected_nodes\": 42 },\n    { \"depth\": 3, \"expected_nodes\": 12345 }\n  ]\n}\n";
        assert_eq!(json, expected);
    }

    #[test]
    fn renders_fixture_with_moves_prefix() {
        let sfen = "9/4k4/9/9/9/9/9/4K4/9 b 9P9p 1";
        let moves = "5h4h 5b4b 4h5h 4b5b";
        let json = render_fixture(sfen, moves, &[1, 2], &[78, 5950]);
        let expected = "{\n  \"sfen\": \"9/4k4/9/9/9/9/9/4K4/9 b 9P9p 1\",\n  \"moves\": [\"5h4h\", \"5b4b\", \"4h5h\", \"4b5b\"],\n  \"results\": [\n    { \"depth\": 1, \"expected_nodes\": 78 },\n    { \"depth\": 2, \"expected_nodes\": 5950 }\n  ]\n}\n";
        assert_eq!(json, expected);
    }

    #[test]
    fn whitespace_only_moves_renders_no_moves_field() {
        // Whitespace-only `moves` must behave like the empty default, so a
        // stray space in `--moves` cannot change a regenerated fixture.
        let json = render_fixture(STARTPOS_SFEN, "   ", &[1], &[30]);
        let expected = "{\n  \"sfen\": \"lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1\",\n  \"results\": [\n    { \"depth\": 1, \"expected_nodes\": 30 }\n  ]\n}\n";
        assert_eq!(json, expected);
    }

    #[test]
    fn write_perft_script_emits_moves_when_non_empty() {
        let sfen = "9/4k4/9/9/9/9/9/4K4/9 b 9P9p 1";
        let moves = "5h4h 5b4b";
        let mut buf = Vec::new();
        write_perft_script(&mut buf, sfen, moves, &[1, 2]).unwrap();
        let expected = format!(
            "usi\nisready\nposition sfen {sfen} moves {moves}\ngo perft 1\ngo perft 2\nquit\n"
        );
        assert_eq!(String::from_utf8(buf).unwrap(), expected);
    }

    #[test]
    fn write_perft_script_omits_moves_when_empty() {
        let mut buf = Vec::new();
        write_perft_script(&mut buf, STARTPOS_SFEN, "", &[3]).unwrap();
        let expected = format!("usi\nisready\nposition sfen {STARTPOS_SFEN}\ngo perft 3\nquit\n");
        assert_eq!(String::from_utf8(buf).unwrap(), expected);
    }

    #[test]
    fn parses_eval_line_positive() {
        let stdout = "\
usiok\n\
readyok\n\
eval = 47\n\
";
        let v = parse_eval_output(stdout.as_bytes()).unwrap();
        assert_eq!(v, 47);
    }

    #[test]
    fn parses_eval_line_negative() {
        // The value actually observed for startpos.
        let stdout = "\
usiok\n\
readyok\n\
eval = -103\n\
";
        let v = parse_eval_output(stdout.as_bytes()).unwrap();
        assert_eq!(v, -103);
    }

    #[test]
    fn parse_eval_errors_when_no_eval_line() {
        let stdout = "usiok\nreadyok\n";
        let err = parse_eval_output(stdout.as_bytes()).unwrap_err();
        assert!(err.contains("no `eval = `"), "got: {err}");
    }

    #[test]
    fn renders_eval_fixture_no_moves() {
        let json = render_eval_fixture(STARTPOS_SFEN, "", -103);
        let expected = "{\n  \"sfen\": \"lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1\",\n  \"eval\": -103\n}\n";
        assert_eq!(json, expected);
    }

    #[test]
    fn renders_eval_fixture_with_moves() {
        let sfen = "9/4k4/9/9/9/9/9/4K4/9 b 9P9p 1";
        let moves = "5h4h 5b4b";
        let json = render_eval_fixture(sfen, moves, 120);
        let expected = "{\n  \"sfen\": \"9/4k4/9/9/9/9/9/4K4/9 b 9P9p 1\",\n  \"moves\": [\"5h4h\", \"5b4b\"],\n  \"eval\": 120\n}\n";
        assert_eq!(json, expected);
    }

    #[test]
    fn renders_eval_fixture_whitespace_moves_omitted() {
        let json = render_eval_fixture(STARTPOS_SFEN, "   ", 0);
        let expected = "{\n  \"sfen\": \"lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1\",\n  \"eval\": 0\n}\n";
        assert_eq!(json, expected);
    }

    #[test]
    fn write_eval_script_emits_e_command() {
        let mut buf = Vec::new();
        write_eval_script(&mut buf, STARTPOS_SFEN, "").unwrap();
        let expected = format!("usi\nisready\nposition sfen {STARTPOS_SFEN}\ne\nquit\n");
        assert_eq!(String::from_utf8(buf).unwrap(), expected);
    }

    #[test]
    fn write_eval_script_emits_moves_when_non_empty() {
        let sfen = "9/4k4/9/9/9/9/9/4K4/9 b 9P9p 1";
        let moves = "5h4h 5b4b";
        let mut buf = Vec::new();
        write_eval_script(&mut buf, sfen, moves).unwrap();
        let expected = format!("usi\nisready\nposition sfen {sfen} moves {moves}\ne\nquit\n");
        assert_eq!(String::from_utf8(buf).unwrap(), expected);
    }

    #[test]
    fn parses_search_info_cp_score() {
        // The exact line shape observed from the reference engine.
        let stdout = "\
usiok\n\
readyok\n\
info depth 1 multipv 1 score cp 0 nodes 0 nps 0 hashfull 0 time 1 pv 1g1f\n\
bestmove 1g1f\n\
";
        let result = parse_search_output(stdout.as_bytes()).unwrap();
        assert_eq!(result.depth, 1);
        assert_eq!(result.bestmove, "1g1f");
        assert!(
            matches!(result.score, SearchScore::Cp(0)),
            "expected Cp(0), got {:?}",
            result.score
        );
        assert_eq!(result.nodes, 0);
        assert_eq!(result.pv, vec!["1g1f"]);
    }

    #[test]
    fn parses_search_info_mate_score() {
        // The reference emits `"mate " + m` with no division by 2, positive for
        // a mate for the side to move.
        let stdout = "\
readyok\n\
info depth 5 seldepth 5 multipv 1 score mate 3 nodes 42 nps 8400 hashfull 0 time 5 pv 2g2f 8b2b 2h2b+\n\
bestmove 2g2f\n\
";
        let result = parse_search_output(stdout.as_bytes()).unwrap();
        assert_eq!(result.depth, 5);
        assert_eq!(result.bestmove, "2g2f");
        assert!(
            matches!(result.score, SearchScore::Mate(3)),
            "expected Mate(3), got {:?}",
            result.score
        );
        assert_eq!(result.nodes, 42);
        assert_eq!(result.pv, vec!["2g2f", "8b2b", "2h2b+"]);
    }

    #[test]
    fn parse_search_skips_lowerbound_lines() {
        let stdout = "\
info depth 3 multipv 1 score cp 150 lowerbound nodes 1000 nps 50000 hashfull 0 time 20 pv 7g7f\n\
info depth 3 multipv 1 score cp 80 nodes 1200 nps 60000 hashfull 0 time 20 pv 2g2f\n\
bestmove 2g2f\n\
";
        let result = parse_search_output(stdout.as_bytes()).unwrap();
        assert!(
            matches!(result.score, SearchScore::Cp(80)),
            "expected Cp(80), got {:?}",
            result.score
        );
        assert_eq!(result.bestmove, "2g2f");
    }

    #[test]
    fn parse_search_errors_when_no_bestmove() {
        let stdout =
            "readyok\ninfo depth 1 multipv 1 score cp 0 nodes 0 nps 0 hashfull 0 time 1 pv 1g1f\n";
        let err = parse_search_output(stdout.as_bytes()).unwrap_err();
        assert!(err.contains("no `bestmove`"), "got: {err}");
    }

    #[test]
    fn parse_search_errors_when_no_info_line() {
        let stdout = "readyok\nbestmove 1g1f\n";
        let err = parse_search_output(stdout.as_bytes()).unwrap_err();
        assert!(err.contains("no `info depth`"), "got: {err}");
    }

    #[test]
    fn write_search_script_no_moves() {
        // The script omits `quit`: the caller closes stdin after reading
        // bestmove, which the engine reads as one (misc.cpp).
        let mut buf = Vec::new();
        write_search_script(&mut buf, STARTPOS_SFEN, "", 3, 1, None).unwrap();
        let expected = format!(
            "usi\nsetoption name Threads value 1\nsetoption name BookFile value no_book\n\
             isready\nusinewgame\nposition sfen {STARTPOS_SFEN}\ngo depth 3\n"
        );
        assert_eq!(String::from_utf8(buf).unwrap(), expected);
    }

    #[test]
    fn write_search_script_with_moves() {
        let sfen = "9/4k4/9/9/9/9/9/4K4/9 b 9P9p 1";
        let moves = "5h4h 5b4b";
        let mut buf = Vec::new();
        write_search_script(&mut buf, sfen, moves, 5, 1, None).unwrap();
        let expected = format!(
            "usi\nsetoption name Threads value 1\nsetoption name BookFile value no_book\n\
             isready\nusinewgame\nposition sfen {sfen} moves {moves}\ngo depth 5\n"
        );
        assert_eq!(String::from_utf8(buf).unwrap(), expected);
    }

    #[test]
    fn write_search_script_sets_usi_hash_when_requested() {
        // `USI_Hash` must be set before `isready`, where the engine allocates
        // and clears the table; a later resize would not reproduce the clean
        // state the fixture is captured in.
        let mut buf = Vec::new();
        write_search_script(&mut buf, STARTPOS_SFEN, "", 3, 1, Some(256)).unwrap();
        let expected = format!(
            "usi\nsetoption name Threads value 1\nsetoption name BookFile value no_book\n\
             setoption name USI_Hash value 256\n\
             isready\nusinewgame\nposition sfen {STARTPOS_SFEN}\ngo depth 3\n"
        );
        assert_eq!(String::from_utf8(buf).unwrap(), expected);
    }

    #[test]
    fn renders_search_fixture_cp_no_moves() {
        let result = SearchResult {
            depth: 3,
            bestmove: "7g7f".to_string(),
            score: SearchScore::Cp(-18),
            nodes: 12345,
            pv: vec!["7g7f".to_string(), "3c3d".to_string()],
            fv_scale: None,
        };
        let json = render_search_fixture(STARTPOS_SFEN, "", None, &result);
        let expected = concat!(
            "{\n",
            "  \"sfen\": \"lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1\",\n",
            "  \"depth\": 3,\n",
            "  \"bestmove\": \"7g7f\",\n",
            "  \"score\": { \"cp\": -18 },\n",
            "  \"nodes\": 12345,\n",
            "  \"pv\": [\"7g7f\", \"3c3d\"]\n",
            "}\n"
        );
        assert_eq!(json, expected);
    }

    #[test]
    fn renders_search_fixture_mate_with_moves() {
        let sfen = "9/4k4/9/9/9/9/9/4K4/9 b 9P9p 1";
        let moves = "5h4h 5b4b";
        let result = SearchResult {
            depth: 5,
            bestmove: "2g2f".to_string(),
            score: SearchScore::Mate(3),
            nodes: 42,
            pv: vec!["2g2f".to_string(), "8b2b".to_string(), "2h2b+".to_string()],
            fv_scale: None,
        };
        let json = render_search_fixture(sfen, moves, None, &result);
        let expected = concat!(
            "{\n",
            "  \"sfen\": \"9/4k4/9/9/9/9/9/4K4/9 b 9P9p 1\",\n",
            "  \"moves\": [\"5h4h\", \"5b4b\"],\n",
            "  \"depth\": 5,\n",
            "  \"bestmove\": \"2g2f\",\n",
            "  \"score\": { \"mate\": 3 },\n",
            "  \"nodes\": 42,\n",
            "  \"pv\": [\"2g2f\", \"8b2b\", \"2h2b+\"]\n",
            "}\n"
        );
        assert_eq!(json, expected);
    }

    #[test]
    fn renders_search_fixture_whitespace_moves_omitted() {
        let result = SearchResult {
            depth: 1,
            bestmove: "1g1f".to_string(),
            score: SearchScore::Cp(0),
            nodes: 0,
            pv: vec!["1g1f".to_string()],
            fv_scale: None,
        };
        let json = render_search_fixture(STARTPOS_SFEN, "   ", None, &result);
        assert!(
            !json.contains("\"moves\""),
            "unexpected moves field in: {json}"
        );
        assert!(json.contains("\"depth\": 1"), "missing depth in: {json}");
        assert!(
            !json.contains("\"hash_mb\""),
            "unexpected hash_mb field in: {json}"
        );
    }

    #[test]
    fn parses_fv_scale_override_line() {
        // Verbatim from the reference at `isready`.
        let line = "info string engine option override. name = FV_SCALE , value = 28";
        assert_eq!(parse_fv_scale_override(line), Some(28));
        assert_eq!(
            parse_fv_scale_override(
                "info string engine option override. name = Threads , value = 4"
            ),
            None
        );
        assert_eq!(parse_fv_scale_override("info string Using 1 thread"), None);
        assert_eq!(
            parse_fv_scale_override("info depth 1 score cp 0 nodes 0 pv 1g1f"),
            None
        );
    }

    #[test]
    fn parse_search_output_records_the_announced_fv_scale() {
        let stdout = "\
usiok\n\
info string engine option override. name = FV_SCALE , value = 28\n\
readyok\n\
info depth 1 multipv 1 score cp 0 nodes 0 nps 0 hashfull 0 time 1 pv 1g1f\n\
bestmove 1g1f\n\
";
        let result = parse_search_output(stdout.as_bytes()).unwrap();
        assert_eq!(result.fv_scale, Some(28));
    }

    #[test]
    fn parse_search_output_leaves_fv_scale_unset_without_an_override() {
        let stdout = "\
readyok\n\
info depth 1 multipv 1 score cp 0 nodes 0 nps 0 hashfull 0 time 1 pv 1g1f\n\
bestmove 1g1f\n\
";
        let result = parse_search_output(stdout.as_bytes()).unwrap();
        assert_eq!(result.fv_scale, None);
    }

    #[test]
    fn renders_search_fixture_records_the_captured_fv_scale() {
        let result = SearchResult {
            depth: 3,
            bestmove: "7g7f".to_string(),
            score: SearchScore::Cp(-18),
            nodes: 12345,
            pv: vec!["7g7f".to_string()],
            fv_scale: Some(28),
        };
        let json = render_search_fixture(STARTPOS_SFEN, "", None, &result);
        assert!(
            json.contains("  \"fv_scale\": 28,\n"),
            "missing fv_scale in: {json}"
        );
    }

    #[test]
    fn renders_search_fixture_records_explicit_hash() {
        let result = SearchResult {
            depth: 3,
            bestmove: "7g7f".to_string(),
            score: SearchScore::Cp(-18),
            nodes: 12345,
            pv: vec!["7g7f".to_string()],
            fv_scale: None,
        };
        let json = render_search_fixture(STARTPOS_SFEN, "", Some(256), &result);
        assert!(
            json.contains("  \"hash_mb\": 256,\n"),
            "missing hash_mb in: {json}"
        );
    }
}
