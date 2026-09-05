use std::collections::BTreeMap;
use std::io::{self, BufRead, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use attic_numa::{DEFAULT_POLICY, NumaConfig, NumaIndex, SysfsOptions};
use attic_search::{
    BookConfig, BookHit, EnteringKingConfig, EnteringKingRule, PonderSignal, Prng, PvBound, PvInfo,
    PvOutputConfig, PvSink, QSearch, RootMove, Search, SearchControl, SharedHistories, TimeControl,
    TimeInput, TimeManagement, WorkerHistories, WorkerResult, WorkerVote, declaration_win,
    generate_root_moves, probe_book, select_best_worker, set_fv_scale,
};
use attic_state::{Move, Position, format_usi_move, parse_sfen, parse_usi_move};
use attic_storage::{Book, TranspositionTable, Value};

use crate::bench;
use crate::engine_options::{OverrideLine, parse_override_line};
use crate::formatter::Formatter;
use crate::option_profile::{ENGINE_OPTION_PROFILE_FILE, read_engine_option_profile};
use crate::options::{OptionStore, OptionValue};
use crate::parser::{Command, GoLimits, MATE_UNLIMITED_MS, PositionSfen, parse_line};

/// The `id name` value. Its version tracks how far upstream YaneuraOu has been
/// ported and nothing else — it moves only on an upstream catch-up, never for a
/// divergence, fix, or addition. The `git` suffix marks a build past the last
/// tagged release; a tagged release snapshot carries the plain `Attic 9.70`.
pub const ENGINE_NAME: &str = "Attic 9.70git";
pub const ENGINE_AUTHOR: &str = "Kei Ishida <ishida.kei@gmail.com>";

/// The largest iterative-deepening depth a `go` ever requests: one below
/// `MAX_PLY` (246), so `run_root`'s own `rootDepth + 1 < MAX_PLY` guard never
/// has to truncate it. Also the clamp for an out-of-range `go depth N`.
const SEARCH_MAX_DEPTH: i32 = 245;

// isready keep-alive: reference `Engine::run_heavy_job`, engine.cpp.
/// Stop-flag poll interval of the keep-alive helper (engine.cpp).
const KEEP_ALIVE_POLL_INTERVAL: Duration = Duration::from_millis(100);
/// Polls between bare keep-alive newlines: `50 * 100ms = 5s` (engine.cpp).
/// A GUI reads the periodic empty line as a sign the engine is alive and does
/// not time out while the `USI_Hash` allocation and the ~215 MiB `nn.bin` load
/// run between `isready` and `readyok`.
const KEEP_ALIVE_TICKS_PER_NEWLINE: u32 = 50;

// --- Reference USI score conversion (score.cpp / usi.cpp `format_score`). ---
/// `VALUE_MATE` (`types.h`).
const VALUE_MATE: Value = 32000;
/// `VALUE_TB_WIN_IN_MAX_PLY` (`types.h`): the `is_decisive` threshold.
const VALUE_TB_WIN_IN_MAX_PLY: Value = VALUE_MATE - 246;
/// `Eval::PawnValue` / `NormalizeToPawnValue` (`usi.cpp`).
const PAWN_VALUE: Value = 90;
/// `VALUE_INFINITE` (`types.h`): the pre-search `rootMoves[0].score`
/// sentinel the `ResignValue` guard excludes (`yaneuraou-search.cpp`).
const VALUE_INFINITE: Value = 32001;

/// The reference `USIEngine::to_cp` (`usi.cpp`). Unlike [`format_score`]
/// it does not special-case mate scores — the reference applies the same linear
/// map to all values.
fn to_cp(v: Value) -> Value {
    100 * v / PAWN_VALUE
}

/// Format a search value the way the reference USI layer does: a mate distance
/// for decisive scores, else centipawns.
fn format_score(v: Value) -> String {
    if v.abs() >= VALUE_TB_WIN_IN_MAX_PLY {
        let distance = VALUE_MATE - v.abs();
        let mate = if v > 0 { distance } else { -distance };
        format!("mate {mate}")
    } else {
        format!("cp {}", 100 * v / PAWN_VALUE)
    }
}

/// A loaded evaluation network paired with the `nn.bin` path it came from.
///
/// The path is retained so `isready` is idempotent: a repeat with the same
/// `<EvalDir>/nn.bin` reuses the already-loaded [`Search`] instead of parsing
/// the file again.
///
/// # Per-NUMA-node replication
///
/// When binding is active *and*
/// [`requires_memory_replication`](NumaConfig::requires_memory_replication), the
/// shared network is replaced by one on-node copy per *system* NUMA node touched
/// by the binding assignment (the in-process analog of the reference
/// `LazyNumaReplicatedSystemWide<Networks>`, minus its POSIX shared-memory
/// layer). Replica granularity is the system node, not the possibly L3-bundled
/// logical node, so logical nodes sharing a system node share one copy
/// (`get_discriminator`, `numa.h`).
struct LoadedEval {
    path: PathBuf,
    /// The instance the file was loaded into (the reference's replication
    /// `source`), and the clone source for the on-node replicas.
    source: Arc<Search>,
    /// System-node → on-node replica. Empty when replication is inactive.
    replicas: BTreeMap<NumaIndex, Arc<Search>>,
}

/// The result of the heavy `isready` initialisation, consumed only once the
/// keep-alive helper has stopped. A failed load emits
/// `info string eval load failed: …` and no `readyok`.
enum IsreadyOutcome {
    Ready,
    LoadFailed(String),
}

/// The opened opening books plus the `IgnoreBookPly` value captured at load
/// time — the reference captures it at `read_book` time (`book.cpp`), so
/// changing it requires a reload.
///
/// `books` is the Multiple Book priority list (`memory_books`, `book.h`):
/// the numbered `stem-000…` series in ascending order then the plain base name,
/// restricted to the names that opened; a probe takes the first hit
/// (`book.cpp`). Only the coordinator probes, once per `go` before
/// helpers start: the on-the-fly read path is not thread-safe by design
/// (`book.h`).
struct LoadedBook {
    books: Vec<Book>,
    ignore_book_ply: bool,
}

/// A search worker running on its own thread. The main thread keeps reading USI
/// lines while this runs; `stop` / `quit` set [`Self::stop`], which the search
/// polls at the reference `check_time` granularity. The worker emits its own
/// `info` / `bestmove`.
struct ActiveSearch {
    handle: JoinHandle<SearchState>,
    stop: Arc<AtomicBool>,
    /// The shared `go ponder` state. A plain `ponderhit` clears it, turning the
    /// pondering search into a normal time-managed one; when `None`, a stray
    /// `ponderhit` falls back to a `stop`.
    ponder: Option<Arc<PonderSignal>>,
    /// Suppresses the coordinator's `bestmove` (and final PV) for the
    /// Stochastic_Ponder ponderhit teardown, which stops the rewound search
    /// without emitting anything (`usi.cpp`).
    suppress: Arc<AtomicBool>,
    /// The root game ply this search ran at, carried so a completed real search
    /// updates the driver's `last_game_ply` (`yaneuraou-search.cpp`).
    game_ply: i32,
}

/// The session-owned search state a `go` lends to its worker and reclaims when
/// the worker finishes. The histories persist across `go`s within one game and
/// are reset by `usinewgame`, matching the reference `search_clear`.
struct SearchState {
    histories: WorkerHistories,
    /// The chosen worker's score / average score and the main worker's final
    /// `timeReduction`, seeding the next `go`'s time management
    /// (`yaneuraou-search.cpp`).
    ///
    /// Always `Some`: the reference runs that bookkeeping on every path, so the
    /// SKIP_SEARCH short-circuits (book / declaration / resign / no legal move)
    /// carry the unsearched defaults `(-VALUE_INFINITE, -VALUE_INFINITE)`. The
    /// third element is `Some` only after a real search — on a short-circuit the
    /// reference never touches `previousTimeReduction`, so the persisted value
    /// must stay put.
    time_state: Option<(Value, Value, Option<f64>)>,
}

pub struct UsiDriver<R: BufRead, W: Write + Send + 'static> {
    reader: R,
    /// The output sink, shared with the search worker; the `Mutex` serialises
    /// its lines against the main thread's.
    writer: Arc<Mutex<W>>,
    options: OptionStore,
    pos: Position,
    /// The loaded network holder, present only after a successful `isready`.
    /// `go` before this is set replies `bestmove resign`.
    eval: Option<LoadedEval>,
    /// The shared transposition table. Sized from `USI_Hash` at the first
    /// successful `isready` and by every later `setoption name USI_Hash`,
    /// cleared on `usinewgame`. `run_root` bumps the generation itself; the
    /// driver never does. `resize` / `clear` go through [`Arc::get_mut`], which
    /// succeeds only once every worker clone has dropped — the
    /// lifecycle-exclusivity contract on [`TranspositionTable`].
    tt: Arc<TranspositionTable>,
    /// Game-scoped worker histories. `None` only while a worker
    /// holds them mid-search.
    histories: Option<WorkerHistories>,
    /// The loaded opening books. `None` means bookless: the default
    /// `BookFile=no_book`, or every listed book failed or was unsupported.
    book: Option<Arc<LoadedBook>>,
    /// The `(resolved-name-list, on-the-fly, ignore-book-ply)` signature of the
    /// last book load; `isready` reloads only when it changes, the reference's
    /// reload-skip (`book.cpp`).
    book_signature: Option<(Vec<PathBuf>, bool, bool)>,
    /// A session-scoped seed advanced per `go`, driving both the book-selection
    /// and `rtime` PRNGs. Seeded from process entropy by default, so the
    /// randomness varies across process runs.
    book_seed: u64,
    /// The in-flight search worker, if any.
    search: Option<ActiveSearch>,
    /// Time-management state persisting across `go`s within a game, reset by
    /// `usinewgame` (`yaneuraou-search.cpp`): the previous move's score
    /// and average score (`VALUE_INFINITE` for the first move) and its final
    /// `timeReduction` (`0.85` initially).
    best_previous_score: Value,
    best_previous_average_score: Value,
    previous_time_reduction: f64,
    /// The root game ply of the last completed real search
    /// (`yaneuraou-search.cpp`). At the next search start an odd
    /// `last_game_ply - game_ply` means the side to move alternated — e.g. a
    /// Stochastic_Ponder rewind — and flips the sign of the persisted previous
    /// scores before they seed that search (`:1470-1483`).
    last_game_ply: i32,
    /// The last `position` command in parsed form (`usi.h`), retained so a
    /// Stochastic_Ponder `go ponder` can rewind it by one move and a
    /// `ponderhit` can re-apply the real position.
    last_position: (PositionSfen, Vec<String>),
    /// The last `go` command's limits (`usi.h`), retained so a
    /// Stochastic_Ponder `ponderhit` can re-issue it with `ponder` stripped.
    last_go: Option<GoLimits>,
    /// The worker thread pool: a main-worker slot plus `Threads − 1` persistent
    /// helper threads following the reference `idle_loop` (park → receive job →
    /// run → report → park). The main worker is the per-`go` coordinator thread
    /// [`Self::handle_go`] spawns. Helpers own game-scoped histories, reset by
    /// recreating the pool on `usinewgame` or a `Threads` resize.
    pool: ThreadPool,
    /// The active NUMA layout, detected once at construction from the engine
    /// default policy and replaced by every `setoption name NumaPolicy`.
    numa_config: NumaConfig,
    /// The current worker → NUMA-node binding assignment, empty when binding is
    /// inactive. Index `i` is worker `i`: slot 0 is the per-`go` coordinator,
    /// `1..` the helper threads.
    numa_bound: Vec<NumaIndex>,
    /// Per-worker handles to the node-shared correction / pawn tables. Unbound,
    /// one table set sized to the whole pool is shared by every worker; bound,
    /// one set per node sized to that node's thread count.
    worker_shared: Vec<Arc<SharedHistories>>,
    /// Per-worker handles to the NNUE network the worker evaluates with, one per
    /// pool slot; empty until a network is loaded. Without replication every
    /// entry clones the one loaded instance, with it each points at its *system*
    /// node's replica.
    worker_networks: Vec<Arc<Search>>,
    /// Poll interval of the `isready` keep-alive helper thread, overridable so a
    /// test can drive it faster than the reference cadence.
    keep_alive_poll: Duration,
}

impl<R: BufRead, W: Write + Send + 'static> UsiDriver<R, W> {
    /// A driver whose book / `rtime` PRNG stream is seeded from process entropy,
    /// like the reference's default-constructed `AsyncPRNG` / `PRNG`
    /// (`book.h`, `timeman.cpp`) — every process run differs.
    pub fn new(reader: R, writer: Arc<Mutex<W>>) -> Self {
        Self::with_book_seed(reader, writer, Prng::random_seed())
    }

    /// A driver with an explicit book-PRNG session seed.
    ///
    /// The engine-option profile is read from [`ENGINE_OPTION_PROFILE_FILE`] in
    /// the process's current directory, matching the reference call site
    /// (`usi.cpp`).
    pub fn with_book_seed(reader: R, writer: Arc<Mutex<W>>, book_seed: u64) -> Self {
        Self::with_option_profile(
            reader,
            writer,
            book_seed,
            Path::new(ENGINE_OPTION_PROFILE_FILE),
        )
    }

    /// A driver whose engine-option profile is read from `profile_path`. The
    /// read happens before the option map is built — and therefore before any
    /// `usi` reply — and prints nothing; a missing file is silently ignored.
    pub fn with_option_profile(
        reader: R,
        writer: Arc<Mutex<W>>,
        book_seed: u64,
        profile_path: &Path,
    ) -> Self {
        let options = OptionStore::with_book_options(read_engine_option_profile(profile_path));
        let threads = options.threads();
        // The engine default policy (`engine.cpp`).
        let numa_config = numa_config_from_policy("auto", &real_sysfs_options())
            .expect("default NumaPolicy `auto` always resolves to a valid config");
        let numa_bound = compute_numa_binding(&numa_config, "auto", threads);
        let pool = ThreadPool::with_binding(threads, bind_plan(&numa_config, &numa_bound));
        let worker_shared = build_worker_shared(&numa_config, &numa_bound, threads);
        let histories = Some(WorkerHistories::with_shared(Arc::clone(&worker_shared[0])));
        Self {
            reader,
            writer,
            options,
            pos: Position::startpos(),
            eval: None,
            tt: Arc::new(TranspositionTable::new()),
            histories,
            book: None,
            book_signature: None,
            book_seed,
            search: None,
            best_previous_score: VALUE_INFINITE,
            best_previous_average_score: VALUE_INFINITE,
            previous_time_reduction: 0.85,
            last_game_ply: 0,
            last_position: (PositionSfen::StartPos, Vec::new()),
            last_go: None,
            pool,
            numa_config,
            numa_bound,
            worker_shared,
            worker_networks: Vec::new(),
            keep_alive_poll: KEEP_ALIVE_POLL_INTERVAL,
        }
    }

    /// Override the `isready` keep-alive poll interval. The newline still fires
    /// only after [`KEEP_ALIVE_TICKS_PER_NEWLINE`] polls, so this scales the
    /// whole cadence rather than just the poll.
    pub fn with_keep_alive_poll(mut self, poll: Duration) -> Self {
        self.keep_alive_poll = poll;
        self
    }

    pub fn run(mut self) -> io::Result<()> {
        let mut buf = String::new();
        loop {
            buf.clear();
            let n = self.reader.read_line(&mut buf)?;
            if n == 0 {
                // EOF: treat as quit.
                self.finish_search_join();
                return Ok(());
            }
            match parse_line(&buf) {
                Command::Usi => self.handle_usi()?,
                Command::IsReady => self.handle_isready()?,
                Command::SetOption { name, value } => self.handle_setoption(&name, &value)?,
                Command::UsiNewGame => self.handle_usinewgame(),
                Command::Position { sfen, moves } => self.handle_position(sfen, &moves)?,
                Command::Go(limits) => self.handle_go(limits)?,
                Command::Stop => self.handle_stop(),
                Command::GameOver => self.handle_gameover(),
                Command::PonderHit => self.handle_ponderhit()?,
                Command::Bench(tokens) => self.handle_bench(&tokens)?,
                Command::Quit => {
                    self.finish_search_join();
                    return Ok(());
                }
                Command::Unknown(line) => self.handle_unknown(&line)?,
                Command::TooLong => self.handle_too_long()?,
            }
        }
    }

    /// Lock the shared output sink, recovering from a poisoned mutex: a worker
    /// panic must not wedge the main loop's own output.
    fn lock_writer(&self) -> MutexGuard<'_, W> {
        self.writer.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Emit one `info string <msg>` line.
    fn info_string(&self, msg: &str) -> io::Result<()> {
        Formatter::new(&mut *self.lock_writer()).info_string(msg)
    }

    /// Emit one verbatim line with no USI keyword prefix — the option-override
    /// `Error : ...` diagnostics the reference writes to raw `std::cout`
    /// (`usioption.cpp`).
    fn emit_raw_line(&self, text: &str) -> io::Result<()> {
        Formatter::new(&mut *self.lock_writer()).raw_line(text)
    }

    /// Emit one `bestmove <mv>` line.
    fn bestmove(&self, mv: &str) -> io::Result<()> {
        Formatter::new(&mut *self.lock_writer()).bestmove(mv)
    }

    /// Emit one `readyok` line.
    fn readyok(&self) -> io::Result<()> {
        Formatter::new(&mut *self.lock_writer()).readyok()
    }

    /// If a search worker is running, request its stop and join it, reclaiming
    /// the session-owned histories. Idempotent, and setting the stop flag is
    /// harmless once the worker has finished naturally: a short fixed-depth
    /// search that never reached a `check_time` checkpoint completes regardless.
    ///
    /// Joining also drops the worker's `Arc` clone of the transposition table,
    /// so afterwards `Arc::get_mut` (used by `resize` / `clear`) succeeds.
    fn finish_search_join(&mut self) {
        if let Some(active) = self.search.take() {
            active.stop.store(true, Ordering::Relaxed);
            let state = active
                .handle
                .join()
                .expect("search worker thread must not panic");
            self.histories = Some(state.histories);
            // Carry the finished search's time-management outputs forward
            // (`yaneuraou-search.cpp`).
            if let Some((score, avg, tr)) = state.time_state {
                self.best_previous_score = score;
                self.best_previous_average_score = avg;
                if let Some(tr) = tr {
                    self.previous_time_reduction = tr;
                }
                self.last_game_ply = active.game_ply;
            }
        }
    }

    /// Resize the shared transposition table to the current `USI_Hash` value in
    /// MiB (`yaneuraou-search.cpp`). The caller must have run
    /// [`Self::finish_search_join`] first so [`Arc::get_mut`] succeeds.
    fn resize_tt_to_hash_option(&mut self) {
        let mb = self.options.spin("USI_Hash").max(1) as usize;
        Arc::get_mut(&mut self.tt)
            .expect("no search worker holds the TT during a USI_Hash resize")
            .resize(mb);
    }

    /// Recompute the worker → NUMA-node binding for the current `Threads` /
    /// `NumaPolicy` options and rebuild the worker pool with it. Every pool
    /// (re)build routes through here so [`Self::numa_bound`] stays consistent
    /// with the live pool (`thread.cpp`). Helpers bind once at spawn;
    /// the per-`go` coordinator binds at each `go`.
    ///
    /// Callers must have joined any running search first — a resize destroys and
    /// recreates the helper threads.
    fn rebuild_pool(&mut self) {
        let requested = self.options.threads();
        let policy = self.options.text("NumaPolicy").to_string();
        self.numa_bound = compute_numa_binding(&self.numa_config, &policy, requested);
        // Every pool rebuild resets the shared tables, matching the reference
        // (`thread.cpp`). The coordinator's own game-scoped per-worker
        // tables persist, so only its shared handle is swapped.
        self.worker_shared = build_worker_shared(&self.numa_config, &self.numa_bound, requested);
        if let Some(h) = self.histories.as_mut() {
            h.set_shared(Arc::clone(&self.worker_shared[0]));
        }
        let plan = bind_plan(&self.numa_config, &self.numa_bound);
        self.pool.set_with_binding(requested, plan);
        // The reference forces replication right after `resize_threads`
        // (`engine.cpp`).
        self.rebuild_networks();
    }

    /// Ensure a network replica exists for every *system* NUMA node the current
    /// binding assignment touches, and resolve the per-worker
    /// [`Self::worker_networks`] handles — the analog of the reference
    /// `ensure_network_replicated` (`thread.cpp`), forced at
    /// configuration time so no replication ever runs on the search path.
    ///
    /// With no network loaded or replication not required, every worker shares
    /// the one loaded instance. Otherwise one copy per distinct system node is
    /// cloned inside a thread bound to a logical node of it, so the copy's pages
    /// first-touch there. Existing replicas are reused — every instance is
    /// byte-identical — so a rebuild that leaves the layout unchanged clones
    /// nothing.
    fn rebuild_networks(&mut self) {
        let requested = self.pool.size().max(1);
        let replication_active =
            !self.numa_bound.is_empty() && self.numa_config.requires_memory_replication();

        // Resolved before `self.eval` is borrowed below. The representative
        // logical node is the lowest-indexed worker's on that system node —
        // stable, and any logical node there first-touches the pages correctly.
        let (sys_nodes, rep_logical) = if replication_active {
            let sys = self
                .numa_config
                .system_nodes_for_binding(&self.numa_bound, &real_sysfs_options());
            let mut rep: BTreeMap<NumaIndex, NumaIndex> = BTreeMap::new();
            for (&s, &logical) in sys.iter().zip(self.numa_bound.iter()) {
                rep.entry(s).or_insert(logical);
            }
            (sys, rep)
        } else {
            (Vec::new(), BTreeMap::new())
        };

        let config = &self.numa_config;
        let Some(eval) = self.eval.as_mut() else {
            self.worker_networks = Vec::new();
            return;
        };

        self.worker_networks = resolve_worker_networks(
            &eval.source,
            &mut eval.replicas,
            &sys_nodes,
            &rep_logical,
            requested,
            replication_active,
            |logical, src| {
                let mut built: Option<Arc<Search>> = None;
                config.execute_on_numa_node(logical, || {
                    built = Some(Arc::new(src.replicate()));
                });
                built.expect("execute_on_numa_node ran the closure")
            },
        );
    }

    /// Emit each non-blank line of `text` as `info string <line>`, mirroring the
    /// reference `print_info_string` (`usi.cpp`).
    fn emit_info_string_lines(&self, text: &str) -> io::Result<()> {
        for line in text.split('\n') {
            if !line.trim().is_empty() {
                self.info_string(line)?;
            }
        }
        Ok(())
    }

    /// Emit the `Available processors: ...` line (`engine.cpp`).
    fn emit_numa_config_information(&self) -> io::Result<()> {
        self.emit_info_string_lines(&numa_config_information_as_string(&self.numa_config))
    }

    /// Emit the `Using N thread[s][ with NUMA node thread binding: ...]` line
    /// (`engine.cpp`).
    fn emit_thread_allocation_information(&self) -> io::Result<()> {
        self.emit_info_string_lines(&thread_allocation_information_as_string(
            self.pool.size(),
            &self.numa_config,
            &self.numa_bound,
        ))
    }

    fn handle_usi(&mut self) -> io::Result<()> {
        let mut guard = self.lock_writer();
        let mut f = Formatter::new(&mut *guard);
        f.id_name(ENGINE_NAME)?;
        f.id_author(ENGINE_AUTHOR)?;
        for decl in self.options.iter_declarations() {
            f.option_decl(decl)?;
        }
        f.usiok()
    }

    /// The `<EvalDir>/nn.bin` path the network is loaded from.
    fn nn_bin_path(&self) -> PathBuf {
        let dir = match self.options.get("EvalDir") {
            Some(OptionValue::String(s)) => s.as_str(),
            // Unreachable: a declared option keeps its declared type.
            _ => "eval",
        };
        Path::new(dir).join("nn.bin")
    }

    /// The absolute path a `<BookDir>/<BookFile>` pair resolves to, mirroring
    /// the reference `get_book_name` (`book.cpp`): `BookDir` joined
    /// onto the binary's folder, then `BookFile`. An absolute `BookDir` wins
    /// over the binary folder.
    fn book_path(&self, book_dir: &str, book_file: &str) -> PathBuf {
        let base = std::env::current_exe()
            .ok()
            .and_then(|p| p.parent().map(Path::to_path_buf))
            .unwrap_or_else(|| PathBuf::from("."));
        base.join(book_dir).join(book_file)
    }

    /// (Re)load the opening books from the current options, mirroring the
    /// reference `BookMoveSelector::read_book` (`yaneuraou-search.cpp`).
    /// Reloads only when the `(name list, on-the-fly, IgnoreBookPly)` capture
    /// changed (`book.cpp`).
    ///
    /// `no_book`, an unsupported (non-`.ybb`) format, or an open failure all
    /// leave that name out of the priority list without panicking; a `.db` whose
    /// file is absent falls back to the `.ybb` sibling.
    fn reload_book(&mut self) -> io::Result<()> {
        let book_file = self.options.text("BookFile").to_string();
        let book_dir = self.options.text("BookDir").to_string();
        let on_the_fly = self.options.check("BookOnTheFly");
        let ignore_book_ply = self.options.check("IgnoreBookPly");
        let base = self.book_path(&book_dir, &book_file);

        // The resolved name list is half of the reload-skip capture, so a
        // numbered file appearing or vanishing between two `isready`s is itself
        // a reason to reload.
        let (names, notices) = book_names(&base);

        let signature = (names.clone(), on_the_fly, ignore_book_ply);
        if self.book_signature.as_ref() == Some(&signature) {
            return Ok(());
        }
        self.book_signature = Some(signature);
        self.book = None;

        // `no_book` → bookless, silently.
        if book_file == "no_book" {
            return Ok(());
        }

        // Verbatim from the reference (`book.cpp`).
        for notice in &notices {
            self.info_string(notice)?;
        }

        let mut books: Vec<Book> = Vec::new();
        for name in &names {
            // Per name, as the reference does inside `MemoryBook::read_book`
            // (`book.cpp`).
            let resolved = resolve_book_filename_with_ybb_fallback(name);
            if &resolved != name {
                self.info_string(&format!(
                    "book file fallback : {} -> {}",
                    name.display(),
                    resolved.display()
                ))?;
            }

            // Only `.ybb` is supported, unlike the reference. Anything else
            // behaves as no-book after a notice — never a silent skip, which
            // would hide a book the reference would have used.
            if !has_book_ext(&resolved, BOOK_EXT_YBB) {
                self.info_string(&format!("unsupported book format : {}", resolved.display()))?;
                continue;
            }

            let opened = if on_the_fly {
                Book::open_on_the_fly(&resolved)
            } else {
                Book::open_in_memory(&resolved)
            };
            match opened {
                Ok(book) => {
                    let count = book.record_count();
                    books.push(book);
                    self.info_string(&format!("book loaded : {count} positions"))?;
                }
                Err(e) => {
                    self.info_string(&format!("book load failed : {e}"))?;
                }
            }
        }

        if !books.is_empty() {
            self.book = Some(Arc::new(LoadedBook {
                books,
                ignore_book_ply,
            }));
        }
        Ok(())
    }

    /// Read an option-override file and apply each line, mirroring the reference
    /// `OptionsMap::read_engine_options` (`usioption.cpp`). A missing or
    /// unreadable file is a silent no-op.
    fn read_engine_options(&mut self, path: &Path) -> io::Result<()> {
        let contents = match std::fs::read_to_string(path) {
            Ok(c) => c,
            Err(_) => return Ok(()),
        };
        self.info_string(&format!("read engine options, path = {}", path.display()))?;
        for line in contents.lines() {
            self.apply_override_line(line)?;
        }
        Ok(())
    }

    /// Apply one override line, mirroring the reference `build_option`
    /// (`usioption.cpp`). An out-of-range or ill-typed value is silently
    /// not stored, as the reference `operator=` range guard is, and the option is
    /// locked FIXED either way so later `setoption`s cannot change it.
    fn apply_override_line(&mut self, line: &str) -> io::Result<()> {
        let (name, value, invalid) = match parse_override_line(line) {
            OverrideLine::Empty => return Ok(()),
            OverrideLine::Plain { name, value } => (name, value, Vec::new()),
            OverrideLine::Full {
                name,
                value,
                invalid_tokens,
            } => (name, value, invalid_tokens),
        };

        // Reported first, but do not abort the override
        // (`usioption.cpp`).
        for tok in &invalid {
            self.emit_raw_line(&format!("Error : invalid command: {tok}"))?;
        }

        let Some(canonical) = self.options.canonical_name(&name) else {
            return self.emit_raw_line(&format!("Error : option name not found : {name}"));
        };

        // Order matters: a fixed option ignores `set_value`.
        let _ = self.options.set_value(canonical, &value);
        self.options.mark_fixed(canonical);

        // Mirror the reference option `on_change` handlers for the options whose
        // value drives a resource.
        if canonical.eq_ignore_ascii_case("Threads") {
            self.finish_search_join();
            self.rebuild_pool();
        } else if canonical.eq_ignore_ascii_case("USI_Hash") {
            self.finish_search_join();
            self.resize_tt_to_hash_option();
        } else if canonical.eq_ignore_ascii_case("NumaPolicy") {
            // Unlike the `setoption` path, an override is a startup-time config
            // step, so a bad value is reported rather than process-fatal and the
            // previously detected config is kept.
            self.finish_search_join();
            let policy = self.options.text("NumaPolicy").to_string();
            match numa_config_from_policy(&policy, &real_sysfs_options()) {
                Ok(cfg) => self.numa_config = cfg,
                Err(msg) => self.info_string(&format!("NumaPolicy error: {msg}"))?,
            }
            self.rebuild_pool();
        }

        self.info_string(&format!(
            "engine option override. name = {name} , value = {value}"
        ))
    }

    fn handle_isready(&mut self) -> io::Result<()> {
        self.finish_search_join();
        // Before the engine's own isready work, as the reference does
        // (`usi.cpp`).
        self.read_engine_options(Path::new("engine_options.txt"))?;
        let eval_options = Path::new(self.options.text("EvalDir")).join("eval_options.txt");
        self.read_engine_options(&eval_options)?;

        // The reference wraps only `Eval::load_eval` in its keep-alive scope
        // (yaneuraou-search.cpp); here the heavy work is the whole block, so
        // the keep-alive brackets all of it.
        let outcome = {
            let _keep_alive = KeepAlive::spawn(Arc::clone(&self.writer), self.keep_alive_poll);
            self.isready_heavy_job()?
        };

        match outcome {
            IsreadyOutcome::Ready => self.readyok(),
            IsreadyOutcome::LoadFailed(reason) => {
                // No `readyok` on a load failure, and a previously-loaded
                // network is left untouched so a bad reload keeps a working net.
                self.info_string(&format!("eval load failed: {reason}"))
            }
        }
    }

    /// The heavy `isready` initialisation. It returns the outcome rather than
    /// emitting it, so the terminal reply never races the keep-alive newlines.
    fn isready_heavy_job(&mut self) -> io::Result<IsreadyOutcome> {
        self.reload_book()?;
        let path = self.nn_bin_path();

        if self.eval.as_ref().is_some_and(|e| e.path == path) {
            return Ok(IsreadyOutcome::Ready);
        }

        match Search::from_network_file_with_warnings(&path) {
            Ok((search, warnings)) => {
                // The loader's non-fatal warnings (hash mismatches), mirroring
                // the reference `Detail::ReadParameters` diagnostics.
                for warning in &warnings {
                    self.info_string(warning)?;
                }
                // Only fires when the host never set `USI_Hash` itself; a
                // `setoption` before this will already have sized the table.
                if self.tt.cluster_count() == 0 {
                    self.resize_tt_to_hash_option();
                }
                self.eval = Some(LoadedEval {
                    path,
                    source: Arc::new(search),
                    // A reload starts with no replicas, dropping the stale set.
                    replicas: BTreeMap::new(),
                });
                // As the reference does after a reload (`engine.cpp`), so no
                // replica is ever built on the search path.
                self.rebuild_networks();
                Ok(IsreadyOutcome::Ready)
            }
            Err(e) => Ok(IsreadyOutcome::LoadFailed(e.to_string())),
        }
    }

    fn handle_setoption(&mut self, name: &str, value: &str) -> io::Result<()> {
        // The option store accepts any string, but mapping it to a `NumaConfig`
        // can fail loudly, and a successful set emits both info lines.
        if name.eq_ignore_ascii_case("NumaPolicy") {
            return self.handle_setoption_numa_policy(value);
        }
        match self.options.set_value(name, value) {
            Ok(()) => {
                // Like the reference `ThreadPool::set`, this destroys and
                // recreates every worker rather than diffing the count.
                if name.eq_ignore_ascii_case("Threads") {
                    self.finish_search_join();
                    self.rebuild_pool();
                    // The reference prints the allocation line from the
                    // `Threads` on_change callback (`engine.cpp`).
                    self.emit_thread_allocation_information()?;
                }
                // The reference's callback waits for any running search before
                // resizing (`yaneuraou-search.cpp`).
                if name.eq_ignore_ascii_case("USI_Hash") {
                    self.finish_search_join();
                    self.resize_tt_to_hash_option();
                }
                Ok(())
            }
            Err(e) => self.info_string(&format!("option {name} rejected: {e}")),
        }
    }

    /// Apply `setoption name NumaPolicy value <v>` (`engine.cpp`).
    ///
    /// `auto` / `system` detect from the system respecting affinity; `hardware`
    /// detects ignoring affinity; `none` is a single all-threads node; anything
    /// else is a custom node string. A string that fails to parse or yields zero
    /// nodes prints an `info string` and terminates the process, where the
    /// reference reaches `std::exit(EXIT_FAILURE)`.
    fn handle_setoption_numa_policy(&mut self, value: &str) -> io::Result<()> {
        // A fixed override silently ignores this, and `text` then returns the
        // fixed value — which is the one resolved below.
        let _ = self.options.set_value("NumaPolicy", value);
        let policy = self.options.text("NumaPolicy").to_string();

        self.finish_search_join();

        match numa_config_from_policy(&policy, &real_sysfs_options()) {
            Ok(cfg) => self.numa_config = cfg,
            Err(msg) => {
                self.info_string(&format!("NumaPolicy error: {msg}"))?;
                let _ = self.lock_writer().flush();
                std::process::exit(1);
            }
        }

        // The reference's callback returns both lines joined by a newline
        // (`engine.cpp`).
        self.rebuild_pool();
        self.emit_numa_config_information()?;
        self.emit_thread_allocation_information()
    }

    fn handle_position(&mut self, sfen: PositionSfen, moves: &[String]) -> io::Result<()> {
        // A scratch position, so `self.pos` survives a malformed line untouched.
        let mut scratch = match &sfen {
            PositionSfen::StartPos => Position::startpos(),
            PositionSfen::Sfen(s) => match parse_sfen(s) {
                Ok(p) => p,
                Err(e) => {
                    return self.info_string(&format!("position parse error: {e}"));
                }
            },
        };
        let mut legal_buf: Vec<Move> = Vec::new();
        for s in moves {
            let parsed = match parse_usi_move(s, &scratch) {
                Ok(m) => m,
                Err(_) => {
                    return self.info_string(&format!("illegal move: {s}"));
                }
            };
            legal_buf.clear();
            scratch.generate_legal_all(&mut legal_buf);
            if !legal_buf.contains(&parsed) {
                return self.info_string(&format!("illegal move: {s}"));
            }
            scratch.do_move(parsed);
        }
        self.pos = scratch;
        self.last_position = (sfen, moves.to_vec());
        Ok(())
    }

    fn handle_usinewgame(&mut self) {
        self.finish_search_join();
        self.pos = Position::startpos();
        Arc::get_mut(&mut self.tt)
            .expect("no search worker holds the TT during usinewgame")
            .clear();
        // `rebuild_pool` below swaps in the freshly built node table set, so
        // cloning the current handle here avoids a throwaway allocation.
        self.histories = Some(WorkerHistories::with_shared(Arc::clone(
            &self.worker_shared[0],
        )));
        // The first-move-of-a-game sentinels (`yaneuraou-search.cpp`).
        self.best_previous_score = VALUE_INFINITE;
        self.best_previous_average_score = VALUE_INFINITE;
        self.previous_time_reduction = 0.85;
        // `yaneuraou-search.cpp`; the `last_position` default is the
        // reference's `"position startpos"` (`usi.h`).
        self.last_game_ply = 0;
        self.last_position = (PositionSfen::StartPos, Vec::new());
        self.last_go = None;
        // The helper workers' game-scoped histories live in the pool threads, so
        // recreating the pool is what gives them the fresh tables the reference
        // `search_clear` provides.
        self.rebuild_pool();
    }

    /// Snapshot the book-selection options into a [`BookConfig`] for one `go`.
    /// `IgnoreBookPly` is not here — it is captured at load time and travels
    /// with [`LoadedBook`].
    ///
    /// Both profiles' fields are snapshotted. An option the active profile did
    /// not register reads as its type's zero, which is inert on the leg that
    /// never consults it.
    fn book_config(&self) -> BookConfig {
        BookConfig {
            book_options_v2: self.options.book_options_v2(),
            narrow_book: self.options.check("NarrowBook"),
            book_moves: self.options.spin("BookMoves"),
            ignore_rate: self.options.spin("BookIgnoreRate"),
            eval_diff: self.options.spin("BookEvalDiff"),
            eval_black_diff: self.options.spin("BookEvalBlackDiff"),
            eval_white_diff: self.options.spin("BookEvalWhiteDiff"),
            eval_black_limit: self.options.spin("BookEvalBlackLimit"),
            eval_white_limit: self.options.spin("BookEvalWhiteLimit"),
            depth_limit: self.options.spin("BookDepthLimit"),
            depth_black_limit: self.options.spin("BookDepthBlackLimit"),
            depth_white_limit: self.options.spin("BookDepthWhiteLimit"),
            consider_move_count: self.options.check("ConsiderBookMoveCount"),
            pv_moves: self.options.spin("BookPvMoves"),
            flipped_book: self.options.check("FlippedBook"),
        }
    }

    fn handle_go(&mut self, limits: GoLimits) -> io::Result<()> {
        self.finish_search_join();

        self.last_go = Some(limits.clone());

        // Stochastic_Ponder ponders one move earlier than the retained position
        // (`usi.cpp`); `ponderMode` stays set.
        if limits.ponder && self.options.check("Stochastic_Ponder") {
            self.apply_stochastic_ponder_rewind();
        }

        // Rewound under Stochastic_Ponder, so read after the rewind above.
        let game_ply = self.pos.ply() as i32;

        let Some(job) = self.prepare_coordinator_job(limits, false) else {
            self.info_string("no eval network loaded; run isready")?;
            return self.bestmove("resign");
        };

        // Cloned out of the job before it moves into the worker thread.
        let stop_for_active = Arc::clone(&job.stop);
        let ponder_for_active = job.ponder.as_ref().map(Arc::clone);
        let suppress_for_active = Arc::clone(&job.suppress_bestmove);
        let handle = std::thread::spawn(move || {
            let outcome = run_coordinated(job);
            SearchState {
                histories: outcome.histories,
                time_state: outcome.time_state,
            }
        });

        self.search = Some(ActiveSearch {
            handle,
            stop: stop_for_active,
            ponder: ponder_for_active,
            suppress: suppress_for_active,
            game_ply,
        });
        Ok(())
    }

    /// Stochastic_Ponder `go ponder` rewind (`usi.cpp`): the retained
    /// position with its last move dropped, installed as the search root. Best
    /// effort — nothing to rewind, or a rebuild failure, leaves it untouched.
    fn apply_stochastic_ponder_rewind(&mut self) {
        let (sfen, moves) = &self.last_position;
        if moves.is_empty() {
            return;
        }
        let rewound = &moves[..moves.len() - 1];
        if let Some(pos) = build_position_from(sfen, rewound) {
            self.pos = pos;
        }
    }

    /// Build the [`CoordinatorJob`] for one search — the shared preamble of both
    /// `go` and `bench`.
    ///
    /// Returns `None` when no network is loaded; the caller emits the resign or
    /// notice appropriate to its context. `disable_pv_interval` mirrors the
    /// reference `limits.disablePvInterval` (`usi.cpp`): it forces the
    /// per-iteration PV interval to zero so every iteration prints, and is set
    /// only by `bench`.
    fn prepare_coordinator_job(
        &mut self,
        mut limits: GoLimits,
        disable_pv_interval: bool,
    ) -> Option<CoordinatorJob<W>> {
        // The reference's mutable global `NNUE::FV_SCALE`. Written at the start
        // of every search, so a `setoption` issued mid-search leaves the running
        // search — which read the scale at its own start — unperturbed.
        set_fv_scale(self.options.spin("FV_SCALE") as i32);

        // An explicit `go` token wins over the option (`usi.cpp`). A
        // seeded depth disables the parallel-search vote below just as an
        // explicit one does: the reference's `!limits.depth` guard
        // (`yaneuraou-search.cpp`) keys off the final value, not its source.
        if limits.depth.is_none() {
            let dl = self.options.spin("DepthLimit");
            if dl != 0 {
                limits.depth = Some(dl as u32);
            }
        }
        if limits.nodes.is_none() {
            let nl = self.options.spin("NodesLimit");
            if nl != 0 {
                limits.nodes = Some(nl as u64);
            }
        }

        // The per-worker handles are resolved alongside `eval`, so a loaded
        // `eval` always has a network for every worker.
        self.eval.as_ref()?;

        // The reference `use_time_management()` (`search.h`) is true only
        // for a real clock or `go rtime`; a `TimeControl` is installed for those
        // and for `go movetime`, and is `None` otherwise.
        let us = self.pos.side_to_move();
        let now = Instant::now();
        let use_time_management = limits.mate.is_none()
            && limits.movetime.is_none()
            && limits.depth.is_none()
            && limits.nodes.is_none()
            && !limits.infinite;
        // The reference leaves `go mate`'s enforcement to a separate mate engine,
        // which this port has none of, so a concrete `go mate <ms>` budget is
        // mapped onto a `movetime`-style bound. A bare or `infinite` `go mate`
        // carries no bound and runs until `stop`.
        let mate_budget = match limits.mate {
            Some(m) if m != MATE_UNLIMITED_MS => Some(m as i64),
            _ => None,
        };
        let movetime = limits.movetime.map(|m| m as i64).or(mate_budget);

        // Side-flip continuity (`yaneuraou-search.cpp`): when the side
        // to move alternated between the last completed search and this one, the
        // persisted previous scores are negated before they seed `iterValue` /
        // `fallingEval`. The first-move sentinel is exempt.
        let flip_previous = (self.last_game_ply - self.pos.ply() as i32) & 1 != 0;
        let best_prev_score = match self.best_previous_score {
            VALUE_INFINITE => VALUE_INFINITE,
            s if flip_previous => -s,
            s => s,
        };
        let best_prev_average_score = match self.best_previous_average_score {
            VALUE_INFINITE => VALUE_INFINITE,
            a if flip_previous => -a,
            a => a,
        };

        let time = if use_time_management || movetime.is_some() {
            let (time_opt, inc_opt) = match us {
                attic_state::Color::Black => (limits.btime, limits.binc),
                attic_state::Color::White => (limits.wtime, limits.winc),
            };
            let mmtd = remap_max_moves_to_draw(self.options.spin("MaxMovesToDraw"));
            // A stream distinct from the book selection's, so `go rtime`'s
            // randomised budget neither perturbs nor is perturbed by book choice.
            let mut prng = Prng::new(self.book_seed ^ 0xA5A5_5A5A_1234_5678);
            let tm = TimeManagement::init(
                &TimeInput {
                    time_us: time_opt.unwrap_or(0) as i64,
                    inc_us: inc_opt.unwrap_or(0) as i64,
                    byoyomi_us: limits.byoyomi.unwrap_or(0) as i64,
                    movetime: movetime.unwrap_or(0),
                    rtime: limits.rtime.unwrap_or(0) as i64,
                    network_delay: self.options.spin("NetworkDelay"),
                    network_delay2: self.options.spin("NetworkDelay2"),
                    minimum_thinking_time: self.options.spin("MinimumThinkingTime"),
                    slow_mover: self.options.spin("SlowMover"),
                    round_up_to_fullsecond: self.options.check("RoundUpToFullSecond"),
                    usi_ponder: self.options.check("USI_Ponder"),
                    stochastic_ponder: self.options.check("Stochastic_Ponder"),
                    ply: self.pos.ply() as i32,
                    max_moves_to_draw: mmtd,
                    start_time: now,
                },
                &mut prng,
            );
            if tm.mtg_error {
                let _ = self.info_string("Error! : MaxMovesToDraw is too small.");
            }
            Some(TimeControl {
                tm,
                use_time_management,
                movetime,
                n_threads: self.pool.size(),
                best_previous_score: best_prev_score,
                best_previous_average_score: best_prev_average_score,
                previous_time_reduction: self.previous_time_reduction,
            })
        } else {
            None
        };
        // Seeded active (`ponderMode`) so a later `ponderhit` can drive both the
        // main worker and the coordinator's hold loop.
        let ponder = limits.ponder.then(|| Arc::new(PonderSignal::new(true)));
        let control = SearchControl {
            stop: Some(Arc::new(AtomicBool::new(false))),
            ponder: ponder.as_ref().map(Arc::clone),
            node_limit: limits.nodes,
            time,
        };
        let depth = match limits.depth {
            Some(d) => (d as i32).clamp(1, SEARCH_MAX_DEPTH),
            None => SEARCH_MAX_DEPTH,
        };

        // Clamped to the legal-move count inside the worker.
        let multi_pv = (self.options.spin("MultiPV").max(1)) as usize;

        // The reference consults `get_best_thread` only for
        // `MultiPV == 1 && !limits.depth && !limits.mate`
        // (`yaneuraou-search.cpp`), so a fixed-depth result stays
        // reproducible, MultiPV shows every line, and a mate proof stays on the
        // main worker's own line.
        let mate_mode = limits.mate.is_some();
        let use_voting = limits.depth.is_none() && multi_pv == 1 && !mate_mode;

        // `yaneuraou-search.cpp`. A zero interval never suppresses, so
        // every iteration prints.
        let consideration_mode = self.options.check("ConsiderationMode");
        let computed_pv_interval = if disable_pv_interval || limits.infinite || consideration_mode {
            Duration::ZERO
        } else {
            Duration::from_millis(self.options.spin("PvInterval").max(0) as u64)
        };
        let pv_config = PvOutputConfig {
            multi_pv,
            pv_interval: computed_pv_interval,
            consideration_mode,
            output_fail_lh_pv: self.options.check("OutputFailLHPV"),
            start_time: now,
        };

        // The pool is never resized while a coordinator runs — every resize path
        // joins first — so these stay valid for the whole `go`.
        let n_threads = self.pool.size();
        let helper_slots = self.pool.helper_slots();
        // Worker `h + 1` gets `worker_shared[h + 1]`, so drop the coordinator's
        // slot-0 handle.
        let helper_shared: Vec<Arc<SharedHistories>> =
            self.worker_shared[1..].iter().map(Arc::clone).collect();
        let helper_networks: Vec<Arc<Search>> =
            self.worker_networks[1..].iter().map(Arc::clone).collect();

        // The main histories are lent take-and-return and reclaimed on join;
        // helper histories live in the pool threads.
        let tt = Arc::clone(&self.tt);
        let histories = self
            .histories
            .take()
            .expect("session histories present when idle");
        let search = Arc::clone(&self.worker_networks[0]);
        let writer = Arc::clone(&self.writer);
        let stop = control
            .stop
            .clone()
            .expect("stop flag installed just above");
        let pos = self.pos.clone();

        let book = self.book.as_ref().map(Arc::clone);
        let own_book = self.options.check("USI_OwnBook");
        let book_config = self.book_config();
        self.book_seed = self
            .book_seed
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        let book_seed = self.book_seed;
        // `go infinite` holds the reply until `stop` / `ponderhit`, the
        // SKIP_SEARCH wait loop (`yaneuraou-search.cpp`).
        let infinite = limits.infinite;
        // When set, the coordinator emits no `bestmove` and no final PV for this
        // search (`usi.cpp`).
        let suppress_bestmove = Arc::new(AtomicBool::new(false));

        // The per-side thresholds are precomputed from the root position, as the
        // reference `set_ekr` does (`yaneuraou-search.cpp`); the material
        // total is invariant across the search, so every worker shares them.
        let entering_king = EnteringKingConfig::new(
            EnteringKingRule::from_option(self.options.text("EnteringKingRule")),
            &pos,
        );

        // A set value of 0 means unlimited, remapped to `100000` as the reference
        // does (`yaneuraou-search.cpp`).
        let max_moves_to_draw = remap_max_moves_to_draw(self.options.spin("MaxMovesToDraw"));

        // `drawValueTable[REPETITION_DRAW][us]` for the root side to move
        // (`yaneuraou-search.cpp`); the search negates it for the
        // opponent, the reference's symmetric `±draw_value`.
        let draw_option = match self.pos.side_to_move() {
            attic_state::Color::Black => self.options.spin("DrawValueBlack"),
            attic_state::Color::White => self.options.spin("DrawValueWhite"),
        };
        let draw_contempt: Value = (draw_option as Value) * PAWN_VALUE / 100;

        // The post-search resign threshold in centipawns
        // (`yaneuraou-search.cpp`), consumed at emit time.
        let resign_value = self.options.spin("ResignValue") as Value;

        // When true the search also considers the non-promoting moves the default
        // generator suppresses (`yaneuraou-search.cpp`).
        let generate_all_legal_moves = self.options.check("GenerateAllLegalMoves");

        // The reference binds pool thread 0 once at creation
        // (`thread.cpp`); this coordinator is spawned per `go`, so it
        // re-binds each time to the same node, which is idempotent.
        let numa_bind = if self.numa_bound.is_empty() {
            None
        } else {
            Some((self.numa_config.clone(), self.numa_bound[0]))
        };

        Some(CoordinatorJob {
            search,
            tt,
            pos,
            depth,
            use_voting,
            control,
            stop,
            histories,
            helper_slots,
            helper_shared,
            helper_networks,
            n_threads,
            numa_bind,
            book,
            book_config,
            own_book,
            book_seed,
            ponder,
            infinite,
            suppress_bestmove,
            entering_king,
            max_moves_to_draw,
            draw_contempt,
            resign_value,
            generate_all_legal_moves,
            mate_mode,
            pv_config,
            writer,
        })
    }

    /// Run one `bench` position synchronously and return its total searched node
    /// count across all workers — the value the reference `bench` accumulates
    /// from the final `info nodes`. A position with no network loaded resigns and
    /// contributes 0 nodes.
    fn bench_run_one(&mut self, limits: GoLimits) -> io::Result<u64> {
        let Some(job) = self.prepare_coordinator_job(limits, true) else {
            self.info_string("no eval network loaded; run isready")?;
            self.bestmove("resign")?;
            return Ok(0);
        };
        let outcome = run_coordinated(job);
        // Bench uses fixed depth / nodes / movetime, so `time_state` is
        // irrelevant to it.
        self.histories = Some(outcome.histories);
        Ok(outcome.nodes)
    }

    /// `bench [ttSizeMB] [threads] [limit] [default|current|<fenFile>] [limitType]`
    /// — a reproducible NPS benchmark ported from the reference `USIEngine::bench`
    /// (`usi.cpp`) + `setup_bench` (`benchmark.cpp`).
    ///
    /// Mirrors the reference command replay: apply `setoption name Threads` /
    /// `setoption name USI_Hash`, run the `usinewgame` equivalent once, reset the
    /// timer, then search each position through the ordinary coordinator path.
    /// Ends with one machine-parsable summary line. A parse failure is reported
    /// as an `info string` and runs nothing.
    fn handle_bench(&mut self, tokens: &[String]) -> io::Result<()> {
        self.finish_search_join();

        let current = bench::current_sfen(&self.pos);
        let config = match bench::parse_bench(tokens, &current) {
            Ok(c) => c,
            Err(e) => return self.info_string(&format!("bench: {e}")),
        };

        // The two option lines the reference emits (`benchmark.cpp`),
        // routed through the ordinary `setoption` path so the pool and TT resize
        // exactly as a host `setoption` would.
        self.handle_setoption("Threads", &config.threads.to_string())?;
        self.handle_setoption("USI_Hash", &config.tt_mb.to_string())?;

        // The `search_clear` the reference runs once before the positions — the
        // identical starting state that makes two runs report equal nodes.
        self.handle_usinewgame();

        // The reference times from after `search_clear`, excluding the clear.
        let start = Instant::now();
        let mut total_nodes: u64 = 0;
        let mut positions: u64 = 0;
        for fen in &config.fens {
            match parse_sfen(fen) {
                Ok(p) => self.pos = p,
                Err(e) => {
                    self.info_string(&format!("bench: skipping bad position `{fen}`: {e}"))?;
                    continue;
                }
            }
            positions += 1;
            total_nodes += self.bench_run_one(config.limits.clone())?;
        }

        // `+1` mirrors the reference's divide-by-zero guard (`usi.cpp`).
        let time_ms = start.elapsed().as_millis() as u64 + 1;
        let nps = 1000 * total_nodes / time_ms;
        self.info_string(&format!(
            "bench: positions={positions} nodes={total_nodes} time_ms={time_ms} nps={nps}"
        ))
    }

    fn handle_stop(&mut self) {
        // The worker emits its own `bestmove`; its state is reclaimed on the next
        // command that needs it, or on `quit`.
        if let Some(active) = &self.search {
            active.stop.store(true, Ordering::Relaxed);
        }
    }

    /// `gameover [win|lose|draw]`, treated exactly like `stop`
    /// (`usi.cpp`). An opponent resign during `go ponder` arrives from a
    /// GUI as `gameover` with no preceding `stop`; unhandled, pondering would
    /// never stop.
    fn handle_gameover(&mut self) {
        self.handle_stop();
    }

    /// `ponderhit`: the opponent played the predicted move (`usi.cpp`).
    ///
    /// Clearing the ponder flag both lets the pondering search continue under
    /// time management and releases a held book reply, whose coordinator wait
    /// loop polls the same flag.
    fn handle_ponderhit(&mut self) -> io::Result<()> {
        let stochastic = self.options.check("Stochastic_Ponder")
            && self.search.as_ref().is_some_and(|a| a.ponder.is_some());
        if stochastic {
            return self.stochastic_ponderhit();
        }

        if let Some(active) = &self.search {
            match &active.ponder {
                Some(p) => p.ponderhit(),
                // A stray `ponderhit` during e.g. `go infinite`: fall back to a
                // stop so any held reply is released.
                None => active.stop.store(true, Ordering::Relaxed),
            }
        }
        Ok(())
    }

    /// Stochastic_Ponder `ponderhit` (`usi.cpp`). The rewound search's
    /// output is suppressed before it is stopped, so exactly one `bestmove`
    /// reaches the GUI.
    fn stochastic_ponderhit(&mut self) -> io::Result<()> {
        if let Some(active) = &self.search {
            active.suppress.store(true, Ordering::Relaxed);
        }
        self.finish_search_join();

        let (sfen, moves) = self.last_position.clone();
        if let Some(pos) = build_position_from(&sfen, &moves) {
            self.pos = pos;
        }

        if let Some(mut go) = self.last_go.clone() {
            go.ponder = false;
            return self.handle_go(go);
        }
        Ok(())
    }

    fn handle_unknown(&mut self, line: &str) -> io::Result<()> {
        self.info_string(&format!("unknown command: {line}"))
    }

    fn handle_too_long(&mut self) -> io::Result<()> {
        self.info_string("command too long")
    }
}

/// Write one PV `info` line — the reference `on_update_full`
/// (`usi.cpp`). The reference's nondeterministic `nps` / `time` /
/// `hashfull` decorations are omitted so a fixed-depth `info` line is
/// reproducible, and `seldepth` / `multipv` are always emitted.
fn write_pv_info<W: Write + ?Sized>(w: &mut W, info: &PvInfo) -> io::Result<()> {
    let mut body = format!(
        "depth {} seldepth {} multipv {} score {}",
        info.depth,
        info.sel_depth,
        info.multipv,
        format_score(info.score),
    );
    match info.bound {
        PvBound::Lower => body.push_str(" lowerbound"),
        PvBound::Upper => body.push_str(" upperbound"),
        PvBound::Exact => {}
    }
    body.push_str(&format!(" nodes {}", info.nodes));
    if !info.pv.is_empty() {
        body.push_str(" pv");
        for m in &info.pv {
            body.push(' ');
            body.push_str(&format_usi_move(*m));
        }
    }
    Formatter::new(w).info(&body)
}

/// A [`PvSink`] writing each PV line straight to the shared USI output.
/// Installed on the main worker only, the reference `main_manager()->pv()`
/// owner; helpers and the fixed-depth path get no sink and emit nothing.
struct WriterPvSink<W: Write + Send> {
    writer: Arc<Mutex<W>>,
}

impl<W: Write + Send> PvSink for WriterPvSink<W> {
    fn emit(&mut self, info: &PvInfo) {
        let mut guard = self.writer.lock().unwrap_or_else(|e| e.into_inner());
        let _ = write_pv_info(&mut *guard, info);
    }
}

/// Apply the reference's `MaxMovesToDraw` remap (`yaneuraou-search.cpp`):
/// a set value of `0` means unlimited and becomes `100000`. The option itself
/// still reports `0` — only the search-side horizon uses the remapped value.
fn remap_max_moves_to_draw(option_value: i64) -> i32 {
    if option_value == 0 {
        100_000
    } else {
        option_value as i32
    }
}

/// The two book extensions the reference's name resolution knows about.
const BOOK_EXT_YBB: &str = "ybb";
const BOOK_EXT_DB: &str = "db";

/// True when `path` carries extension `ext`.
///
/// The reference compares the raw suffix and is therefore case-sensitive
/// (`book.cpp`); this test is case-insensitive, one convention for every
/// book path in the module.
fn has_book_ext(path: &Path, ext: &str) -> bool {
    path.extension()
        .is_some_and(|e| e.eq_ignore_ascii_case(ext))
}

/// Strip a trailing `.db` / `.ybb` from a book name, returning the stem
/// (`book_name_without_extension`, `book.cpp`).
///
/// Any other name — notably the `no_book` sentinel — yields `None`, the
/// reference's empty stem: this name has no numbered priority series.
fn book_name_without_extension(name: &Path) -> Option<PathBuf> {
    if has_book_ext(name, BOOK_EXT_DB) || has_book_ext(name, BOOK_EXT_YBB) {
        Some(name.with_extension(""))
    } else {
        None
    }
}

/// `<stem>-<index zero-padded to 3><extension>` (`book.cpp`). An index
/// past 999 grows past three digits, as the reference's padding loop does.
fn priority_book_filename(stem: &Path, index: usize, extension: &str) -> PathBuf {
    let mut name = stem.as_os_str().to_os_string();
    name.push(format!("-{index:03}.{extension}"));
    PathBuf::from(name)
}

/// Resolve priority book `index` for `base` (`resolve_priority_book_filename`,
/// `book.cpp`).
///
/// The primary extension is the base name's own; the secondary is the other one,
/// and the primary wins when both files exist, producing the reference's
/// `priority book file exists twice` notice as the second tuple element.
///
/// `None` means neither extension exists at this index, which ends the series.
fn resolve_priority_book_filename(base: &Path, index: usize) -> Option<(PathBuf, Option<String>)> {
    let stem = book_name_without_extension(base)?;

    let (primary_ext, secondary_ext) = if has_book_ext(base, BOOK_EXT_YBB) {
        (BOOK_EXT_YBB, BOOK_EXT_DB)
    } else {
        (BOOK_EXT_DB, BOOK_EXT_YBB)
    };
    let primary = priority_book_filename(&stem, index, primary_ext);
    let secondary = priority_book_filename(&stem, index, secondary_ext);

    if primary.exists() {
        let notice = secondary.exists().then(|| {
            format!(
                "priority book file exists twice. use : {}",
                primary.display()
            )
        });
        return Some((primary, notice));
    }
    if secondary.exists() {
        return Some((secondary, None));
    }
    None
}

/// The Multiple Book priority list for `base` (`book.cpp`):
/// `<stem>-000`, `<stem>-001`, … stopping at the first index where neither
/// extension exists — a gap ends the series, so a `-003` after a missing `-002`
/// is never reached — then the plain `base` last.
///
/// The second tuple element carries the notices the enumeration produced, in
/// list order, for the caller to emit.
fn book_names(base: &Path) -> (Vec<PathBuf>, Vec<String>) {
    let mut names = Vec::new();
    let mut notices = Vec::new();
    for index in 0.. {
        let Some((name, notice)) = resolve_priority_book_filename(base, index) else {
            break;
        };
        names.push(name);
        notices.extend(notice);
    }
    names.push(base.to_path_buf());
    (names, notices)
}

/// Resolve `<name>.db` whose file is absent to its `<name>.ybb` sibling
/// (`resolve_book_filename_with_ybb_fallback`, `book.cpp`). Returns the
/// original path when it exists, or when no `.ybb` sibling is present.
fn resolve_book_filename_with_ybb_fallback(requested: &Path) -> PathBuf {
    if requested.exists() {
        return requested.to_path_buf();
    }
    if has_book_ext(requested, BOOK_EXT_DB) {
        let sibling = requested.with_extension(BOOK_EXT_YBB);
        if sibling.exists() {
            return sibling;
        }
    }
    requested.to_path_buf()
}

/// Rebuild a [`Position`] from a parsed `position` command, returning `None` on
/// any parse or legality failure — the Stochastic_Ponder rewind / re-issue paths
/// (`usi.cpp`) need the reconstruction without the diagnostic
/// side effects of [`UsiDriver::handle_position`].
fn build_position_from(sfen: &PositionSfen, moves: &[String]) -> Option<Position> {
    let mut pos = match sfen {
        PositionSfen::StartPos => Position::startpos(),
        PositionSfen::Sfen(s) => parse_sfen(s).ok()?,
    };
    let mut legal_buf: Vec<Move> = Vec::new();
    for s in moves {
        let parsed = parse_usi_move(s, &pos).ok()?;
        legal_buf.clear();
        pos.generate_legal_all(&mut legal_buf);
        if !legal_buf.contains(&parsed) {
            return None;
        }
        pos.do_move(parsed);
    }
    Some(pos)
}

/// Emit a bare `bestmove <mv>` for the resign / declaration-win short-circuits,
/// which produce no `info` line. Best-effort: a broken pipe must not panic the
/// coordinator.
fn emit_bestmove<W: Write>(writer: &Arc<Mutex<W>>, mv: &str) {
    let mut guard = writer.lock().unwrap_or_else(|e| e.into_inner());
    let _ = Formatter::new(&mut *guard).bestmove(mv);
}

/// Emit one `info string <msg>` from the coordinator (best-effort).
fn emit_info_string<W: Write>(writer: &Arc<Mutex<W>>, msg: &str) {
    let mut guard = writer.lock().unwrap_or_else(|e| e.into_inner());
    let _ = Formatter::new(&mut *guard).info_string(msg);
}

/// A running keep-alive: a helper thread emitting a bare newline every
/// [`KEEP_ALIVE_TICKS_PER_NEWLINE`] polls so a GUI does not time out while the
/// heavy `isready` initialisation runs. Dropping the guard stops and joins the
/// thread, so the join runs whether the wrapped work returns normally or bails
/// out early via `?` — the reference's `SCOPE_EXIT` (engine.cpp).
///
/// Reference: `Engine::run_heavy_job` (engine.cpp).
struct KeepAlive {
    /// Set on drop to stop the helper (`thread_end`, engine.cpp).
    stop: Arc<AtomicBool>,
    /// `Some` until the guard is dropped; taken to join exactly once.
    handle: Option<JoinHandle<()>>,
}

impl KeepAlive {
    /// Spawn the helper thread and block until it has actually started. The
    /// heavy work must run *after* this returns so a CPU-bound job cannot delay
    /// the helper's first tick; the reference spins on a `thread_started` flag
    /// for the same reason (engine.cpp).
    fn spawn<W: Write + Send + 'static>(writer: Arc<Mutex<W>>, poll: Duration) -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        let started = Arc::new(AtomicBool::new(false));
        let handle = thread::spawn({
            let stop = Arc::clone(&stop);
            let started = Arc::clone(&started);
            move || {
                started.store(true, Ordering::Release);
                let mut count: u32 = 0;
                while !stop.load(Ordering::Acquire) {
                    thread::sleep(poll);
                    count += 1;
                    if count >= KEEP_ALIVE_TICKS_PER_NEWLINE {
                        count = 0;
                        // A bare newline with no `info string` prefix, holding
                        // the output lock so it cannot interleave mid-line with
                        // the heavy work's own output.
                        let mut guard = writer.lock().unwrap_or_else(|e| e.into_inner());
                        let _ = Formatter::new(&mut *guard).raw_line("");
                    }
                }
            }
        });
        // Finer than the reference's 100 ms spin (engine.cpp), so
        // wrapping a fast `isready` adds no perceptible latency.
        while !started.load(Ordering::Acquire) {
            thread::sleep(Duration::from_millis(1));
        }
        Self {
            stop,
            handle: Some(handle),
        }
    }
}

impl Drop for KeepAlive {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

/// Join book PV moves into a USI ` `-separated string.
fn pv_string(pv: &[Move]) -> String {
    pv.iter()
        .map(|m| format_usi_move(*m))
        .collect::<Vec<_>>()
        .join(" ")
}

/// Emit a book hit's output the way the reference does on `search_skipped`
/// (`yaneuraou-search.cpp`): one `info` line per surviving candidate,
/// then a final depth-0 `info` line and the `bestmove [ponder]`.
///
/// Under `go ponder` / `go infinite` the final line and `bestmove` are held
/// until `stop` or a `ponderhit` — the SKIP_SEARCH wait loop (`1162-1199`).
fn emit_book_hit<W: Write>(
    writer: &Arc<Mutex<W>>,
    hit: &BookHit,
    ponder: Option<&Arc<PonderSignal>>,
    infinite: bool,
    stop: &AtomicBool,
    suppress_bestmove: &AtomicBool,
) {
    // Emitted immediately, like the reference's in-probe isRoot block.
    {
        let mut guard = writer.lock().unwrap_or_else(|e| e.into_inner());
        let mut f = Formatter::new(&mut *guard);
        for line in &hit.info_lines {
            let body = format!(
                "depth {} seldepth 0 multipv {} score {} nodes 0 pv {}",
                line.depth,
                line.multipv,
                format_score(Value::from(line.score)),
                pv_string(&line.pv),
            );
            let _ = f.info(&body);
        }
    }

    while !stop.load(Ordering::Relaxed) && (ponder.is_some_and(|p| p.is_active()) || infinite) {
        std::thread::sleep(Duration::from_millis(1));
    }

    if suppress_bestmove.load(Ordering::Relaxed) {
        return;
    }

    let mut pv = format_usi_move(hit.best);
    if let Some(p) = hit.ponder {
        pv.push(' ');
        pv.push_str(&format_usi_move(p));
    }
    let mut bm = format_usi_move(hit.best);
    if let Some(p) = hit.ponder {
        bm.push_str(" ponder ");
        bm.push_str(&format_usi_move(p));
    }
    let mut guard = writer.lock().unwrap_or_else(|e| e.into_inner());
    let mut f = Formatter::new(&mut *guard);
    let _ = f.info(&format!(
        "depth 0 seldepth 0 multipv 1 score {} nodes 0 pv {pv}",
        format_score(Value::from(hit.value)),
    ));
    let _ = f.bestmove(&bm);
}

/// Everything one helper needs to run its own iterative deepening for a single
/// `go`. The position and the root-move list are per-helper copies, as the
/// reference `start_thinking` copies the root-move list to every worker.
struct HelperJob {
    search: Arc<Search>,
    tt: Arc<TranspositionTable>,
    pos: Position,
    root_moves: Vec<RootMove>,
    limit_depth: i32,
    stop: Arc<AtomicBool>,
    /// Per-worker node counters; this helper publishes to `node_slots[index]`.
    node_slots: Arc<Vec<AtomicU64>>,
    /// Per-worker best-move-change counters. The main worker folds every slot
    /// each iteration (`yaneuraou-search.cpp`).
    bmc_slots: Arc<Vec<AtomicU64>>,
    /// This helper's index into `node_slots` / `bmc_slots`; `>= 1`, since index
    /// 0 is the main worker.
    index: usize,
    entering_king: EnteringKingConfig,
    /// The `MaxMovesToDraw` horizon, already `0 → 100000` remapped.
    max_moves_to_draw: i32,
    /// The root-side draw contempt, already pawn-scaled.
    draw_contempt: Value,
    generate_all_legal_moves: bool,
    mate_mode: bool,
    /// The raw `MultiPV` option value, clamped to the legal-move count inside
    /// `run_worker`. Helpers run the MultiPV loop too but never emit.
    multi_pv: usize,
    /// This helper's node-shared correction / pawn tables. Stable across `go`s
    /// within a pool lifetime, so the helper attaches it once to its persistent
    /// per-worker tables.
    shared: Arc<SharedHistories>,
}

/// The state of one helper's coordination slot. The coordinator drives
/// `Parked → Assigned` and `Finished → Parked`; the helper thread drives
/// `Assigned → Running → Finished`. The pool sets `Exit`, only ever over
/// `Parked`: every teardown path first joins the coordinator, which has
/// returned every helper to `Parked`.
enum SlotState {
    Parked,
    /// A job the coordinator posted, not yet picked up. Boxed because a
    /// `HelperJob` holds a full root position, which would otherwise size every
    /// other variant.
    Assigned(Box<HelperJob>),
    Running,
    Finished(WorkerResult),
    Exit,
}

/// One persistent helper's coordination slot: a [`SlotState`] behind a mutex and
/// a condvar both the coordinator and the helper wait on.
struct HelperSlot {
    state: Mutex<SlotState>,
    cv: Condvar,
}

impl HelperSlot {
    fn new() -> Self {
        HelperSlot {
            state: Mutex::new(SlotState::Parked),
            cv: Condvar::new(),
        }
    }

    fn lock(&self) -> MutexGuard<'_, SlotState> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Post a search job to a parked helper and wake it.
    fn assign(&self, job: HelperJob) {
        *self.lock() = SlotState::Assigned(Box::new(job));
        self.cv.notify_all();
    }

    /// Block until the helper has finished, then take its result and return the
    /// slot to `Parked`.
    fn collect(&self) -> WorkerResult {
        let mut st = self.lock();
        loop {
            if matches!(&*st, SlotState::Finished(_)) {
                break;
            }
            st = self.cv.wait(st).unwrap_or_else(|e| e.into_inner());
        }
        match std::mem::replace(&mut *st, SlotState::Parked) {
            SlotState::Finished(r) => r,
            _ => unreachable!("slot must be Finished after the wait loop"),
        }
    }
}

/// The persistent helper thread body (the reference `idle_loop`). The
/// game-scoped histories persist across jobs; the pool is recreated to reset
/// them.
fn helper_loop(slot: Arc<HelperSlot>) {
    // Built lazily from the first job's `shared` handle, so the helper never
    // allocates a throwaway single-thread set before its node's real one
    // arrives. The handle is stable within a pool lifetime.
    let mut histories: Option<WorkerHistories> = None;
    loop {
        let job = {
            let mut st = slot.lock();
            loop {
                match &*st {
                    SlotState::Assigned(_) => break,
                    SlotState::Exit => return,
                    _ => st = slot.cv.wait(st).unwrap_or_else(|e| e.into_inner()),
                }
            }
            match std::mem::replace(&mut *st, SlotState::Running) {
                SlotState::Assigned(job) => *job,
                _ => unreachable!("slot must be Assigned to break the wait loop"),
            }
        };

        // A helper's control is stop-only — no deadlines, no node ceiling. The
        // reference runs `check_time` on the main worker alone (2403-2404).
        let histories_in =
            histories.unwrap_or_else(|| WorkerHistories::with_shared(Arc::clone(&job.shared)));
        let (result, reclaimed) = {
            let net = job.search.network();
            let mut qs = QSearch::with_histories(net, &job.tt, histories_in);
            qs.set_control(SearchControl {
                stop: Some(Arc::clone(&job.stop)),
                ponder: None,
                node_limit: None,
                time: None,
            });
            qs.set_node_tally(Arc::clone(&job.node_slots), job.index);
            qs.set_best_move_tally(Arc::clone(&job.bmc_slots), job.index);
            qs.set_entering_king(job.entering_king);
            qs.set_max_moves_to_draw(job.max_moves_to_draw);
            qs.set_draw_value(job.draw_contempt);
            qs.set_generate_all_legal_moves(job.generate_all_legal_moves);
            qs.set_mate_mode(job.mate_mode);
            qs.set_multi_pv(job.multi_pv);
            let result = qs.run_worker(&job.pos, job.root_moves, job.limit_depth);
            (result, qs.into_histories())
        };
        histories = Some(reclaimed);

        // Every shared `Arc` clone must be released BEFORE publishing
        // `Finished`. `finish_search_join` joins only the coordinator, yet
        // `isready` / `usinewgame` then call `Arc::get_mut(&mut self.tt)`
        // assuming sole ownership — which holds only once every re-parking
        // helper has dropped its `job.tt` clone too. Dropping after the
        // `Finished` store would leave a window where a descheduled helper still
        // holds a clone when `get_mut` runs, panicking the engine.
        drop(job.tt);
        drop(job.search);
        drop(job.stop);
        drop(job.node_slots);

        *slot.lock() = SlotState::Finished(result);
        slot.cv.notify_all();
    }
}

/// The engine's worker thread pool, modelling the reference `ThreadPool`
/// (`thread.*`): a main-worker slot plus `size − 1` persistent helper threads,
/// each parked in [`helper_loop`] until a `go` dispatches it a [`HelperJob`].
/// The main worker runs on the per-`go` coordinator thread, which dispatches to
/// and collects from these helper slots.
struct ThreadPool {
    /// One coordination slot per helper, shared with the coordinator via
    /// [`Self::helper_slots`].
    slots: Vec<Arc<HelperSlot>>,
    handles: Vec<JoinHandle<()>>,
}

impl ThreadPool {
    /// Build a pool of `size` slots with no NUMA binding — the unit tests' entry
    /// point; the driver uses [`Self::with_binding`].
    #[cfg(test)]
    fn new(size: usize) -> Self {
        Self::with_binding(size, None)
    }

    /// Build a pool of `size` slots with an optional NUMA binding plan. Each
    /// helper binds itself to its assigned node once at spawn, as the reference
    /// does at thread creation (`thread.cpp`).
    fn with_binding(size: usize, plan: Option<Arc<NumaBindPlan>>) -> Self {
        let mut pool = ThreadPool {
            slots: Vec::new(),
            handles: Vec::new(),
        };
        pool.set_with_binding(size, plan);
        pool
    }

    #[cfg(test)]
    fn set(&mut self, size: usize) {
        self.set_with_binding(size, None);
    }

    /// Resize to `size` slots. Like the reference `ThreadPool::set` this never
    /// diffs: it joins and destroys the current helpers, dropping their
    /// histories, then recreates the requested number. Callers must have joined
    /// any running search first, so every helper is parked when this runs.
    fn set_with_binding(&mut self, size: usize, plan: Option<Arc<NumaBindPlan>>) {
        self.shutdown();
        let size = size.max(1);
        for worker_id in 1..size {
            let slot = Arc::new(HelperSlot::new());
            let slot_for_thread = Arc::clone(&slot);
            let plan_for_thread = plan.clone();
            self.handles.push(std::thread::spawn(move || {
                if let Some(p) = &plan_for_thread
                    && !p.bound.is_empty()
                {
                    p.config
                        .bind_current_thread_to_numa_node(p.bound[worker_id]);
                }
                helper_loop(slot_for_thread);
            }));
            self.slots.push(slot);
        }
    }

    /// Ask every helper to exit and join it, leaving only the main slot.
    /// Idempotent. Every helper must be parked first, or a late `Finished` write
    /// would overwrite the `Exit`.
    fn shutdown(&mut self) {
        for slot in &self.slots {
            *slot.lock() = SlotState::Exit;
            slot.cv.notify_all();
        }
        for handle in self.handles.drain(..) {
            let _ = handle.join();
        }
        self.slots.clear();
    }

    fn size(&self) -> usize {
        self.slots.len() + 1
    }

    fn helper_slots(&self) -> Vec<Arc<HelperSlot>> {
        self.slots.iter().map(Arc::clone).collect()
    }
}

impl Drop for ThreadPool {
    fn drop(&mut self) {
        self.shutdown();
    }
}

/// The NUMA binding plan shared with the helper threads: the active layout plus
/// the worker → node assignment.
struct NumaBindPlan {
    config: NumaConfig,
    /// The worker → node assignment, index `i` being worker `i`. Empty means no
    /// helper binds.
    bound: Vec<NumaIndex>,
}

/// The `SysfsOptions` for the live machine. [`numa_config_from_policy`] takes
/// them as a parameter so tests can substitute a fixture tree.
fn real_sysfs_options() -> SysfsOptions {
    SysfsOptions {
        root: PathBuf::from("/sys"),
        allowed_cpus: attic_numa::startup_affinity().clone(),
        system_threads: attic_numa::system_threads(),
    }
}

/// Map a `NumaPolicy` option value to a [`NumaConfig`] (`engine.cpp`).
///
/// * `auto` / `system` → detect from the system respecting process affinity;
/// * `hardware` → detect ignoring process affinity;
/// * `none` → the default single all-threads node;
/// * anything else → a custom node string via [`NumaConfig::from_string`].
///
/// A custom string that fails to parse or yields zero nodes is an `Err`, where
/// the reference reaches `std::exit(EXIT_FAILURE)`.
fn numa_config_from_policy(policy: &str, opts: &SysfsOptions) -> Result<NumaConfig, String> {
    let cfg = match policy {
        "auto" | "system" => NumaConfig::from_sysfs(&DEFAULT_POLICY, true, opts),
        "hardware" => NumaConfig::from_sysfs(&DEFAULT_POLICY, false, opts),
        "none" => NumaConfig::default(),
        other => {
            let cfg = NumaConfig::from_string(other).map_err(|e| e.to_string())?;
            if cfg.num_numa_nodes() == 0 {
                return Err(format!("NumaPolicy `{other}` yields zero NUMA nodes"));
            }
            cfg
        }
    };
    Ok(cfg)
}

/// The worker → NUMA-node assignment for `requested` threads under `policy`
/// (`thread.cpp`).
///
/// `do_bind` is `false` for `none`, `suggests_binding_threads(requested)` for
/// `auto`, and `true` otherwise (`system` / `hardware` / a custom string). When
/// binding is off the assignment is empty; otherwise it is
/// [`NumaConfig::distribute_threads_among_numa_nodes`].
fn compute_numa_binding(config: &NumaConfig, policy: &str, requested: usize) -> Vec<NumaIndex> {
    let do_bind = match policy {
        "none" => false,
        "auto" => config.suggests_binding_threads(requested),
        _ => true,
    };
    if do_bind {
        config.distribute_threads_among_numa_nodes(requested)
    } else {
        Vec::new()
    }
}

/// Wrap a non-empty binding assignment into a shareable [`NumaBindPlan`]; an
/// empty assignment yields `None`, meaning no thread binds.
fn bind_plan(config: &NumaConfig, bound: &[NumaIndex]) -> Option<Arc<NumaBindPlan>> {
    if bound.is_empty() {
        None
    } else {
        Some(Arc::new(NumaBindPlan {
            config: config.clone(),
            bound: bound.to_vec(),
        }))
    }
}

/// Build the per-worker handles to the node-shared correction / pawn tables,
/// mirroring the reference per-node construction (`thread.cpp`). One
/// [`SharedHistories`] per distinct node, sized `next_power_of_two(count)`, and
/// one [`Arc`] returned per worker.
///
/// Rust's `usize::next_power_of_two` and the reference's own helper
/// (`thread.cpp`) agree on every `count >= 1`.
fn build_worker_shared(
    config: &NumaConfig,
    bound: &[NumaIndex],
    requested: usize,
) -> Vec<Arc<SharedHistories>> {
    let requested = requested.max(1);
    let counts = shared_node_counts(bound, requested);
    // With binding active each node's set is allocated and filled on that node,
    // so its pages first-touch there (`thread.cpp`).
    let binding_active = !bound.is_empty();

    let mut node_shared: std::collections::BTreeMap<NumaIndex, Arc<SharedHistories>> =
        std::collections::BTreeMap::new();
    for (&node, &count) in &counts {
        let thread_count = count.next_power_of_two();
        let arc = if binding_active {
            let mut built: Option<Arc<SharedHistories>> = None;
            config.execute_on_numa_node(node, || {
                built = Some(Arc::new(SharedHistories::new(thread_count)));
            });
            built.expect("execute_on_numa_node ran the closure")
        } else {
            Arc::new(SharedHistories::new(thread_count))
        };
        node_shared.insert(node, arc);
    }

    worker_nodes(bound, requested)
        .into_iter()
        .map(|node| Arc::clone(&node_shared[&node]))
        .collect()
}

/// The node → thread-count map for the shared-history construction
/// (`thread.cpp`). An empty `bound` puts every thread on node 0, as the
/// reference does.
fn shared_node_counts(
    bound: &[NumaIndex],
    requested: usize,
) -> std::collections::BTreeMap<NumaIndex, usize> {
    let mut counts: std::collections::BTreeMap<NumaIndex, usize> =
        std::collections::BTreeMap::new();
    if bound.is_empty() {
        counts.insert(0, requested.max(1));
    } else {
        for &node in bound {
            *counts.entry(node).or_insert(0) += 1;
        }
    }
    counts
}

/// The node each worker's shared table set belongs to (`search.h`):
/// `bound[i]` when binding is active, else node 0 for every worker.
fn worker_nodes(bound: &[NumaIndex], requested: usize) -> Vec<NumaIndex> {
    if bound.is_empty() {
        vec![0; requested.max(1)]
    } else {
        bound.to_vec()
    }
}

/// The `Arc` bookkeeping of [`UsiDriver::rebuild_networks`], factored out so it
/// is unit-testable without a loaded network or a live `/sys` tree, and generic
/// over the payload so a test can stand one in.
///
/// `sys_nodes[i]` is worker `i`'s *system* NUMA node; `rep_logical` maps each
/// distinct system node to a representative logical node to clone on. Without
/// replication every worker shares `source` and `clone_on_node` is never called.
/// Reusing an existing replica is sound because every instance is
/// byte-identical.
fn resolve_worker_networks<T>(
    source: &Arc<T>,
    replicas: &mut BTreeMap<NumaIndex, Arc<T>>,
    sys_nodes: &[NumaIndex],
    rep_logical: &BTreeMap<NumaIndex, NumaIndex>,
    requested: usize,
    replication_active: bool,
    mut clone_on_node: impl FnMut(NumaIndex, &Arc<T>) -> Arc<T>,
) -> Vec<Arc<T>> {
    if !replication_active {
        replicas.clear();
        return vec![Arc::clone(source); requested.max(1)];
    }

    replicas.retain(|sys, _| rep_logical.contains_key(sys));
    for (&sys, &logical) in rep_logical {
        replicas
            .entry(sys)
            .or_insert_with(|| clone_on_node(logical, source));
    }

    sys_nodes
        .iter()
        .map(|sys| Arc::clone(&replicas[sys]))
        .collect()
}

/// `"Available processors: " + cfg.to_string()` (`engine.cpp`).
fn numa_config_information_as_string(cfg: &NumaConfig) -> String {
    format!("Available processors: {cfg}")
}

/// The `(bound_count, cpus_in_node)` pairs per node (`thread.cpp`,
/// `engine.cpp`). Empty when nothing is bound; otherwise the pairs cover
/// every node up to `num_numa_nodes`, at zero past the highest bound one.
fn bound_thread_counts(cfg: &NumaConfig, bound: &[NumaIndex]) -> Vec<(usize, usize)> {
    if bound.is_empty() {
        return Vec::new();
    }
    let highest = bound.iter().copied().max().unwrap_or(0);
    let mut counts = vec![0usize; highest + 1];
    for &n in bound {
        counts[n] += 1;
    }
    let mut ratios: Vec<(usize, usize)> = Vec::new();
    for (n, &c) in counts.iter().enumerate() {
        ratios.push((c, cfg.num_cpus_in_numa_node(n)));
    }
    for n in (highest + 1)..cfg.num_numa_nodes() {
        ratios.push((0, cfg.num_cpus_in_numa_node(n)));
    }
    ratios
}

/// The `a/x:b/y:...` per-node `bound/total` string (`engine.cpp`); empty
/// when nothing is bound.
fn thread_binding_information_as_string(cfg: &NumaConfig, bound: &[NumaIndex]) -> String {
    bound_thread_counts(cfg, bound)
        .iter()
        .map(|(current, total)| format!("{current}/{total}"))
        .collect::<Vec<_>>()
        .join(":")
}

/// `"Using N thread[s]"`, plus `" with NUMA node thread binding: a/x:b/y..."`
/// when any thread is bound (`engine.cpp`).
fn thread_allocation_information_as_string(
    threads_size: usize,
    cfg: &NumaConfig,
    bound: &[NumaIndex],
) -> String {
    let mut s = format!(
        "Using {threads_size} {}",
        if threads_size > 1 {
            "threads"
        } else {
            "thread"
        }
    );
    let binding = thread_binding_information_as_string(cfg, bound);
    if binding.is_empty() {
        return s;
    }
    s.push_str(" with NUMA node thread binding: ");
    s.push_str(&binding);
    s
}

/// The bundle [`UsiDriver::handle_go`] hands its coordinator thread.
struct CoordinatorJob<W: Write + Send + 'static> {
    search: Arc<Search>,
    tt: Arc<TranspositionTable>,
    pos: Position,
    depth: i32,
    use_voting: bool,
    control: SearchControl,
    stop: Arc<AtomicBool>,
    histories: WorkerHistories,
    helper_slots: Vec<Arc<HelperSlot>>,
    /// Each helper's node-shared correction / pawn tables, aligned with
    /// `helper_slots`. The main worker's own handle lives inside `histories`.
    helper_shared: Vec<Arc<SharedHistories>>,
    /// Each helper's per-NUMA-node network replica, aligned with `helper_slots`.
    /// The main worker's own is `search`.
    helper_networks: Vec<Arc<Search>>,
    n_threads: usize,
    /// The node the coordinator binds itself to at the start of this `go`, or
    /// `None` when binding is inactive.
    numa_bind: Option<(NumaConfig, NumaIndex)>,
    book: Option<Arc<LoadedBook>>,
    book_config: BookConfig,
    /// `USI_OwnBook`: when off the book is never probed.
    own_book: bool,
    book_seed: u64,
    /// The shared `go ponder` signal. The coordinator's hold loop runs while it
    /// is active, withholding `bestmove` until a `ponderhit` clears it or `stop`
    /// fires.
    ponder: Option<Arc<PonderSignal>>,
    /// `limits.infinite`: hold the reply until `stop` regardless of the clock
    /// (`yaneuraou-search.cpp`).
    infinite: bool,
    /// When set the coordinator emits no `bestmove` and no final PV for this
    /// search (`usi.cpp`).
    suppress_bestmove: Arc<AtomicBool>,
    entering_king: EnteringKingConfig,
    /// The `MaxMovesToDraw` horizon, already `0 → 100000` remapped.
    max_moves_to_draw: i32,
    /// The root-side draw contempt `drawValueTable[REPETITION_DRAW][us]`,
    /// already pawn-scaled.
    draw_contempt: Value,
    /// A searched best score at or below `-resign_value` resigns.
    resign_value: Value,
    generate_all_legal_moves: bool,
    /// `go mate` mode: no early mate break, and a mate-found stop rule.
    mate_mode: bool,
    pv_config: PvOutputConfig,
    writer: Arc<Mutex<W>>,
}

/// What [`run_coordinated`] hands back: the main worker's histories, the
/// aggregate searched-node total (0 for the short-circuits, and the value
/// `bench` accumulates), and the time-management carry-forward, whose third
/// element is `None` for a short-circuited `go` — see
/// [`SearchState::time_state`].
struct CoordinatedOutcome {
    histories: WorkerHistories,
    nodes: u64,
    time_state: Option<(Value, Value, Option<f64>)>,
}

/// The time-management carry-forward for a SKIP_SEARCH short-circuit — a book
/// hit, declaration win, resign, or no legal move.
///
/// The reference falls straight through to the same `1249-1253` bookkeeping on
/// these paths (`yaneuraou-search.cpp`), where `rootMoves[0]` is the
/// unsearched default scoring `-VALUE_INFINITE` (`search.h`); a book
/// probe never writes `rootMoves`. Only `previousTimeReduction` is left
/// untouched, since its sole writer `iterative_deepening` did not run.
fn skip_search_carry() -> Option<(Value, Value, Option<f64>)> {
    Some((-VALUE_INFINITE, -VALUE_INFINITE, None))
}

fn run_coordinated<W: Write + Send + 'static>(job: CoordinatorJob<W>) -> CoordinatedOutcome {
    let CoordinatorJob {
        search,
        tt,
        pos,
        depth,
        use_voting,
        control,
        stop,
        histories,
        helper_slots,
        helper_shared,
        helper_networks,
        n_threads,
        numa_bind,
        book,
        book_config,
        own_book,
        book_seed,
        ponder,
        infinite,
        suppress_bestmove,
        entering_king,
        max_moves_to_draw,
        draw_contempt,
        resign_value,
        generate_all_legal_moves,
        mate_mode,
        pv_config,
        writer,
    } = job;
    let multi_pv = pv_config.multi_pv.max(1);

    // Before any search work. Idempotent across the per-`go` coordinator
    // respawns: the target node is stable until the next pool rebuild.
    if let Some((cfg, node)) = &numa_bind {
        cfg.bind_current_thread_to_numa_node(*node);
    }

    // One bump per `go`, before any helper starts
    // (`yaneuraou-search.cpp`), so the observable single-thread sequence is
    // the reference's: bump, then search.
    tt.new_search();

    // The reference `start_thinking`. The short-circuits below return before any
    // helper is dispatched, as `start_searching` exits before
    // `threads.start_searching()`.
    let root_moves = generate_root_moves(&pos, generate_all_legal_moves);
    if root_moves.is_empty() {
        emit_bestmove(&writer, "resign");
        return CoordinatedOutcome {
            histories,
            nodes: 0,
            time_state: skip_search_carry(),
        };
    }
    // Point and `None` rules yield `Move::win()`, emitted as the bare `win`
    // token; `TryRule` yields the actual king move onto the try square, which
    // must be emitted verbatim so the host plays it.
    if let Some(mv) = declaration_win(&pos, &entering_king) {
        if mv == Move::win() {
            emit_bestmove(&writer, "win");
        } else {
            emit_bestmove(&writer, &format_usi_move(mv));
        }
        return CoordinatedOutcome {
            histories,
            nodes: 0,
            time_state: skip_search_carry(),
        };
    }

    // Once, on the coordinator, before any helper starts: the on-the-fly read
    // path is not thread-safe by design (`book.h`,
    // `yaneuraou-search.cpp`).
    if own_book && let Some(loaded) = &book {
        let mut prng = Prng::new(book_seed);
        let probed = probe_book(
            &loaded.books,
            loaded.ignore_book_ply,
            &pos,
            &book_config,
            &mut prng,
        );
        for diag in &probed.diagnostics {
            emit_info_string(&writer, diag);
        }
        if let Some(hit) = probed.hit {
            emit_book_hit(
                &writer,
                &hit,
                ponder.as_ref(),
                infinite,
                &stop,
                &suppress_bestmove,
            );
            return CoordinatedOutcome {
                histories,
                nodes: 0,
                time_state: skip_search_carry(),
            };
        }
    }

    // Index 0 is the main worker, `1..` the helpers.
    let node_slots: Arc<Vec<AtomicU64>> =
        Arc::new((0..n_threads).map(|_| AtomicU64::new(0)).collect());
    // Each worker bumps its own slot at the root; the main worker folds them all
    // each iteration (`yaneuraou-search.cpp`).
    let bmc_slots: Arc<Vec<AtomicU64>> =
        Arc::new((0..n_threads).map(|_| AtomicU64::new(0)).collect());

    // Index `h` in `helper_slots` is worker `h + 1`.
    for (h, slot) in helper_slots.iter().enumerate() {
        slot.assign(HelperJob {
            search: Arc::clone(&helper_networks[h]),
            tt: Arc::clone(&tt),
            pos: pos.clone(),
            root_moves: root_moves.clone(),
            limit_depth: depth,
            stop: Arc::clone(&stop),
            node_slots: Arc::clone(&node_slots),
            bmc_slots: Arc::clone(&bmc_slots),
            index: h + 1,
            entering_king,
            max_moves_to_draw,
            draw_contempt,
            generate_all_legal_moves,
            mate_mode,
            multi_pv,
            shared: Arc::clone(&helper_shared[h]),
        });
    }

    // The main worker gets the full control and the node ceiling, and is the
    // only worker given a PV sink.
    let net = search.network();
    let mut qs = QSearch::with_histories(net, &tt, histories);
    qs.set_control(control);
    qs.set_node_tally(Arc::clone(&node_slots), 0);
    qs.set_best_move_tally(Arc::clone(&bmc_slots), 0);
    qs.set_entering_king(entering_king);
    qs.set_max_moves_to_draw(max_moves_to_draw);
    qs.set_draw_value(draw_contempt);
    qs.set_generate_all_legal_moves(generate_all_legal_moves);
    qs.set_mate_mode(mate_mode);
    qs.set_pv_output(
        pv_config,
        Box::new(WriterPvSink {
            writer: Arc::clone(&writer),
        }),
    );
    let main_result = qs.run_worker(&pos, root_moves, depth);

    // No `bestmove` while still pondering or under `go infinite` (the SKIP_SEARCH
    // wait loop, `yaneuraou-search.cpp`). A `ponderhit` clears the flag
    // mid-search, so this catches only the case where the search finished before
    // one arrived.
    while !stop.load(Ordering::Relaxed)
        && (ponder.as_ref().is_some_and(|p| p.is_active()) || infinite)
    {
        std::thread::sleep(Duration::from_millis(1));
    }

    stop.store(true, Ordering::Relaxed);
    let mut results: Vec<WorkerResult> = Vec::with_capacity(n_threads);
    results.push(main_result);
    for slot in &helper_slots {
        results.push(slot.collect());
    }

    // The reference `threads.nodes_searched()`.
    let total_nodes: u64 = results.iter().map(|r| r.nodes).sum();

    let chosen = if use_voting {
        let votes: Vec<WorkerVote> = results
            .iter()
            .map(|r| WorkerVote {
                score: r.best.score,
                pv0: r.best.pv[0],
                pv_len: r.best.pv.len(),
                completed_depth: r.completed_depth,
            })
            .collect();
        select_best_worker(&votes)
    } else {
        0
    };
    let chosen_result = &results[chosen];
    let mut best = chosen_result.best.clone();
    let mut pv_lines = chosen_result.pv_lines.clone();
    let completed_depth = chosen_result.completed_depth.max(1);
    let ponder_candidate = chosen_result.ponder_candidate;
    // `uciPvSent` is the main worker's flag, whatever worker is chosen.
    let mut uci_pv_sent = results[0].uci_pv_sent;

    // `yaneuraou-search.cpp`. The `timeReduction` comes from the
    // main worker even when the vote chose another.
    let out_best_previous_score = chosen_result.best.score;
    let out_best_previous_average_score = chosen_result.best.average_score;
    let out_previous_time_reduction = results[0].time_reduction;

    let ponder_before = best.pv.len();
    let mut work = pos.clone();
    qs.extract_ponder(&mut work, &mut best, ponder_candidate);
    let ponder_extended = best.pv.len() != ponder_before;

    // So the final-PV fallback re-emits the exact PV `bestmove [ponder]` plays.
    if let Some(line0) = pv_lines.get_mut(0) {
        *line0 = best.clone();
    }

    // A ponder-extended PV differs from what was emitted during the search
    // (1277-1280), and a non-main worker's PV was never emitted at all.
    if ponder_extended || chosen != 0 {
        uci_pv_sent = false;
    }

    // `ResignValue` (`yaneuraou-search.cpp`), decided before the final
    // PV output because a resign forces that PV out so the GUI sees the score the
    // decision was made on. The reference judges the printed `uciScore` in
    // centipawns, not the raw internal score, and maps an unset one — an
    // iteration aborted before any PV line was scored — to zero rather than
    // resigning outright.
    let resign_by_value = best.score != -VALUE_INFINITE && {
        let resign_score = if best.uci_score == -VALUE_INFINITE {
            0
        } else {
            best.uci_score
        };
        to_cp(resign_score) <= -resign_value
    };

    // Final PV output before `bestmove` (1300-1312). `pv_idx == lines.len()`
    // makes every line exact, matching the reference `pv()` after the MultiPV
    // loop (`worker.pvIdx == multiPV`).
    if !uci_pv_sent || resign_by_value {
        let n = pv_lines.len();
        let infos = qs.build_pv_infos(&pos, &pv_lines, n, completed_depth, n, total_nodes);
        let mut guard = writer.lock().unwrap_or_else(|e| e.into_inner());
        for info in &infos {
            let _ = write_pv_info(&mut *guard, info);
        }
    }

    let mut bm = format_usi_move(best.mv);
    if best.pv.len() >= 2 {
        bm.push_str(" ponder ");
        bm.push_str(&format_usi_move(best.pv[1]));
    }

    // Resigning replaces the whole reply (`1337-1342`), so it carries no ponder
    // move.
    if resign_by_value {
        bm = "resign".to_string();
    }

    // A Stochastic_Ponder teardown emits nothing (`usi.cpp`); the
    // re-issued `go` produces the single reply the GUI sees. The `time_state`
    // below is still returned, seeding the re-issue's side-flip continuity.
    if !suppress_bestmove.load(Ordering::Relaxed) {
        emit_bestmove(&writer, &bm);
    }

    CoordinatedOutcome {
        histories: qs.into_histories(),
        nodes: total_nodes,
        time_state: Some((
            out_best_previous_score,
            out_best_previous_average_score,
            // A real search produced a fresh `timeReduction`
            // (`yaneuraou-search.cpp`).
            Some(out_previous_time_reduction),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Drive a full canned session in-process and return everything written.
    /// `run` joins any search worker, so the buffer is the complete transcript.
    fn run_with(input: &str) -> String {
        let output = Arc::new(Mutex::new(Vec::<u8>::new()));
        let driver = UsiDriver::new(input.as_bytes(), Arc::clone(&output));
        driver.run().expect("driver run");
        let bytes = output.lock().expect("output lock").clone();
        String::from_utf8(bytes).expect("utf-8")
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn quit_returns_immediately() {
        assert_eq!(run_with("quit\n"), "");
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn eof_returns_ok() {
        assert_eq!(run_with(""), "");
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn isready_without_network_reports_load_failure() {
        // `eval/nn.bin` is absent in the test CWD, so the load fails.
        let out = run_with("isready\nquit\n");
        assert!(
            out.contains("info string eval load failed:"),
            "expected eval-load-failure notice, got: {out:?}"
        );
        assert!(
            !out.contains("readyok"),
            "readyok must not appear on a failed load: {out:?}"
        );
        // At the default cadence the first keep-alive tick never elapses.
        assert_eq!(
            bare_newline_count(&out),
            0,
            "a fast isready must emit no keep-alive newline: {out:?}"
        );
    }

    /// Count bare keep-alive newlines. Splitting on `\n` yields one trailing
    /// empty segment for the final terminator, which is not one.
    fn bare_newline_count(out: &str) -> usize {
        let parts: Vec<&str> = out.split('\n').collect();
        parts
            .iter()
            .take(parts.len().saturating_sub(1))
            .filter(|s| s.is_empty())
            .count()
    }

    /// Everything written through a shared test writer so far.
    fn shared_output(writer: &Arc<Mutex<Vec<u8>>>) -> String {
        String::from_utf8(writer.lock().unwrap_or_else(|e| e.into_inner()).clone()).unwrap()
    }

    /// Block until the shared writer holds at least `want` bare newlines.
    ///
    /// [`KeepAlive`] promises a newline every [`KEEP_ALIVE_TICKS_PER_NEWLINE`]
    /// polls, not once per fixed span of wall time: on a loaded machine each
    /// poll's sleep overruns, so the first newline can land arbitrarily late.
    /// Waiting for the newline rather than for a duration that ought to cover it
    /// keeps the wait honest — the deadline below is long enough that only a
    /// helper which never writes at all can trip it.
    fn wait_for_bare_newlines(writer: &Arc<Mutex<Vec<u8>>>, want: usize) {
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            let out = shared_output(writer);
            if bare_newline_count(&out) >= want {
                return;
            }
            assert!(
                Instant::now() < deadline,
                "keep-alive wrote fewer than {want} bare newline(s) before the deadline: {out:?}"
            );
            thread::sleep(Duration::from_millis(2));
        }
    }

    /// Write the heavy job's line through the shared writer, reporting the
    /// bare-newline count as it stood when the line went out. Counting under the
    /// same lock as the write stops a tick landing alongside it from being taken
    /// for one that preceded it.
    fn emit_busy_line(writer: &Arc<Mutex<Vec<u8>>>) -> usize {
        let mut guard = writer.lock().unwrap_or_else(|e| e.into_inner());
        let before = bare_newline_count(&String::from_utf8(guard.clone()).unwrap());
        Formatter::new(&mut *guard)
            .info_string("busy")
            .expect("a write to a Vec cannot fail");
        before
    }

    #[test]
    fn keep_alive_emits_bare_newline_through_shared_writer() {
        // The slowed job writes its own line through the same shared writer, so
        // this also asserts no keep-alive newline interleaves mid-line.
        let writer = Arc::new(Mutex::new(Vec::<u8>::new()));
        {
            // 1 ms poll → a bare newline every 50 polls.
            let keep_alive = KeepAlive::spawn(Arc::clone(&writer), Duration::from_millis(1));
            wait_for_bare_newlines(&writer, 1);
            // Wait for a newline after the job's line too, so the interleaving
            // check has helper output on both sides of it.
            let before = emit_busy_line(&writer);
            wait_for_bare_newlines(&writer, before + 1);
            drop(keep_alive);
        }
        let out = shared_output(&writer);

        assert!(
            bare_newline_count(&out) >= 1,
            "expected at least one bare keep-alive newline, got: {out:?}"
        );
        for line in out.split('\n') {
            assert!(
                line.is_empty() || line == "info string busy",
                "keep-alive newline interleaved with output: {out:?}"
            );
        }
        assert!(
            out.contains("info string busy\n"),
            "the heavy job's line must survive intact: {out:?}"
        );
    }

    #[test]
    fn keep_alive_stops_and_joins_when_job_finishes() {
        // The guard's Drop must stop and join the helper without hanging.
        let writer = Arc::new(Mutex::new(Vec::<u8>::new()));
        {
            // There is nothing to wait for here — the claim is that the helper
            // has *not* ticked — so the window before its first tick has to be
            // wide enough to survive the main thread being descheduled. The
            // default cadence puts that tick 5 s out while `Drop`'s join still
            // waits out at most one poll.
            let _keep_alive = KeepAlive::spawn(Arc::clone(&writer), KEEP_ALIVE_POLL_INTERVAL);
        }
        let out = shared_output(&writer);
        assert_eq!(
            bare_newline_count(&out),
            0,
            "a job that finishes before the first tick emits no newline: {out:?}"
        );
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn usinewgame_is_no_op() {
        assert_eq!(run_with("usinewgame\nquit\n"), "");
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn unknown_command_echoes_back() {
        assert_eq!(
            run_with("frobnicate\nquit\n"),
            "info string unknown command: frobnicate\n"
        );
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn setoption_happy_path_silent() {
        assert_eq!(run_with("setoption name USI_Hash value 256\nquit\n"), "");
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn setoption_unknown_option_rejected() {
        assert_eq!(
            run_with("setoption name Nonexistent value foo\nquit\n"),
            "info string option Nonexistent rejected: unknown option\n"
        );
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn setoption_bad_int_rejected() {
        assert_eq!(
            run_with("setoption name USI_Hash value not-a-number\nquit\n"),
            "info string option USI_Hash rejected: value `not-a-number` is not an integer\n"
        );
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn position_startpos_silent() {
        assert_eq!(run_with("position startpos\nquit\n"), "");
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn position_sfen_startpos_silent() {
        let sfen = attic_state::STARTPOS_SFEN;
        assert_eq!(run_with(&format!("position sfen {sfen}\nquit\n")), "");
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn position_startpos_moves_silent() {
        assert_eq!(run_with("position startpos moves 7g7f\nquit\n"), "");
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn position_sfen_malformed_emits_info_string() {
        let out = run_with("position sfen not-a-board b - 1\nquit\n");
        assert!(
            out.starts_with("info string position parse error:"),
            "unexpected output: {out:?}",
        );
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn position_with_illegal_move_emits_info_string() {
        let out = run_with("position startpos moves 1a1b\nquit\n");
        assert!(
            out.starts_with("info string illegal move:"),
            "unexpected output: {out:?}",
        );
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn position_with_pseudo_legal_but_illegal_move_emits_info_string() {
        // Syntactically valid, but the pawn on 7g cannot jump to 5g.
        let out = run_with("position startpos moves 7g5g\nquit\n");
        assert!(
            out.starts_with("info string illegal move:"),
            "unexpected output: {out:?}",
        );
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn position_parse_error_leaves_prior_state_intact() {
        // The malformed line must not clobber the position the legal move built.
        let session = "position startpos moves 7g7f\n\
                       position sfen not-a-board b - 1\n\
                       go\n\
                       quit\n";
        let out = run_with(session);
        assert!(
            out.contains("info string position parse error:"),
            "missing parse-error info string in: {out}"
        );
        let bestmoves: Vec<&str> = out.lines().filter(|l| l.starts_with("bestmove ")).collect();
        assert_eq!(
            bestmoves.len(),
            1,
            "expected one bestmove line, got {bestmoves:?}"
        );
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn go_without_network_resigns_with_notice() {
        // No successful `isready`, so no network is loaded.
        let out = run_with("go\nquit\n");
        assert!(
            out.contains("info string no eval network loaded; run isready"),
            "expected the no-network notice, got: {out:?}"
        );
        let bestmoves: Vec<&str> = out.lines().filter(|l| l.starts_with("bestmove ")).collect();
        assert_eq!(bestmoves, vec!["bestmove resign"]);
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn go_with_limit_subtokens_still_emits_one_bestmove() {
        let session = "go depth 8 wtime 60000 btime 60000 byoyomi 5000\nquit\n";
        let out = run_with(session);
        let bestmoves: Vec<&str> = out.lines().filter(|l| l.starts_with("bestmove ")).collect();
        assert_eq!(bestmoves.len(), 1);
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn stop_is_silent() {
        assert_eq!(run_with("stop\nquit\n"), "");
        assert_eq!(run_with("go\nstop\nquit\n"), run_with("go\nquit\n"));
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn setoption_threads_emits_allocation_line() {
        // `NumaPolicy none` keeps binding off, so the line is deterministic
        // across machines.
        let out = run_with(
            "setoption name NumaPolicy value none\n\
             setoption name Threads value 1\n\
             setoption name Threads value 4\n\
             setoption name Threads value 2\n\
             quit\n",
        );
        assert!(out.contains("info string Using 1 thread\n"), "{out}");
        assert!(out.contains("info string Using 4 threads\n"), "{out}");
        assert!(out.contains("info string Using 2 threads\n"), "{out}");
        assert!(
            !out.contains("with NUMA node thread binding"),
            "none policy must not bind: {out}"
        );
    }

    /// A miniature 2-node sysfs fixture tree with no L3 cache dirs. The caller
    /// cleans the returned root up.
    fn write_two_node_sysfs_fixture() -> PathBuf {
        let mut root = std::env::temp_dir();
        root.push(format!("numa367_optmap_{}", std::process::id()));
        let node = root.join("devices/system/node");
        std::fs::create_dir_all(node.join("node0")).expect("mkdir node0");
        std::fs::create_dir_all(node.join("node1")).expect("mkdir node1");
        std::fs::write(node.join("online"), "0-1\n").expect("write online");
        std::fs::write(node.join("node0/cpulist"), "0-1\n").expect("write node0");
        std::fs::write(node.join("node1/cpulist"), "2-3\n").expect("write node1");
        root
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn numa_policy_option_mapping() {
        let root = write_two_node_sysfs_fixture();
        let opts = SysfsOptions {
            root: root.clone(),
            allowed_cpus: [0usize, 1, 2, 3].into_iter().collect(),
            system_threads: 4,
        };

        let auto = numa_config_from_policy("auto", &opts).unwrap();
        assert_eq!(auto.num_numa_nodes(), 2);
        assert!(!auto.is_custom_affinity());
        let system = numa_config_from_policy("system", &opts).unwrap();
        assert_eq!(system.num_numa_nodes(), 2);
        assert!(!system.is_custom_affinity());

        let hardware = numa_config_from_policy("hardware", &opts).unwrap();
        assert_eq!(hardware.num_numa_nodes(), 2);
        assert!(hardware.is_custom_affinity());

        let none = numa_config_from_policy("none", &opts).unwrap();
        assert_eq!(none.num_numa_nodes(), 1);
        assert!(!none.is_custom_affinity());

        let custom = numa_config_from_policy("0-3:4-7", &opts).unwrap();
        assert_eq!(custom.num_numa_nodes(), 2);
        assert!(custom.is_custom_affinity());

        // A duplicate CPU, then a zero-node config.
        assert!(numa_config_from_policy("0,0", &opts).is_err());
        assert!(numa_config_from_policy("", &opts).is_err());

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn info_strings_exact_formats() {
        let cfg = NumaConfig::from_string("0-3,8:16-31").unwrap();
        assert_eq!(
            numa_config_information_as_string(&cfg),
            "Available processors: 0-3,8:16-31"
        );

        assert_eq!(
            thread_allocation_information_as_string(1, &cfg, &[]),
            "Using 1 thread"
        );
        assert_eq!(
            thread_allocation_information_as_string(2, &cfg, &[]),
            "Using 2 threads"
        );

        let two = NumaConfig::from_string("0-1:2-3").unwrap();
        assert_eq!(
            thread_allocation_information_as_string(2, &two, &[0, 1]),
            "Using 2 threads with NUMA node thread binding: 1/2:1/2"
        );

        // Trailing nodes are extended with `0/total` (`engine.cpp`).
        let three = NumaConfig::from_string("0-1:2-3:4-5").unwrap();
        assert_eq!(
            thread_allocation_information_as_string(2, &three, &[0, 0]),
            "Using 2 threads with NUMA node thread binding: 2/2:0/2:0/2"
        );
    }

    #[test]
    fn thread_pool_new_sizes_to_main_plus_helpers() {
        let pool = ThreadPool::new(4);
        assert_eq!(pool.size(), 4, "4 slots = 1 main + 3 helpers");
        let single = ThreadPool::new(1);
        assert_eq!(single.size(), 1, "1 slot = main only, no helpers");
    }

    #[test]
    fn thread_pool_set_rebuilds_without_leaking() {
        // Only the slot count is observable here; the no-leak property is what
        // `shutdown`'s join gives.
        let mut pool = ThreadPool::new(2);
        assert_eq!(pool.size(), 2);
        pool.set(1);
        assert_eq!(pool.size(), 1);
        pool.set(4);
        assert_eq!(pool.size(), 4);
        pool.set(4);
        assert_eq!(pool.size(), 4, "a same-size set still rebuilds cleanly");
    }

    #[test]
    fn thread_pool_zero_is_clamped_to_one() {
        // The driver never passes 0, but `size − 1` must not underflow.
        let pool = ThreadPool::new(0);
        assert_eq!(pool.size(), 1);
    }

    #[test]
    fn shared_node_counts_unbound_and_bound() {
        // Unbound, every thread counts on node 0 (`thread.cpp`).
        let c = shared_node_counts(&[], 5);
        assert_eq!(c.len(), 1);
        assert_eq!(c[&0], 5);

        let c = shared_node_counts(&[0, 1, 0, 1, 0], 5);
        assert_eq!(c[&0], 3);
        assert_eq!(c[&1], 2);
    }

    #[test]
    fn worker_nodes_selects_each_worker_node() {
        assert_eq!(worker_nodes(&[], 3), vec![0, 0, 0]);
        assert_eq!(worker_nodes(&[1, 0, 1], 3), vec![1, 0, 1]);
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn build_worker_shared_unbound_shares_one_set() {
        let cfg = NumaConfig::from_string("0-3").unwrap();
        let ws = build_worker_shared(&cfg, &[], 4);
        assert_eq!(ws.len(), 4, "one handle per worker");
        for i in 1..4 {
            assert!(
                Arc::ptr_eq(&ws[0], &ws[i]),
                "unbound: every worker points at one shared set"
            );
        }
        assert_eq!(ws[0].thread_count(), 4);
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn build_worker_shared_unbound_rounds_thread_count_up() {
        let cfg = NumaConfig::from_string("0-3").unwrap();
        let ws = build_worker_shared(&cfg, &[], 3);
        assert_eq!(ws.len(), 3);
        assert_eq!(ws[0].thread_count(), 4);
        let ws1 = build_worker_shared(&cfg, &[], 1);
        assert_eq!(ws1.len(), 1);
        assert_eq!(ws1[0].thread_count(), 1);
    }

    // The reference's SKIP_SEARCH bookkeeping
    // (`yaneuraou-search.cpp`), driven through the real
    // `finish_search_join` with the carry every short-circuit returns.
    #[test]
    #[cfg_attr(miri, ignore)]
    fn skip_search_carry_updates_scores_and_ply_but_not_time_reduction() {
        assert_eq!(
            skip_search_carry(),
            Some((-VALUE_INFINITE, -VALUE_INFINITE, None)),
            "the short-circuit carry is the -VALUE_INFINITE sentinel with no tr"
        );

        let output = Arc::new(Mutex::new(Vec::<u8>::new()));
        let mut driver = UsiDriver::new(&b""[..], Arc::clone(&output));
        driver.best_previous_score = 123;
        driver.best_previous_average_score = 456;
        driver.previous_time_reduction = 0.42;
        driver.last_game_ply = 7;

        let handle = std::thread::spawn(|| SearchState {
            histories: WorkerHistories::new(),
            time_state: skip_search_carry(),
        });
        driver.search = Some(ActiveSearch {
            handle,
            stop: Arc::new(AtomicBool::new(false)),
            ponder: None,
            suppress: Arc::new(AtomicBool::new(false)),
            game_ply: 20,
        });
        driver.finish_search_join();

        assert_eq!(driver.best_previous_score, -VALUE_INFINITE);
        assert_eq!(driver.best_previous_average_score, -VALUE_INFINITE);
        assert_eq!(
            driver.last_game_ply, 20,
            "ply advances to the short-circuit's"
        );
        assert_eq!(
            driver.previous_time_reduction, 0.42,
            "previousTimeReduction is left untouched on a short-circuit"
        );
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn real_search_carry_overwrites_time_reduction() {
        let output = Arc::new(Mutex::new(Vec::<u8>::new()));
        let mut driver = UsiDriver::new(&b""[..], Arc::clone(&output));
        driver.previous_time_reduction = 0.42;

        let handle = std::thread::spawn(|| SearchState {
            histories: WorkerHistories::new(),
            time_state: Some((10, 20, Some(1.25))),
        });
        driver.search = Some(ActiveSearch {
            handle,
            stop: Arc::new(AtomicBool::new(false)),
            ponder: None,
            suppress: Arc::new(AtomicBool::new(false)),
            game_ply: 3,
        });
        driver.finish_search_join();

        assert_eq!(driver.best_previous_score, 10);
        assert_eq!(driver.best_previous_average_score, 20);
        assert_eq!(driver.previous_time_reduction, 1.25);
        assert_eq!(driver.last_game_ply, 3);
    }

    // These exercise `resolve_worker_networks` with a stand-in payload, so they
    // need neither a loaded network nor multi-node hardware.

    /// A `clone_on_node` stand-in recording each `(system, logical)` build. Each
    /// value is distinct, so no replica is accidentally `ptr_eq` to the source or
    /// to another.
    fn counting_cloner<'a>(
        calls: &'a std::cell::RefCell<Vec<(NumaIndex, NumaIndex)>>,
        next: &'a std::cell::Cell<u32>,
    ) -> impl FnMut(NumaIndex, &Arc<u32>) -> Arc<u32> + 'a {
        move |logical, src| {
            calls.borrow_mut().push((**src as NumaIndex, logical));
            let v = next.get();
            next.set(v + 1);
            Arc::new(v)
        }
    }

    #[test]
    fn resolve_networks_inactive_shares_source_and_builds_nothing() {
        let source = Arc::new(1000u32);
        let mut replicas: BTreeMap<NumaIndex, Arc<u32>> = BTreeMap::new();
        replicas.insert(9, Arc::new(7));
        let calls = std::cell::RefCell::new(Vec::new());
        let next = std::cell::Cell::new(0);

        let workers = resolve_worker_networks(
            &source,
            &mut replicas,
            &[],
            &BTreeMap::new(),
            3,
            false,
            counting_cloner(&calls, &next),
        );

        assert_eq!(workers.len(), 3);
        for w in &workers {
            assert!(
                Arc::ptr_eq(w, &source),
                "unbound worker must share the base"
            );
        }
        assert!(replicas.is_empty(), "stale replicas dropped");
        assert!(calls.borrow().is_empty(), "no on-node clone when inactive");
    }

    #[test]
    fn resolve_networks_shares_one_copy_within_a_system_node() {
        let source = Arc::new(1u32);
        let mut replicas = BTreeMap::new();
        let calls = std::cell::RefCell::new(Vec::new());
        let next = std::cell::Cell::new(0);
        let sys_nodes = [0usize, 0];
        let rep_logical: BTreeMap<NumaIndex, NumaIndex> = [(0usize, 0usize)].into_iter().collect();

        let workers = resolve_worker_networks(
            &source,
            &mut replicas,
            &sys_nodes,
            &rep_logical,
            2,
            true,
            counting_cloner(&calls, &next),
        );

        assert_eq!(workers.len(), 2);
        assert!(
            Arc::ptr_eq(&workers[0], &workers[1]),
            "same system node → one shared copy"
        );
        assert!(
            !Arc::ptr_eq(&workers[0], &source),
            "replica is a fresh copy"
        );
        assert_eq!(replicas.len(), 1);
        assert_eq!(calls.borrow().len(), 1, "exactly one on-node clone");
    }

    #[test]
    fn resolve_networks_distinct_copies_across_system_nodes() {
        let source = Arc::new(1u32);
        let mut replicas = BTreeMap::new();
        let calls = std::cell::RefCell::new(Vec::new());
        let next = std::cell::Cell::new(0);
        let sys_nodes = [0usize, 1];
        let rep_logical: BTreeMap<NumaIndex, NumaIndex> =
            [(0usize, 0usize), (1, 1)].into_iter().collect();

        let workers = resolve_worker_networks(
            &source,
            &mut replicas,
            &sys_nodes,
            &rep_logical,
            2,
            true,
            counting_cloner(&calls, &next),
        );

        assert!(
            !Arc::ptr_eq(&workers[0], &workers[1]),
            "distinct system nodes → distinct copies"
        );
        assert!(!Arc::ptr_eq(&workers[0], &source));
        assert!(!Arc::ptr_eq(&workers[1], &source));
        assert_eq!(replicas.len(), 2);
        assert_eq!(calls.borrow().len(), 2);
    }

    #[test]
    fn resolve_networks_reuses_unchanged_layout_without_recloning() {
        // The reference rebuilds byte-identical copies; these are kept instead.
        let source = Arc::new(1u32);
        let mut replicas = BTreeMap::new();
        let calls = std::cell::RefCell::new(Vec::new());
        let next = std::cell::Cell::new(0);
        let sys_nodes = [0usize, 1];
        let rep_logical: BTreeMap<NumaIndex, NumaIndex> =
            [(0usize, 0usize), (1, 1)].into_iter().collect();

        let first = resolve_worker_networks(
            &source,
            &mut replicas,
            &sys_nodes,
            &rep_logical,
            2,
            true,
            counting_cloner(&calls, &next),
        );
        assert_eq!(calls.borrow().len(), 2);

        let second = resolve_worker_networks(
            &source,
            &mut replicas,
            &sys_nodes,
            &rep_logical,
            2,
            true,
            counting_cloner(&calls, &next),
        );
        assert_eq!(calls.borrow().len(), 2, "no new clone on unchanged layout");
        assert!(Arc::ptr_eq(&first[0], &second[0]));
        assert!(Arc::ptr_eq(&first[1], &second[1]));
    }

    #[test]
    fn resolve_networks_drops_stale_replicas_when_a_system_node_leaves() {
        let source = Arc::new(1u32);
        let mut replicas = BTreeMap::new();
        let calls = std::cell::RefCell::new(Vec::new());
        let next = std::cell::Cell::new(0);

        let _ = resolve_worker_networks(
            &source,
            &mut replicas,
            &[0usize, 1],
            &[(0usize, 0usize), (1, 1)].into_iter().collect(),
            2,
            true,
            counting_cloner(&calls, &next),
        );
        assert_eq!(replicas.len(), 2);
        let node0_before = Arc::clone(&replicas[&0]);

        let workers = resolve_worker_networks(
            &source,
            &mut replicas,
            &[0usize],
            &[(0usize, 0usize)].into_iter().collect(),
            1,
            true,
            counting_cloner(&calls, &next),
        );
        assert_eq!(replicas.len(), 1, "system node 1's replica dropped");
        assert!(replicas.contains_key(&0));
        assert!(
            Arc::ptr_eq(&workers[0], &node0_before),
            "the surviving replica is reused, not rebuilt"
        );
    }

    #[test]
    fn resolve_networks_reload_from_new_source_replaces_the_set() {
        let source_a = Arc::new(1u32);
        let mut replicas = BTreeMap::new();
        let calls = std::cell::RefCell::new(Vec::new());
        let next = std::cell::Cell::new(0);
        let sys_nodes = [0usize, 1];
        let rep_logical: BTreeMap<NumaIndex, NumaIndex> =
            [(0usize, 0usize), (1, 1)].into_iter().collect();

        let old = resolve_worker_networks(
            &source_a,
            &mut replicas,
            &sys_nodes,
            &rep_logical,
            2,
            true,
            counting_cloner(&calls, &next),
        );

        // A reload installs a new `LoadedEval` with an empty replica map.
        let source_b = Arc::new(2u32);
        let mut fresh_replicas = BTreeMap::new();
        let new = resolve_worker_networks(
            &source_b,
            &mut fresh_replicas,
            &sys_nodes,
            &rep_logical,
            2,
            true,
            counting_cloner(&calls, &next),
        );

        for old_w in &old {
            for new_w in &new {
                assert!(!Arc::ptr_eq(old_w, new_w), "old copies are not reused");
            }
        }
        assert_eq!(fresh_replicas.len(), 2);
    }

    /// A fresh empty directory under `$TMPDIR`; the caller removes it.
    fn book_name_fixture_dir(tag: &str) -> PathBuf {
        static CTR: AtomicU64 = AtomicU64::new(0);
        let n = CTR.fetch_add(1, Ordering::Relaxed);
        let root = std::env::temp_dir().join(format!(
            "engine-book-names-{}-{tag}-{n}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).expect("mkdir book-name fixture");
        root
    }

    /// Touch an empty file; only existence matters to the name resolution.
    fn touch(path: &Path) {
        std::fs::write(path, b"").expect("touch");
    }

    /// The file names of a resolved list, without their directories.
    fn file_names(paths: &[PathBuf]) -> Vec<String> {
        paths
            .iter()
            .map(|p| p.file_name().unwrap().to_string_lossy().into_owned())
            .collect()
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn priority_series_stops_at_the_first_gap_and_appends_the_base_last() {
        let dir = book_name_fixture_dir("series");
        let base = dir.join("user_book1.ybb");
        touch(&base);
        touch(&dir.join("user_book1-000.ybb"));
        touch(&dir.join("user_book1-001.ybb"));
        touch(&dir.join("user_book1-003.ybb"));

        let (names, notices) = book_names(&base);
        assert_eq!(
            file_names(&names),
            vec!["user_book1-000.ybb", "user_book1-001.ybb", "user_book1.ybb",]
        );
        assert!(
            notices.is_empty(),
            "no duplicate-extension notice: {notices:?}"
        );

        let stem = book_name_without_extension(&base).expect("stem");
        assert_eq!(
            priority_book_filename(&stem, 7, "ybb")
                .file_name()
                .unwrap()
                .to_string_lossy(),
            "user_book1-007.ybb"
        );
        assert_eq!(
            priority_book_filename(&stem, 42, "db")
                .file_name()
                .unwrap()
                .to_string_lossy(),
            "user_book1-042.db"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn no_numbered_files_yields_just_the_base_name() {
        let dir = book_name_fixture_dir("bare");
        let base = dir.join("user_book1.ybb");
        touch(&base);
        let (names, notices) = book_names(&base);
        assert_eq!(names, vec![base]);
        assert!(notices.is_empty());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn no_book_has_an_empty_series() {
        let dir = book_name_fixture_dir("nobook");
        let base = dir.join("no_book");
        // A stray numbered file cannot start a series: the sentinel has no
        // `.db` / `.ybb` extension, so its stem is empty.
        touch(&dir.join("no_book-000.ybb"));
        assert_eq!(book_name_without_extension(&base), None);
        assert!(resolve_priority_book_filename(&base, 0).is_none());
        let (names, notices) = book_names(&base);
        assert_eq!(names, vec![base], "only the base name, no series");
        assert!(notices.is_empty());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn cross_extension_resolution_prefers_the_bases_own_extension() {
        let dir = book_name_fixture_dir("crossext");

        let ybb_base = dir.join("user_book1.ybb");
        touch(&ybb_base);
        touch(&dir.join("user_book1-000.ybb"));
        touch(&dir.join("user_book1-000.db"));
        let (names, notices) = book_names(&ybb_base);
        assert_eq!(
            file_names(&names),
            vec!["user_book1-000.ybb", "user_book1.ybb"]
        );
        assert_eq!(
            notices,
            vec![format!(
                "priority book file exists twice. use : {}",
                dir.join("user_book1-000.ybb").display()
            )]
        );

        let db_base = dir.join("user_book2.db");
        touch(&dir.join("user_book2-000.ybb"));
        touch(&dir.join("user_book2-000.db"));
        let (names, notices) = book_names(&db_base);
        assert_eq!(
            file_names(&names),
            vec!["user_book2-000.db", "user_book2.db"]
        );
        assert_eq!(
            notices,
            vec![format!(
                "priority book file exists twice. use : {}",
                dir.join("user_book2-000.db").display()
            )]
        );

        // A `.ybb` base with only a `.db` at index 0 resolves to the `.db`, which
        // `reload_book` then routes to the unsupported-format path.
        let solo = dir.join("user_book3.ybb");
        touch(&dir.join("user_book3-000.db"));
        let (names, notices) = book_names(&solo);
        assert_eq!(
            file_names(&names),
            vec!["user_book3-000.db", "user_book3.ybb"]
        );
        assert!(notices.is_empty(), "one file only → no notice: {notices:?}");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn db_names_still_resolve_to_their_ybb_sibling() {
        // The `BookFile` combo advertises only `.ybb` names, so this fallback is
        // unreachable from the option surface, yet `reload_book` routes every
        // enumerated name through it.
        let dir = book_name_fixture_dir("fallback");

        let ybb = dir.join("user_book1.ybb");
        touch(&ybb);
        assert_eq!(
            resolve_book_filename_with_ybb_fallback(&dir.join("user_book1.db")),
            ybb
        );

        let db = dir.join("user_book2.db");
        touch(&db);
        touch(&dir.join("user_book2.ybb"));
        assert_eq!(resolve_book_filename_with_ybb_fallback(&db), db);

        // No `.ybb` sibling: the caller reports the load failure.
        let missing = dir.join("user_book3.db");
        assert_eq!(resolve_book_filename_with_ybb_fallback(&missing), missing);

        let absent_ybb = dir.join("user_book4.ybb");
        assert_eq!(
            resolve_book_filename_with_ybb_fallback(&absent_ybb),
            absent_ybb
        );

        let _ = std::fs::remove_dir_all(&dir);
    }
}
