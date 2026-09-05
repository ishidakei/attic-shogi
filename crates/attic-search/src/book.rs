//! Opening-book probe and move-selection policy, ported from
//! `BookMoveSelector::probe_impl` / `find_in_books` / `MemoryBook::find`
//! (`book/book.cpp`). It covers the BookOptions V2 profile and Multiple Book,
//! where an ordered list is consulted in priority order and the first hit wins
//! (`book.cpp`); the `.ybb8` rework is out of scope.
//!
//! The raw reader speaks only packed keys and move fragments, so everything
//! needing [`Position`] or movegen knowledge lives here, above Storage. The
//! option surface is registered in Protocol; this module takes an already-parsed
//! snapshot and an injectable [`Prng`], so that selection is deterministic under
//! test.

use attic_state::{
    Color, Move, PackedSfen, Piece, PieceKind, Position, Square, flip_move16, sfen_pack,
};
use attic_storage::{Book, BookMove};

/// Piece kinds in Apery hand order (Pawn..=Rook), used when swapping hands to
/// build the color-flipped position.
const HAND_KINDS: [PieceKind; 7] = [
    PieceKind::Pawn,
    PieceKind::Lance,
    PieceKind::Knight,
    PieceKind::Silver,
    PieceKind::Gold,
    PieceKind::Bishop,
    PieceKind::Rook,
];

/// A small, seedable xorshift64* PRNG for book move selection. Random-bit
/// parity with the reference is explicitly *not* required.
#[derive(Clone, Debug)]
pub struct Prng(u64);

impl Prng {
    /// Seed the generator. A zero seed would make xorshift degenerate, so it is
    /// nudged to a fixed nonzero constant.
    pub fn new(seed: u64) -> Self {
        Prng(if seed == 0 {
            0x9E37_79B9_7F4A_7C15
        } else {
            seed
        })
    }

    /// A fresh process-entropy seed, mixing the wall clock, an ASLR-varied stack
    /// address, and a process-monotonic counter so that two seeds drawn in the
    /// same nanosecond still differ. The reference's default `PRNG()` mixes the
    /// same three kinds of source (`misc.h`).
    pub fn random_seed() -> u64 {
        use std::sync::atomic::{AtomicU64, Ordering};
        use std::time::{SystemTime, UNIX_EPOCH};

        static SEQ: AtomicU64 = AtomicU64::new(0);
        let seq = SEQ.fetch_add(1, Ordering::Relaxed);

        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0);

        // ASLR makes a stack address vary across process runs.
        let anchor = 0u8;
        let addr = &anchor as *const u8 as u64;

        nanos ^ addr.rotate_left(32) ^ seq.wrapping_mul(0x9E37_79B9_7F4A_7C15)
    }

    /// A generator seeded from process entropy. Tests inject a fixed seed with
    /// [`Self::new`] instead.
    pub fn from_entropy() -> Self {
        Prng::new(Self::random_seed())
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// A uniform draw in `[0, n)`. Returns `0` when `n == 0`.
    pub fn rand(&mut self, n: u64) -> u64 {
        if n == 0 { 0 } else { self.next_u64() % n }
    }
}

/// A snapshot of the book-relevant USI options for one probe.
///
/// `ignore_book_ply` is deliberately absent: the reference captures it at load
/// time, so it travels with the loaded book rather than the per-`go` config.
///
/// `book_options_v2` selects between the V1 fields and the V2 `*_black_*` /
/// `*_white_*` pairs, as the reference resolves the option *name* at probe time
/// from the root side to move (`book.cpp`).
#[derive(Clone, Debug)]
pub struct BookConfig {
    /// Whether the options were registered under `BOOK_OPTIONS=V2`.
    pub book_options_v2: bool,
    /// `NarrowBook` — drop moves under 10% adoption; V1 only.
    pub narrow_book: bool,
    /// `BookMoves` — maximum game ply at which the book is consulted.
    pub book_moves: i64,
    /// `BookIgnoreRate` — percentage chance (0..=100) to ignore the book.
    pub ignore_rate: i64,
    /// `BookEvalDiff` — allowed eval gap below the best move (V1 only).
    pub eval_diff: i64,
    /// `BookEvalBlackDiff` — the V2 eval gap when Black is to move.
    pub eval_black_diff: i64,
    /// `BookEvalWhiteDiff` — the V2 eval gap when White is to move.
    pub eval_white_diff: i64,
    /// `BookEvalBlackLimit` — eval floor when Black is to move.
    pub eval_black_limit: i64,
    /// `BookEvalWhiteLimit` — eval floor when White is to move.
    pub eval_white_limit: i64,
    /// `BookDepthLimit` — clear the whole entry if the best move's depth is
    /// below this; V1 only.
    pub depth_limit: i64,
    /// `BookDepthBlackLimit` — the V2 depth floor when Black is to move.
    pub depth_black_limit: i64,
    /// `BookDepthWhiteLimit` — the V2 depth floor when White is to move.
    pub depth_white_limit: i64,
    /// `ConsiderBookMoveCount` — weight selection by adoption count; V1 only.
    pub consider_move_count: bool,
    /// `BookPvMoves` — how many plies of book PV to build for the info lines.
    pub pv_moves: i64,
    /// `FlippedBook` — also probe the 180°-rotated position on a miss.
    pub flipped_book: bool,
}

impl BookConfig {
    /// `NarrowBook`, forced false under V2 (`book.cpp`).
    fn narrow_book_active(&self) -> bool {
        !self.book_options_v2 && self.narrow_book
    }

    /// `ConsiderBookMoveCount`, forced false under V2 (`book.cpp`).
    fn consider_move_count_active(&self) -> bool {
        !self.book_options_v2 && self.consider_move_count
    }

    /// The depth-floor option consulted at the root, as a `(name, value)` pair.
    /// Under V2 the *name* is side-to-move dependent (`book.cpp`), and
    /// it is what the info string reports.
    fn depth_limit_for(&self, stm: Color) -> (&'static str, i64) {
        match (self.book_options_v2, stm) {
            (false, _) => ("BookDepthLimit", self.depth_limit),
            (true, Color::Black) => ("BookDepthBlackLimit", self.depth_black_limit),
            (true, Color::White) => ("BookDepthWhiteLimit", self.depth_white_limit),
        }
    }

    /// The eval-gap option actually consulted at the root, likewise
    /// (`book.cpp`).
    fn eval_diff_for(&self, stm: Color) -> (&'static str, i64) {
        match (self.book_options_v2, stm) {
            (false, _) => ("BookEvalDiff", self.eval_diff),
            (true, Color::Black) => ("BookEvalBlackDiff", self.eval_black_diff),
            (true, Color::White) => ("BookEvalWhiteDiff", self.eval_white_diff),
        }
    }

    /// The per-side eval floor and its option name, already side-to-move
    /// dependent under V1 (`book.cpp`).
    fn eval_limit_for(&self, stm: Color) -> (&'static str, i64) {
        if stm == Color::Black {
            ("BookEvalBlackLimit", self.eval_black_limit)
        } else {
            ("BookEvalWhiteLimit", self.eval_white_limit)
        }
    }
}

/// One `info` line's worth of book-PV data for a surviving candidate. Exact
/// string equality with the reference is not required.
#[derive(Clone, Debug)]
pub struct BookInfoLine {
    /// 1-based candidate index (`multipv`).
    pub multipv: usize,
    /// Stored eval of the candidate.
    pub score: i16,
    /// Stored search depth of the candidate.
    pub depth: u16,
    /// Book PV: the candidate move followed by best-move continuations.
    pub pv: Vec<Move>,
}

/// A successful book probe: the chosen move, an optional ponder move, its score,
/// and the per-candidate info lines to emit at the root.
#[derive(Clone, Debug)]
pub struct BookHit {
    /// The selected (legal, widened) book move.
    pub best: Move,
    /// The ponder move (legal in the post-best position), if any.
    pub ponder: Option<Move>,
    /// Stored eval of the selected move.
    pub value: i16,
    /// One info line per surviving candidate, best-first.
    pub info_lines: Vec<BookInfoLine>,
}

/// The outcome of a probe: an optional hit plus the diagnostic bodies the
/// reference would have emitted, which the driver prefixes with `info string `.
#[derive(Clone, Debug, Default)]
pub struct BookProbeResult {
    /// The chosen move, or `None` on a miss.
    pub hit: Option<BookHit>,
    /// Diagnostic message bodies to surface as `info string` lines.
    pub diagnostics: Vec<String>,
}

/// A book move already widened to a legal [`Move`], carrying its stored stats.
#[derive(Clone, Copy, Debug)]
struct Candidate {
    mv: Move,
    value: i16,
    depth: u16,
    count: u16,
}

/// Probe `books` for `pos` and select a move under `config` — `probe_impl` for
/// the root case.
///
/// `books` is the Multiple Book priority list, and every lookup on this path
/// goes through [`find_in_books`], which returns the **first** non-empty hit and
/// never merges across books (`book.cpp`).
///
/// The `USI_OwnBook` gate is the caller's responsibility. Returns a miss for
/// every early-out the reference takes.
pub fn probe_book(
    books: &[Book],
    ignore_book_ply: bool,
    pos: &Position,
    config: &BookConfig,
    prng: &mut Prng,
) -> BookProbeResult {
    let mut result = BookProbeResult::default();

    // Random skip (`BookIgnoreRate`).
    if config.ignore_rate > 0 && (prng.rand(100) as i64) < config.ignore_rate {
        return result;
    }

    // Past the book horizon (`game_ply > BookMoves`).
    if i64::from(pos.ply()) > config.book_moves {
        return result;
    }

    let Some(raw) = find_in_books(books, ignore_book_ply, pos, config.flipped_book) else {
        return result;
    };
    if raw.is_empty() {
        return result;
    }

    // An entry that does not widen against the root's legal moves is an illegal
    // move, and is dropped with a diagnostic.
    let mut legal: Vec<Move> = Vec::new();
    pos.generate_legal_all(&mut legal);
    let mut candidates: Vec<Candidate> = Vec::with_capacity(raw.len());
    for bm in &raw {
        match widen(&legal, bm.move16) {
            Some(mv) => candidates.push(Candidate {
                mv,
                value: bm.value,
                depth: bm.depth,
                count: bm.count,
            }),
            None => result.diagnostics.push(format!(
                "Error! : Illegal Move In Book DB : move16 = 0x{:04x}",
                bm.move16
            )),
        }
    }
    if candidates.is_empty() {
        return result;
    }

    // Built from the post-legality candidate set, before any narrow, eval or
    // depth filtering removes moves.
    let info_lines = build_info_lines(books, ignore_book_ply, pos, config, &candidates);

    let move_count_total: u64 = candidates.iter().map(|c| u64::from(c.count)).sum();
    let has_move_count = move_count_total != 0;

    if config.narrow_book_active() && has_move_count {
        let before = candidates.len();
        candidates.retain(|c| f64::from(c.count) / move_count_total as f64 >= 0.1);
        if candidates.len() != before {
            result.diagnostics.push(format!(
                "NarrowBook : {before} moves to {} moves.",
                candidates.len()
            ));
        }
    }
    if candidates.is_empty() {
        return result;
    }

    // The depth floor clears the **whole entry** when the best move's depth is
    // below the limit — a per-position skip, not a per-move filter.
    let stm = pos.side_to_move();
    let (depth_limit_name, depth_limit) = config.depth_limit_for(stm);
    if depth_limit != 0 && i64::from(candidates[0].depth) < depth_limit {
        result.diagnostics.push(format!(
            "{depth_limit_name} is lower than the depth of this node."
        ));
        candidates.clear();
    } else {
        let best_value = i64::from(candidates[0].value);
        let (eval_diff_name, eval_diff) = config.eval_diff_for(stm);
        let value_limit1 = best_value - eval_diff;
        let (limit_name, value_limit2) = config.eval_limit_for(stm);
        let value_limit = value_limit1.max(value_limit2);
        let before = candidates.len();
        candidates.retain(|c| i64::from(c.value) >= value_limit);
        if candidates.len() != before {
            result.diagnostics.push(format!(
                "{eval_diff_name} = {eval_diff} , {limit_name} = {value_limit2} , {before} moves to {} moves.",
                candidates.len()
            ));
        }
    }
    if candidates.is_empty() {
        return result;
    }

    // A uniform baseline, refined to an adoption-count-weighted pick when
    // `ConsiderBookMoveCount` is on.
    let mut best = candidates[prng.rand(candidates.len() as u64) as usize];
    if config.consider_move_count_active() {
        let sum: u64 = candidates.iter().map(|c| u64::from(c.count)).sum();
        let mut acc: u64 = 0;
        for c in &candidates {
            let w = if sum == 0 { 1 } else { u64::from(c.count) };
            acc += w;
            if acc != 0 && prng.rand(acc) < w {
                best = *c;
            }
        }
    }

    // A `.ybb` stores no ponder move, so one is derived by probing the position
    // after the best move.
    let ponder = ponder_move(books, ignore_book_ply, pos, config.flipped_book, best.mv);

    result.hit = Some(BookHit {
        best: best.mv,
        ponder,
        value: best.value,
        info_lines,
    });
    result
}

/// The ponder move for `best`: probe the post-`best` position, take its first
/// sorted move, and keep it only if it is legal there.
fn ponder_move(
    books: &[Book],
    ignore_book_ply: bool,
    pos: &Position,
    flipped: bool,
    best: Move,
) -> Option<Move> {
    let mut child = pos.clone();
    child.do_move(best);
    let moves = find_in_books(books, ignore_book_ply, &child, flipped)?;
    let first = moves.first()?;
    let mut legal: Vec<Move> = Vec::new();
    child.generate_legal_all(&mut legal);
    widen(&legal, first.move16)
}

/// Build the per-candidate book-PV info lines (the `isRoot` multipv block).
fn build_info_lines(
    books: &[Book],
    ignore_book_ply: bool,
    pos: &Position,
    config: &BookConfig,
    candidates: &[Candidate],
) -> Vec<BookInfoLine> {
    let pv_moves = config.pv_moves.max(1) as usize;
    candidates
        .iter()
        .enumerate()
        .map(|(i, c)| BookInfoLine {
            multipv: i + 1,
            score: c.value,
            depth: c.depth,
            pv: build_pv(
                books,
                ignore_book_ply,
                pos,
                config.flipped_book,
                c.mv,
                pv_moves,
            ),
        })
        .collect()
}

/// Build a book PV starting with `first`: walk best (first-sorted) book moves
/// forward for up to `pv_moves` plies. A deterministic simplification of the
/// reference `pv_builder` (which force-hits and may randomize); exact PV content
/// is not gated, only its USI shape.
fn build_pv(
    books: &[Book],
    ignore_book_ply: bool,
    pos: &Position,
    flipped: bool,
    first: Move,
    pv_moves: usize,
) -> Vec<Move> {
    let mut pv = vec![first];
    let mut work = pos.clone();
    work.do_move(first);
    let mut remaining = pv_moves.saturating_sub(1);
    while remaining > 0 {
        let Some(moves) = find_in_books(books, ignore_book_ply, &work, flipped) else {
            break;
        };
        let Some(next) = moves.first() else {
            break;
        };
        let mut legal: Vec<Move> = Vec::new();
        work.generate_legal_all(&mut legal);
        let Some(mv) = widen(&legal, next.move16) else {
            break;
        };
        pv.push(mv);
        work.do_move(mv);
        remaining -= 1;
    }
    pv
}

/// `BookMoveSelector::find_in_books` (`book.cpp`): consult the books in
/// priority order and return the FIRST non-empty hit.
///
/// A hit in an upper book never falls through to a lower one and results are
/// never merged — so a position present in book 0 is answered by book 0 alone,
/// even when a lower book stores different (or more) moves for it. The
/// reference skips a book whose `find` returns null *or* an empty move list;
/// both map to "keep walking" here.
///
/// The per-book flipped-position fallback lives inside [`find_in_book`] (the
/// reference does it inside `MemoryBook::find`, `book.cpp`), so a
/// flipped hit in
/// book 0 also stops the walk.
fn find_in_books(
    books: &[Book],
    ignore_book_ply: bool,
    pos: &Position,
    flipped: bool,
) -> Option<Vec<BookMove>> {
    for book in books {
        if let Some(moves) = find_in_book(book, ignore_book_ply, pos, flipped)
            && !moves.is_empty()
        {
            return Some(moves);
        }
    }
    None
}

/// `MemoryBook::find` for the `.ybb` case: pack `pos`, probe; on a miss and when
/// `flipped` is set, probe the color-flipped position and flip its moves back.
/// The returned list is sorted best-first (count desc, then value desc).
fn find_in_book(
    book: &Book,
    ignore_book_ply: bool,
    pos: &Position,
    flipped: bool,
) -> Option<Vec<BookMove>> {
    let packed = sfen_pack(pos);
    if let Ok(Some(mut moves)) = book.probe(&packed, pos.ply(), ignore_book_ply) {
        sort_book_moves(&mut moves);
        return Some(moves);
    }
    if flipped {
        let fpacked = flipped_packed(pos);
        if let Ok(Some(mut moves)) = book.probe(&fpacked, pos.ply(), ignore_book_ply) {
            for m in &mut moves {
                m.move16 = flip_move16(m.move16);
            }
            sort_book_moves(&mut moves);
            return Some(moves);
        }
    }
    None
}

/// Stable sort matching `BookMove::operator<`: move_count descending, ties
/// broken by value descending. `.ybb` stores no counts (all 0), so this is
/// value-descending with stored order preserved on ties.
fn sort_book_moves(moves: &mut [BookMove]) {
    moves.sort_by(|a, b| b.count.cmp(&a.count).then(b.value.cmp(&a.value)));
}

/// Find the legal move whose `move16` equals `m16` (the widen-gate: an accepted
/// book move is always a member of the generated legal-move set — a raw `Move`
/// is never constructed from stored bits).
fn widen(legal: &[Move], m16: u16) -> Option<Move> {
    legal.iter().copied().find(|m| m.move16() == m16)
}

/// Pack the 180°-rotated, color-swapped image of `pos` — the Rust equivalent of
/// `PackedSfen::flipped()`. Board squares map by `Flip(sq) = 80 - sq` with the
/// piece's color inverted, the two hands swap, and the side to move flips.
fn flipped_packed(pos: &Position) -> PackedSfen {
    let mut f = Position::empty();
    for idx in 0..Square::COUNT as u8 {
        let sq = Square::from_index(idx).expect("idx < COUNT");
        if let Some(p) = pos.board().get(sq) {
            let fsq = Square::from_index(80 - idx).expect("80 - idx < COUNT");
            f.board_mut().set(
                fsq,
                Some(Piece {
                    kind: p.kind,
                    color: p.color.flip(),
                    promoted: p.promoted,
                }),
            );
        }
    }
    for kind in HAND_KINDS {
        for _ in 0..pos.hand(Color::Black).count(kind) {
            f.hand_mut(Color::White).increment(kind);
        }
        for _ in 0..pos.hand(Color::White).count(kind) {
            f.hand_mut(Color::Black).increment(kind);
        }
    }
    f.set_side_to_move(pos.side_to_move().flip());
    sfen_pack(&f)
}

#[cfg(test)]
mod tests {
    use super::*;
    use attic_state::{parse_sfen, parse_usi_move};

    // Two independently constructed entropy-seeded streams differ (the time /
    // address / counter mix), mirroring the reference's per-process `PRNG()` seed.
    #[test]
    #[cfg_attr(miri, ignore)]
    fn entropy_seeds_differ_across_constructions() {
        let a = Prng::random_seed();
        let b = Prng::random_seed();
        assert_ne!(a, b, "two entropy seeds must differ");

        let mut pa = Prng::from_entropy();
        let mut pb = Prng::from_entropy();
        let sa: Vec<u64> = (0..4).map(|_| pa.rand(u64::MAX)).collect();
        let sb: Vec<u64> = (0..4).map(|_| pb.rand(u64::MAX)).collect();
        assert_ne!(sa, sb, "two entropy-seeded streams must diverge");
    }

    // Seed injection is fully reproducible — the determinism tests and the
    // production seed-injection path both rely on this.
    #[test]
    fn injected_seed_reproduces_exactly() {
        let seq = |seed| {
            let mut p = Prng::new(seed);
            (0..8).map(|_| p.rand(1000)).collect::<Vec<_>>()
        };
        assert_eq!(seq(12345), seq(12345), "same seed => identical sequence");
        assert_ne!(seq(1), seq(2), "different seeds => different sequences");
    }

    const STARTPOS_B: &str = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1";
    const STARTPOS_W: &str = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL w - 1";

    /// One raw move: `(move16, value, depth)`. `.ybb` stores no per-move count.
    type RawMove = (u16, i16, u16);
    /// One record: `(packed key, ply, moves)`.
    type RawRecord = (PackedSfen, u16, Vec<RawMove>);

    /// Build a well-formed `.ybb` byte image (depth-carrying) from records,
    /// sorted by packed key as the format requires.
    fn build_ybb(records: &[RawRecord]) -> Vec<u8> {
        const MAGIC: &[u8; 16] = b"YANE-BINBOOK-V1\0";
        let mut sorted = records.to_vec();
        sorted.sort_by_key(|r| r.0);

        let mut header = Vec::new();
        header.extend_from_slice(MAGIC);
        header.extend_from_slice(&(sorted.len() as u64).to_le_bytes());
        header.extend_from_slice(&1u64.to_le_bytes()); // flags: move-depth present

        let mut index = Vec::new();
        let mut moves = Vec::new();
        for (packed, ply, mvs) in &sorted {
            let moves_offset = moves.len() as u64;
            index.extend_from_slice(packed);
            index.extend_from_slice(&moves_offset.to_le_bytes());
            index.extend_from_slice(&ply.to_le_bytes());
            index.extend_from_slice(&(mvs.len() as u16).to_le_bytes());
            for (m, v, d) in mvs {
                moves.extend_from_slice(&m.to_le_bytes());
                moves.extend_from_slice(&(*v as u16).to_le_bytes());
                moves.extend_from_slice(&d.to_le_bytes());
            }
        }
        let mut out = header;
        out.extend_from_slice(&index);
        out.extend_from_slice(&moves);
        out
    }

    fn pos(sfen: &str) -> Position {
        parse_sfen(sfen).expect("valid sfen")
    }

    fn m16(sfen: &str, usi: &str) -> u16 {
        let p = pos(sfen);
        parse_usi_move(usi, &p).expect("valid move").move16()
    }

    /// A permissive V1 config: no filtering, book always consulted.
    fn cfg() -> BookConfig {
        BookConfig {
            book_options_v2: false,
            narrow_book: false,
            book_moves: 10000,
            ignore_rate: 0,
            eval_diff: 99999,
            eval_black_diff: 99999,
            eval_white_diff: 99999,
            eval_black_limit: -99999,
            eval_white_limit: -99999,
            depth_limit: 0,
            depth_black_limit: 0,
            depth_white_limit: 0,
            consider_move_count: false,
            pv_moves: 8,
            flipped_book: false,
        }
    }

    /// The same, under the V2 profile.
    fn cfg_v2() -> BookConfig {
        BookConfig {
            book_options_v2: true,
            ..cfg()
        }
    }

    /// A three-move startpos book: 7g7f(v100,d20), 2g2f(v50,d18), 6i7h(v5,d16).
    fn three_move_book() -> Book {
        let key = sfen_pack(&pos(STARTPOS_B));
        let recs = vec![(
            key,
            1u16,
            vec![
                (m16(STARTPOS_B, "7g7f"), 100, 20),
                (m16(STARTPOS_B, "2g2f"), 50, 18),
                (m16(STARTPOS_B, "6i7h"), 5, 16),
            ],
        )];
        Book::from_memory(build_ybb(&recs)).expect("valid ybb")
    }

    /// A three-move White-root book: 3c3d(v100,d20), 8c8d(v50,d18), 4a3b(v5,d16).
    fn three_move_book_white() -> Book {
        let key = sfen_pack(&pos(STARTPOS_W));
        let recs = vec![(
            key,
            1u16,
            vec![
                (m16(STARTPOS_W, "3c3d"), 100, 20),
                (m16(STARTPOS_W, "8c8d"), 50, 18),
                (m16(STARTPOS_W, "4a3b"), 5, 16),
            ],
        )];
        Book::from_memory(build_ybb(&recs)).expect("valid ybb")
    }

    fn usi(m: Move) -> String {
        attic_state::format_usi_move(m)
    }

    /// A one-element priority list, for the tests that exercise a single
    /// loaded book.
    fn one(book: &Book) -> &[Book] {
        std::slice::from_ref(book)
    }

    #[test]
    fn eval_diff_zero_keeps_only_the_best_value_move() {
        let book = three_move_book();
        let mut c = cfg();
        c.eval_diff = 0; // only the top value (100) survives → deterministic pick.
        // Any seed picks the sole survivor.
        for seed in [1u64, 2, 12345, 0xDEAD_BEEF] {
            let mut prng = Prng::new(seed);
            let r = probe_book(one(&book), false, &pos(STARTPOS_B), &c, &mut prng);
            let hit = r.hit.expect("hit");
            assert_eq!(usi(hit.best), "7g7f", "seed {seed}");
            assert_eq!(hit.value, 100);
        }
    }

    #[test]
    fn wide_eval_diff_selects_within_the_surviving_set() {
        let book = three_move_book();
        let c = cfg(); // eval_diff 99999 → all three survive.
        let allowed = ["7g7f", "2g2f", "6i7h"];
        for seed in [1u64, 7, 99, 4321] {
            let mut prng = Prng::new(seed);
            let hit = probe_book(one(&book), false, &pos(STARTPOS_B), &c, &mut prng)
                .hit
                .expect("hit");
            assert!(allowed.contains(&usi(hit.best).as_str()), "seed {seed}");
        }
        // The info block reports every surviving candidate, best-first.
        let mut prng = Prng::new(1);
        let hit = probe_book(one(&book), false, &pos(STARTPOS_B), &c, &mut prng)
            .hit
            .expect("hit");
        assert_eq!(hit.info_lines.len(), 3);
        assert_eq!(hit.info_lines[0].multipv, 1);
        assert_eq!(usi(hit.info_lines[0].pv[0]), "7g7f");
        assert_eq!(hit.info_lines[0].score, 100);
        assert_eq!(hit.info_lines[0].depth, 20);
    }

    #[test]
    fn per_side_eval_limit_can_filter_everything() {
        let book = three_move_book();
        let mut c = cfg();
        // Black to move; a floor above every stored value drops all moves.
        c.eval_black_limit = 200;
        let mut prng = Prng::new(1);
        let r = probe_book(one(&book), false, &pos(STARTPOS_B), &c, &mut prng);
        assert!(r.hit.is_none(), "all moves below the per-side floor → miss");
    }

    #[test]
    fn depth_limit_clears_the_whole_entry() {
        let book = three_move_book();
        let mut c = cfg();
        // Best move depth is 20; a limit above it clears the entry (per-position).
        c.depth_limit = 25;
        let mut prng = Prng::new(1);
        assert!(
            probe_book(one(&book), false, &pos(STARTPOS_B), &c, &mut prng)
                .hit
                .is_none()
        );
        // A limit at/below the best depth keeps the entry.
        c.depth_limit = 20;
        let mut prng = Prng::new(1);
        assert!(
            probe_book(one(&book), false, &pos(STARTPOS_B), &c, &mut prng)
                .hit
                .is_some()
        );
    }

    // --- BookOptions V2. ---

    #[test]
    fn v2_resolves_option_names_by_side_to_move() {
        let v1 = cfg();
        assert_eq!(v1.depth_limit_for(Color::Black).0, "BookDepthLimit");
        assert_eq!(v1.depth_limit_for(Color::White).0, "BookDepthLimit");
        assert_eq!(v1.eval_diff_for(Color::Black).0, "BookEvalDiff");
        assert_eq!(v1.eval_diff_for(Color::White).0, "BookEvalDiff");

        let v2 = cfg_v2();
        assert_eq!(v2.depth_limit_for(Color::Black).0, "BookDepthBlackLimit");
        assert_eq!(v2.depth_limit_for(Color::White).0, "BookDepthWhiteLimit");
        assert_eq!(v2.eval_diff_for(Color::Black).0, "BookEvalBlackDiff");
        assert_eq!(v2.eval_diff_for(Color::White).0, "BookEvalWhiteDiff");

        // The eval FLOOR is already per-side under V1 and unchanged under V2.
        for c in [&v1, &v2] {
            assert_eq!(c.eval_limit_for(Color::Black).0, "BookEvalBlackLimit");
            assert_eq!(c.eval_limit_for(Color::White).0, "BookEvalWhiteLimit");
        }
    }

    #[test]
    fn v2_forces_narrow_book_and_consider_move_count_off() {
        let mut v1 = cfg();
        v1.narrow_book = true;
        v1.consider_move_count = true;
        assert!(v1.narrow_book_active());
        assert!(v1.consider_move_count_active());

        let v2 = BookConfig {
            book_options_v2: true,
            ..v1
        };
        assert!(
            !v2.narrow_book_active(),
            "NarrowBook is always off under V2"
        );
        assert!(
            !v2.consider_move_count_active(),
            "ConsiderBookMoveCount is always off under V2"
        );
    }

    #[test]
    fn v2_eval_diff_uses_the_black_option_at_a_black_root() {
        let book = three_move_book();
        let mut c = cfg_v2();
        // The Black gap is tight, the White one wide: only the top value (100)
        // may survive at this Black root.
        c.eval_black_diff = 0;
        c.eval_white_diff = 99999;
        let mut prng = Prng::new(1);
        let r = probe_book(one(&book), false, &pos(STARTPOS_B), &c, &mut prng);
        let hit = r.hit.expect("hit");
        assert_eq!(usi(hit.best), "7g7f");
        assert!(
            r.diagnostics
                .iter()
                .any(|d| d.starts_with("BookEvalBlackDiff = 0 , BookEvalBlackLimit = ")),
            "expected a BookEvalBlackDiff notice, got {:?}",
            r.diagnostics
        );

        // Swapped: the wide Black gap governs, so every move survives.
        c.eval_black_diff = 99999;
        c.eval_white_diff = 0;
        let allowed = ["7g7f", "2g2f", "6i7h"];
        for seed in [1u64, 7, 99, 4321] {
            let mut prng = Prng::new(seed);
            let hit = probe_book(one(&book), false, &pos(STARTPOS_B), &c, &mut prng)
                .hit
                .expect("hit");
            assert!(allowed.contains(&usi(hit.best).as_str()), "seed {seed}");
        }
    }

    #[test]
    fn v2_eval_diff_uses_the_white_option_at_a_white_root() {
        let book = three_move_book_white();
        let mut c = cfg_v2();
        c.eval_black_diff = 99999;
        c.eval_white_diff = 0;
        let mut prng = Prng::new(1);
        let r = probe_book(one(&book), false, &pos(STARTPOS_W), &c, &mut prng);
        let hit = r.hit.expect("hit");
        assert_eq!(usi(hit.best), "3c3d");
        assert!(
            r.diagnostics
                .iter()
                .any(|d| d.starts_with("BookEvalWhiteDiff = 0 , BookEvalWhiteLimit = ")),
            "expected a BookEvalWhiteDiff notice, got {:?}",
            r.diagnostics
        );

        // The Black gap is inert at a White root.
        c.eval_black_diff = 0;
        c.eval_white_diff = 99999;
        let allowed = ["3c3d", "8c8d", "4a3b"];
        for seed in [1u64, 7, 99, 4321] {
            let mut prng = Prng::new(seed);
            let hit = probe_book(one(&book), false, &pos(STARTPOS_W), &c, &mut prng)
                .hit
                .expect("hit");
            assert!(allowed.contains(&usi(hit.best).as_str()), "seed {seed}");
        }
    }

    #[test]
    fn v2_depth_limit_uses_the_side_to_move_option() {
        // Best-move depth is 20 in both books.
        let black_book = three_move_book();
        let white_book = three_move_book_white();

        // Black root: only BookDepthBlackLimit bites.
        let mut c = cfg_v2();
        c.depth_black_limit = 25;
        c.depth_white_limit = 0;
        let mut prng = Prng::new(1);
        let r = probe_book(one(&black_book), false, &pos(STARTPOS_B), &c, &mut prng);
        assert!(r.hit.is_none(), "Black depth floor above 20 → miss");
        assert!(
            r.diagnostics
                .iter()
                .any(|d| d == "BookDepthBlackLimit is lower than the depth of this node."),
            "expected a BookDepthBlackLimit notice, got {:?}",
            r.diagnostics
        );

        c.depth_black_limit = 0;
        c.depth_white_limit = 25;
        let mut prng = Prng::new(1);
        assert!(
            probe_book(one(&black_book), false, &pos(STARTPOS_B), &c, &mut prng)
                .hit
                .is_some(),
            "the White depth floor is inert at a Black root"
        );

        // White root: the mirror image.
        let mut prng = Prng::new(1);
        let r = probe_book(one(&white_book), false, &pos(STARTPOS_W), &c, &mut prng);
        assert!(r.hit.is_none(), "White depth floor above 20 → miss");
        assert!(
            r.diagnostics
                .iter()
                .any(|d| d == "BookDepthWhiteLimit is lower than the depth of this node."),
            "expected a BookDepthWhiteLimit notice, got {:?}",
            r.diagnostics
        );

        c.depth_black_limit = 25;
        c.depth_white_limit = 0;
        let mut prng = Prng::new(1);
        assert!(
            probe_book(one(&white_book), false, &pos(STARTPOS_W), &c, &mut prng)
                .hit
                .is_some(),
            "the Black depth floor is inert at a White root"
        );
    }

    #[test]
    fn v2_ignores_the_v1_only_filter_values() {
        // A V1-only eval gap / depth floor that would filter everything must have
        // no effect once the V2 profile is active.
        let book = three_move_book();
        let mut c = cfg_v2();
        c.eval_diff = 0;
        c.depth_limit = 25;
        let allowed = ["7g7f", "2g2f", "6i7h"];
        for seed in [1u64, 2, 3, 4] {
            let mut prng = Prng::new(seed);
            let hit = probe_book(one(&book), false, &pos(STARTPOS_B), &c, &mut prng)
                .hit
                .expect("V1 fields are inert under V2");
            assert!(allowed.contains(&usi(hit.best).as_str()), "seed {seed}");
        }
    }

    // --- Multiple Book: first-hit probe over the priority list. ---

    /// Priority book 0: only the Black startpos, answering `7g7f`.
    fn priority_book_0() -> Book {
        let recs = vec![(
            sfen_pack(&pos(STARTPOS_B)),
            1u16,
            vec![(m16(STARTPOS_B, "7g7f"), 100, 20)],
        )];
        Book::from_memory(build_ybb(&recs)).expect("valid ybb")
    }

    /// Priority book 1: the Black startpos with a DIFFERENT move (`2g2f`), plus a
    /// White-startpos entry that book 0 does not carry.
    fn priority_book_1() -> Book {
        let recs = vec![
            (
                sfen_pack(&pos(STARTPOS_B)),
                1u16,
                vec![(m16(STARTPOS_B, "2g2f"), 100, 20)],
            ),
            (
                sfen_pack(&pos(STARTPOS_W)),
                1u16,
                vec![(m16(STARTPOS_W, "3c3d"), 100, 20)],
            ),
        ];
        Book::from_memory(build_ybb(&recs)).expect("valid ybb")
    }

    #[test]
    fn first_hit_wins_and_never_merges_across_books() {
        let books = vec![priority_book_0(), priority_book_1()];
        let c = cfg();

        // Present in book 0 AND book 1 with different moves → book 0 answers,
        // for every seed (no fall-through, no merged candidate set).
        for seed in [1u64, 2, 99, 4321] {
            let mut prng = Prng::new(seed);
            let hit = probe_book(&books, false, &pos(STARTPOS_B), &c, &mut prng)
                .hit
                .expect("hit");
            assert_eq!(usi(hit.best), "7g7f", "seed {seed}");
            assert_eq!(
                hit.info_lines.len(),
                1,
                "book 1's move must not be merged in: {:?}",
                hit.info_lines
                    .iter()
                    .map(|l| usi(l.pv[0]))
                    .collect::<Vec<_>>()
            );
        }

        // Reversing the priority order reverses the answer — the list order is
        // what decides, not the file contents.
        let reversed = vec![priority_book_1(), priority_book_0()];
        let mut prng = Prng::new(1);
        let hit = probe_book(&reversed, false, &pos(STARTPOS_B), &c, &mut prng)
            .hit
            .expect("hit");
        assert_eq!(usi(hit.best), "2g2f");
    }

    #[test]
    fn a_miss_in_book_0_falls_through_to_book_1() {
        let books = vec![priority_book_0(), priority_book_1()];
        let c = cfg(); // flipped_book off, so the White root cannot hit book 0.
        let mut prng = Prng::new(1);
        let hit = probe_book(&books, false, &pos(STARTPOS_W), &c, &mut prng)
            .hit
            .expect("book 1 answers what book 0 lacks");
        assert_eq!(usi(hit.best), "3c3d");
    }

    #[test]
    fn absent_everywhere_is_a_miss() {
        let books = vec![priority_book_0(), priority_book_1()];
        // The post-7g7f position is in neither book.
        let after = {
            let mut p = pos(STARTPOS_B);
            p.do_move(parse_usi_move("7g7f", &p).unwrap());
            p
        };
        let mut prng = Prng::new(1);
        assert!(
            probe_book(&books, false, &after, &cfg(), &mut prng)
                .hit
                .is_none()
        );

        // An empty priority list is a miss too (the bookless engine).
        let mut prng = Prng::new(1);
        assert!(
            probe_book(&[], false, &pos(STARTPOS_B), &cfg(), &mut prng)
                .hit
                .is_none()
        );
    }

    #[test]
    fn pv_and_ponder_follow_the_same_first_hit_path() {
        // Book 0 has the root only; book 1 has the root's child. The PV walk and
        // the ponder lookup both continue into book 1 once book 0 misses.
        let after = {
            let mut p = pos(STARTPOS_B);
            p.do_move(parse_usi_move("7g7f", &p).unwrap());
            p
        };
        let child = vec![(
            sfen_pack(&after),
            after.ply(),
            vec![(parse_usi_move("3c3d", &after).unwrap().move16(), 80, 18)],
        )];
        let books = vec![
            priority_book_0(),
            Book::from_memory(build_ybb(&child)).expect("valid ybb"),
        ];
        let mut prng = Prng::new(1);
        let hit = probe_book(&books, false, &pos(STARTPOS_B), &cfg(), &mut prng)
            .hit
            .expect("hit");
        assert_eq!(usi(hit.best), "7g7f");
        assert_eq!(hit.ponder.map(usi), Some("3c3d".to_string()));
        assert_eq!(
            hit.info_lines[0]
                .pv
                .iter()
                .copied()
                .map(usi)
                .collect::<Vec<_>>(),
            vec!["7g7f".to_string(), "3c3d".to_string()],
            "the book PV crosses into the lower-priority book"
        );
    }

    #[test]
    fn illegal_entries_are_rejected_with_a_diagnostic() {
        // 7g5g is a syntactically valid but illegal startpos move (pawn jump).
        // Give it the highest value so it would be chosen if not rejected.
        let key = sfen_pack(&pos(STARTPOS_B));
        let recs = vec![(
            key,
            1u16,
            vec![
                (m16(STARTPOS_B, "7g5g"), 999, 30),
                (m16(STARTPOS_B, "7g7f"), 100, 20),
            ],
        )];
        let book = Book::from_memory(build_ybb(&recs)).expect("valid ybb");
        let mut c = cfg();
        c.eval_diff = 0; // only the best legal value survives.
        let mut prng = Prng::new(1);
        let r = probe_book(one(&book), false, &pos(STARTPOS_B), &c, &mut prng);
        let hit = r.hit.expect("legal move still selectable");
        assert_eq!(usi(hit.best), "7g7f");
        assert!(
            r.diagnostics.iter().any(|d| d.contains("Illegal Move")),
            "expected an illegal-move diagnostic, got {:?}",
            r.diagnostics
        );
    }

    #[test]
    fn consider_move_count_all_zero_counts_stays_in_set() {
        // `.ybb` has no counts → the all-zero-counts rule weights every move 1.
        let book = three_move_book();
        let mut c = cfg();
        c.consider_move_count = true;
        let allowed = ["7g7f", "2g2f", "6i7h"];
        for seed in [1u64, 2, 3, 4, 5] {
            let mut prng = Prng::new(seed);
            let hit = probe_book(one(&book), false, &pos(STARTPOS_B), &c, &mut prng)
                .hit
                .expect("hit");
            assert!(allowed.contains(&usi(hit.best).as_str()), "seed {seed}");
        }
    }

    #[test]
    fn game_ply_past_book_moves_misses() {
        let book = three_move_book();
        let mut c = cfg();
        c.book_moves = 0; // ply 1 > 0 → miss.
        let mut prng = Prng::new(1);
        assert!(
            probe_book(one(&book), false, &pos(STARTPOS_B), &c, &mut prng)
                .hit
                .is_none()
        );
    }

    #[test]
    fn ignore_rate_100_always_skips() {
        let book = three_move_book();
        let mut c = cfg();
        c.ignore_rate = 100;
        for seed in [1u64, 2, 3] {
            let mut prng = Prng::new(seed);
            assert!(
                probe_book(one(&book), false, &pos(STARTPOS_B), &c, &mut prng)
                    .hit
                    .is_none()
            );
        }
    }

    #[test]
    fn wrong_ply_misses_unless_ignore_book_ply() {
        // Book keyed at ply 1; probe the same board at ply 2.
        let book = three_move_book();
        let at_ply_2 = pos("lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 2");
        let c = cfg();
        let mut prng = Prng::new(1);
        assert!(
            probe_book(one(&book), false, &at_ply_2, &c, &mut prng)
                .hit
                .is_none(),
            "ply mismatch → miss when enforced"
        );
        let mut prng = Prng::new(1);
        assert!(
            probe_book(one(&book), true, &at_ply_2, &c, &mut prng)
                .hit
                .is_some(),
            "ply ignored → hit"
        );
    }

    #[test]
    fn flipped_book_hits_the_rotated_position() {
        // Startpos is symmetric under 180°-rotation + color swap, so the flip of
        // the White-to-move startpos packs to the Black-to-move key. Book the
        // Black key; probe White.
        let book = three_move_book();
        let mut c = cfg();
        c.eval_diff = 0; // isolate the flipped best move (7g7f → 3c3d).
        c.flipped_book = true;
        let mut prng = Prng::new(1);
        let hit = probe_book(one(&book), false, &pos(STARTPOS_W), &c, &mut prng)
            .hit
            .expect("flipped hit");
        assert_eq!(usi(hit.best), "3c3d", "7g7f mirrored 180° is 3c3d");

        // With FlippedBook off, the rotated position misses.
        c.flipped_book = false;
        let mut prng = Prng::new(1);
        assert!(
            probe_book(one(&book), false, &pos(STARTPOS_W), &c, &mut prng)
                .hit
                .is_none()
        );
    }

    #[test]
    fn ponder_fallback_reads_the_child_positions_best() {
        // Two chained entries: startpos → 7g7f, and the post-7g7f position →
        // 3c3d. The ponder for 7g7f is the child's first sorted move (3c3d).
        let after = {
            let mut p = pos(STARTPOS_B);
            p.do_move(parse_usi_move("7g7f", &p).unwrap());
            p
        };
        let after_sfen = attic_state::format_sfen(&after);
        let recs = vec![
            (
                sfen_pack(&pos(STARTPOS_B)),
                1u16,
                vec![(m16(STARTPOS_B, "7g7f"), 100, 20)],
            ),
            (
                sfen_pack(&after),
                after.ply(),
                vec![(parse_usi_move("3c3d", &after).unwrap().move16(), 80, 18)],
            ),
        ];
        let book = Book::from_memory(build_ybb(&recs)).expect("valid ybb");
        let mut prng = Prng::new(1);
        let hit = probe_book(one(&book), false, &pos(STARTPOS_B), &cfg(), &mut prng)
            .hit
            .expect("hit");
        assert_eq!(usi(hit.best), "7g7f");
        assert_eq!(
            hit.ponder.map(usi),
            Some("3c3d".to_string()),
            "ponder from child of {after_sfen}"
        );

        // A leaf (no child entry) yields no ponder.
        let solo = vec![(
            sfen_pack(&pos(STARTPOS_B)),
            1u16,
            vec![(m16(STARTPOS_B, "7g7f"), 100, 20)],
        )];
        let leaf_book = Book::from_memory(build_ybb(&solo)).expect("valid ybb");
        let mut prng = Prng::new(1);
        let hit = probe_book(one(&leaf_book), false, &pos(STARTPOS_B), &cfg(), &mut prng)
            .hit
            .expect("hit");
        assert_eq!(hit.ponder, None, "no child entry → no ponder");
    }
}
