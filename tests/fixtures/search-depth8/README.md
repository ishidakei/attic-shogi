# search-depth8 fixtures

Captured fixed-**depth-8** alpha-beta search ground truth for
`search(sfen, depth)`. These are the same six positions as
`tests/fixtures/search/` (depth 3) and `tests/fixtures/search-depth5/` (depth 5),
re-captured at `go depth 8` so the parity test exercises the regimes that only
activate at depth ≥ 6 — chiefly the **singular-extension family** (its guard
`!rootNode && depth >= 6 + ss->ttPv` firing directly at interior nodes: the
singular search, multi-cut pruning, double/triple-margin extensions, and
negative extensions), internal iterative reduction (`depth >= 6`, including the
`priorReduction <= 3` term), and the `depth > 5` disjunct of the non-PV early TT
cutoff — and, via the cumulative node count, every depth-1..7 iteration
transitively.

Each file pins the reference engine's search result for a single starting `sfen`
at depth 8 with deterministic parameters. `tests/depth8_parity.rs` asserts
**bestmove, score, and nodes** exactly for each fixture (each triple inseparable).

> **All six fixtures now match the ported engine exactly.** `startpos` used to
> diverge due to **two** pre-existing bugs (present before the
> singular-extension port), both since fixed: (1) the engine's Zobrist tables
> aliased the hash-indexed
> pawn / correction histories differently from the reference — fixed by
> reproducing the reference Zobrist bit-for-bit
> (`crates/attic-state/src/key.rs`), reproducible from depth 2 and pinned by
> `tests/depth2_parity.rs`; and (2) the qsearch `do_move` left the continuation /
> continuation-correction planes stale (the reference sets them for every move),
> which shifted a corrected eval by ~1 from depth 6 up — fixed in `qsearch.rs`.
> See the module docs in `crates/attic-search/tests/depth8_parity.rs` for the
> full diagnosis.

## Schema

```json
{
  "sfen": "<position in SFEN>",
  "depth": 8,
  "bestmove": "<USI move>",
  "score": { "cp": 138 },
  "nodes": 12636,
  "pv": ["<m1>", "<m2>", "..."]
}
```

- `depth` — the fixed search depth requested via `go depth <D>` (always 8 here).
- `bestmove` — the move the engine selected, in USI notation.
- `score` — either `{ "cp": <N> }` (centipawn) or `{ "mate": <N> }` (mate in N
  plies), from the **side-to-move perspective**: positive means the side to move
  is better.
- `nodes` — node count reported by the engine, **cumulative over the whole `go`**
  (iterations 1..8 and every aspiration re-search).
- `pv` — principal variation as a JSON array of USI moves.

### Optional `moves` prefix

Fixtures that depend on position history (here, `sennichite.json`) carry an
optional `moves` field: the listed USI moves are played after parsing the SFEN,
before the search — USI's `position sfen <SFEN> moves <m1> <m2> ...` shape.
`cargo xtask capture-search` writes it iff `--moves` is non-empty.

## Build target and fixed parameters

Identical to `tests/fixtures/search/README.md`: captured with the **`tournament`**
build (loads `nn.bin` on `isready`), `Threads=1`, `BookFile=no_book`,
`usinewgame` before `position`, and the engine-default `USI_Hash=1024` MiB
(`capture-search` does not set it; TT size affects node counts, so the Rust TT
is resized to 1024 MiB and cleared per fixture). FV_SCALE is the engine default
16.

## Determinism

Running `cargo xtask capture-search` twice against the same reference build and the
same `nn.bin` produces a byte-identical file — re-capturing on the reference
build leaves `git diff` empty.

## Regenerating

```sh
# 1. Build the reference binary (tournament target, default).
cargo xtask build-reference

# 2. Place a YaneuraOu-compatible eval network at
#      eval/nn.bin  (obtained out-of-band, not committed).

# 3. Re-capture, e.g. startpos at depth 8:
cargo xtask capture-search \
  --sfen "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1" \
  --depth 8 \
  --fixture tests/fixtures/search-depth8/startpos.json

# The sennichite fixture additionally passes:
#   --moves "5h4h 5b4b 4h5h 4b5b 5h4h 5b4b 4h5h 4b5b 5h4h 5b4b 4h5h 4b5b"
```
