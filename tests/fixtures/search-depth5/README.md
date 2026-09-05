# search-depth5 fixtures

Captured fixed-**depth-5** alpha-beta search ground truth for
`search(sfen, depth)`. These are the same six positions as
`tests/fixtures/search/` (depth 3), re-captured at `go depth 5` so the parity
test exercises the deeper regimes that only activate at depth ≥ 5: the ProbCut
reduced search (`prob_cut_depth == 1`), the deeper null-move / futility / LMR
thresholds, and — via the cumulative node count — every depth-1..4 iteration
transitively.

Each file pins the reference engine's search result for a single starting `sfen`
at depth 5 with deterministic parameters. `tests/depth5_parity.rs` asserts
**bestmove, score, and nodes** exactly for all six as one inseparable set.

## Schema

```json
{
  "sfen": "<position in SFEN>",
  "depth": 5,
  "bestmove": "<USI move>",
  "score": { "cp": 121 },
  "nodes": 2569,
  "pv": ["<m1>", "<m2>", "..."]
}
```

- `depth` — the fixed search depth requested via `go depth <D>` (always 5 here).
- `bestmove` — the move the engine selected, in USI notation.
- `score` — either `{ "cp": <N> }` (centipawn) or `{ "mate": <N> }` (mate in N
  plies), from the **side-to-move perspective**: positive means the side to move
  is better.
- `nodes` — node count reported by the engine, **cumulative over the whole `go`**
  (iterations 1..5 and every aspiration re-search).
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

# 3. Re-capture, e.g. startpos at depth 5:
cargo xtask capture-search \
  --sfen "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1" \
  --depth 5 \
  --fixture tests/fixtures/search-depth5/startpos.json

# The sennichite fixture additionally passes:
#   --moves "5h4h 5b4b 4h5h 4b5b 5h4h 5b4b 4h5h 4b5b 5h4h 5b4b 4h5h 4b5b"
```
