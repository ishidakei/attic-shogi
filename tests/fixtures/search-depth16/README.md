# search-depth16 fixtures

Captured fixed-**depth-16** alpha-beta search ground truth for
`search(sfen, depth)`. One position only — `startpos`, the same SFEN as
`tests/fixtures/search/` (depth 3) through `tests/fixtures/search-depth8/`
(depth 8) — re-captured at `go depth 16`.

`crates/attic-search/tests/depth16_parity.rs` asserts **bestmove, score, and
nodes** exactly, as an inseparable triple.

## Why this tier exists

Depth 16 is the threshold for Step 9's **null-move verification search**
(`yaneuraou-search.cpp` at the pin). Its guard is
`nmpMinPly == 0 && depth >= 16`, so below depth 16 a null-move fail high returns
`nullValue` outright and the entire verification block is unreachable — the
depth-1/2/3/5/8 tiers cannot exercise it at all. From depth 16 up, a fail high
instead re-searches the **same node** (same `ss`, no `do_move`) at `depth - R`
with null-move pruning disabled until `ss->ply` climbs past
`nmpMinPly = ss->ply + 3 * (depth - R) / 4`, and returns `nullValue` only if that
verification also fails high; otherwise the node falls through to its ordinary
moves loop.

Because the verification search re-enters on the node's own stack cell, it also
rewrites `ss->staticEval` and can flip `ss->ttPv`. Every reference read of those
two after Step 9 is a live `ss->` read, so a port that caches them across Step 9
passes every shallower tier and diverges only here.

One position keeps the tier affordable: `startpos` at depth 16 is ~230k
cumulative nodes (~20x the depth-8 fixture, ~20 s in a debug test build). The
six-position sweep stays at depth 8.

## Schema

```json
{
  "sfen": "<position in SFEN>",
  "depth": 16,
  "bestmove": "<USI move>",
  "score": { "cp": 157 },
  "nodes": 229872,
  "pv": ["<m1>", "<m2>", "..."]
}
```

- `depth` — the fixed search depth requested via `go depth <D>` (always 16 here).
- `bestmove` — the move the engine selected, in USI notation.
- `score` — either `{ "cp": <N> }` (centipawn) or `{ "mate": <N> }` (mate in N
  plies), from the **side-to-move perspective**: positive means the side to move
  is better.
- `nodes` — node count reported by the engine, **cumulative over the whole `go`**
  (iterations 1..16 and every aspiration re-search).
- `pv` — principal variation as a JSON array of USI moves. Not gated.

### Optional `moves` prefix

Supported exactly as in the other search tiers (`cargo xtask capture-search`
writes the field iff `--moves` is non-empty); the single fixture here does not
use it.

## Build target and fixed parameters

Identical to `tests/fixtures/search/README.md`: captured with the **`tournament`**
build (loads `nn.bin` on `isready`), `Threads=1`, `BookFile=no_book`,
`usinewgame` before `position`, and the engine-default `USI_Hash=1024` MiB
(`capture-search` does not set it; TT size affects node counts, so the Rust TT
is resized to 1024 MiB and cleared before the fixture). FV_SCALE is the engine
default 16.

## Determinism

Running `cargo xtask capture-search` twice against the same submodule pin and the
same `nn.bin` produces a byte-identical file — re-capturing on the reference
build leaves `git diff` empty.

## Regenerating

```sh
# 1. Build the reference binary (tournament target, default).
cargo xtask build-reference

# 2. Place a YaneuraOu-compatible eval network at
#      eval/nn.bin  (obtained out-of-band, not committed).

# 3. Re-capture startpos at depth 16:
cargo xtask capture-search \
  --sfen "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1" \
  --depth 16 \
  --fixture tests/fixtures/search-depth16/startpos.json
```
