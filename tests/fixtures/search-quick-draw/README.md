# search-quick-draw fixtures

Reference ground truth for the position where the two repetition configurations
disagree. Consumed by `crates/attic-search/tests/quick_draw_parity.rs`, which is
compiled **only** in the `quick-draw` configuration (the workspace default,
mirroring upstream's `ENABLE_QUICK_DRAW`).

## Why this directory exists separately

The fixtures under `tests/fixtures/search*` pin positions where the
QUICK_DRAW and non-QUICK_DRAW variants of `Position::is_repetition` happen to
produce the same tree. These two pin bench position 3, where they do not:

```text
6n1l/2+S1k4/2lp4p/1np1B2b1/3PP4/1N1S3rP/1P2+pPP+p1/1p1G5/3KG2r1 b GSN2L4Pgs2p 1
```

Along `S*5c K5bx5c G*5b K5cx5b`, ply 4 repeats the root board with Black's hand
strictly poorer. `ENABLE_QUICK_DRAW` (`source/position.cpp`) has no
`st->repetition < ply` root gate, so it adjudicates `REPETITION_INFERIOR`
immediately; the non-QUICK_DRAW gate evaluates `4 < 4`, declines, and searches
on. At depth 3 that is 3,913 nodes versus 3,924.

## Schema

The `tests/fixtures/search/README.md` schema, plus two fields this directory
always carries:

- `hash_mb` — the `USI_Hash` (MiB) the capture ran with. The transposition table
  size changes node counts, so an exact `nodes` comparison is meaningless without it.
  The test sizes its table to this value.
- `fv_scale` — the `FV_SCALE` the reference actually searched with, taken from
  the engine's own `info string engine option override. name = FV_SCALE , value
  = <N>` line. The reference reads this from
  `eval/eval_options.txt`, an out-of-band file that is
  **not** implied by the reference build, so the value has to travel with the
  fixture. The test calls `attic_eval::set_fv_scale` with it.

## Fixed capture parameters

| Parameter | Value |
|-----------|-------|
| build | `tournament` (`cargo xtask build-reference`) — defines `FOR_TOURNAMENT` → `ENABLE_QUICK_DRAW` |
| `Threads` | 1 |
| `BookFile` | `no_book` |
| `usinewgame` | sent before `position` |
| `USI_Hash` | 256 (MiB), explicit |
| `go depth` | 3 and 8 |

Depth 3 is the shallowest depth that reaches the diverging line; depth 8 pins
that the divergence stays closed once the transposition table and the history
tables are warm.

## Regenerating

```sh
# 1. Build the reference binary (tournament target, default).
cargo xtask build-reference

# 2. Place a YaneuraOu-compatible eval network at
#    eval/nn.bin (obtained out-of-band, never
#    committed).

# 3. Recapture both depths.
SFEN="6n1l/2+S1k4/2lp4p/1np1B2b1/3PP4/1N1S3rP/1P2+pPP+p1/1p1G5/3KG2r1 b GSN2L4Pgs2p 1"
cargo xtask capture-search --sfen "$SFEN" --depth 3 --hash 256 \
  --fixture tests/fixtures/search-quick-draw/bench-pos3-depth3.json
cargo xtask capture-search --sfen "$SFEN" --depth 8 --hash 256 \
  --fixture tests/fixtures/search-quick-draw/bench-pos3-depth8.json
```
