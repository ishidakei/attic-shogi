# search-depth2 fixtures

A single fixed-**depth-2** alpha-beta search fixture: `position startpos moves
7g7f`, `go depth 2`. It is the minimal reproduction of the pawn-history-aliasing
regression since fixed.

A fresh `go depth 2` on the `7g7f` child used to report **1752** nodes against
the reference's **1753** — a single node. The divergence was root-caused to the
engine's **Zobrist tables**: they were generated from a private seed rather than
the reference's, so the hash-indexed pawn history (`pawnHistory`, 8192 planes)
and the correction histories aliased differently, flipping a quiet move's
ordering on the first colliding pawn structure. `crates/attic-state/src/key.rs`
now reproduces the reference Zobrist bit-for-bit, so the aliasing — and the node
counts — match. `tests/depth2_parity.rs` gates **bestmove, score, and nodes**
exactly.

The fixture was captured with Threads=1, no book, `usinewgame`, `go depth 2`,
USI_Hash default 1024 MiB, FV_SCALE 16:

```console
cargo xtask capture-search --moves "7g7f" --depth 2 \
  --fixture tests/fixtures/search-depth2/startpos-7g7f.json
```

Re-capturing leaves the git diff empty (single-thread fixed-depth search is
byte-reproducible against the pinned submodule + `nn.bin`).

## Schema

```json
{
  "sfen": "<start SFEN>",
  "moves": ["<usi move>", ...],
  "depth": 2,
  "bestmove": "<usi move>",
  "score": { "cp": <int> },
  "nodes": <int>,
  "pv": ["<usi move>", ...]
}
```
