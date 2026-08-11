# search fixtures

Captured fixed-depth alpha-beta search ground truth for `search(sfen, depth)`.
Each file pins the reference engine's search result for a single starting `sfen`
at a fixed depth with deterministic parameters. The result serves as a parity
gate: future Rust search implementations must match this reference output.

## Schema

```json
{
  "sfen": "<position in SFEN>",
  "depth": 3,
  "bestmove": "<USI move>",
  "score": { "cp": 95 },
  "nodes": 1813,
  "pv": ["<m1>", "<m2>", "..."]
}
```

- `depth` — the fixed search depth requested via `go depth <D>`.
- `bestmove` — the move the engine selected, in USI notation.
- `score` — either `{ "cp": <N> }` (centipawn) or `{ "mate": <N> }` (mate in N
  plies). The value is from the **side-to-move perspective**: positive means the
  side to move is better.
- `nodes` — node count reported by the engine for the completed search.
- `pv` — principal variation as a JSON array of USI moves.

### Optional `moves` prefix

Fixtures that depend on position history accept an optional `moves` field. When
present, the listed USI moves are played after parsing the SFEN before the
search. This mirrors USI's `position sfen <SFEN> moves <m1> <m2> ...` shape.

```json
{
  "sfen": "9/4k4/9/9/9/9/9/4K4/9 b 9P9p 1",
  "moves": ["5h4h", "5b4b"],
  "depth": 3,
  "bestmove": "...",
  "score": { "cp": 42 },
  "nodes": 500,
  "pv": ["..."]
}
```

The field is **optional**. `cargo xtask capture-search` writes it iff `--moves`
is non-empty.

## Build target

Fixtures are captured using the **`tournament`** build of the reference binary
(the default target for `cargo xtask build-reference`). The `tournament` target
loads `nn.bin` on `isready`, which is required for the NNUE evaluation used
during search.

## Fixed search parameters

All fixtures in this directory are captured with the following deterministic
parameters, which must not change between capture runs for a given position:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `Threads` | 1 | Single thread eliminates scheduling non-determinism |
| `BookFile` | `no_book` | Disables opening book so alpha-beta always runs |
| `usinewgame` | sent before `position` | Clears TT for a fresh hash state |
| `USI_Hash` | 1024 (MB) | Engine default — `capture-search` does not set it; TT size affects node counts, so a Rust TT must default to the same size for the `nodes` gate to be meaningful |
| `go depth` | 3 (default) | Fixed depth; no time management involved |

The engine's `StandardInput::input()` (misc.cpp) converts stdin EOF
to a synthetic `quit`, which aborts the search if stdin is closed before
`bestmove` is output. `cargo xtask capture-search` keeps stdin open until
`bestmove` is read, then closes it — this is why the captured depth matches the
requested depth rather than terminating at depth 1.

## Determinism

Running `cargo xtask capture-search` twice against the same submodule pin and
the same `nn.bin` produces a byte-identical file. This was verified by running
three successive captures and confirming all three outputs are identical.

## Score perspective

Scores are reported from the **side-to-move perspective**: a positive `cp` means
the side to move has an advantage; a negative `cp` means the opponent has an
advantage. This matches the YaneuraOu USI `info score` output convention.

## Regenerating

```sh
# 1. Build the reference binary (tournament target, default).
cargo xtask build-reference

# 2. Place a YaneuraOu-compatible eval network at:
#      eval/nn.bin
#    The network is obtained out-of-band
#    and must NOT be committed.

# 3. Capture the startpos fixture (depth 3, single thread, no book).
cargo xtask capture-search

# 4. (Optional) Capture a fixture for a different position or depth.
cargo xtask capture-search \
  --sfen "9/4k4/9/9/9/9/9/4K4/9 b 9P9p 1" \
  --depth 5 \
  --fixture tests/fixtures/search/my-position.json
```
