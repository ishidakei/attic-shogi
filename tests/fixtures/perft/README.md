# perft fixtures

Captured node-count ground truth for `perft(sfen, depth)`. Each file pins the
reference engine's node count for a single starting `sfen` across one or more
depths; the Rust engine's parity tests assert byte-equality against this data.

## Schema

```json
{
  "sfen": "<position in SFEN>",
  "results": [
    { "depth": 1, "expected_nodes": 30 },
    { "depth": 2, "expected_nodes": 900 }
  ]
}
```

`results` is ordered by ascending depth. `expected_nodes` is the perft node
count returned by the reference engine for the given `sfen` at that depth.

### Optional `moves` prefix

Fixtures that depend on **position history** (sennichite / 連続王手の千日手)
accept an optional `moves` field. When present, the parity-test loader plays
the listed USI moves with `do_move` after parsing the SFEN, populating the
position history before invoking `perft`. This mirrors USI's
`position sfen <SFEN> moves <m1> <m2> ...` shape — the same shape the
reference engine receives during fixture capture.

```json
{
  "sfen": "9/4k4/9/9/9/9/9/4K4/9 b 9P9p 1",
  "moves": ["5h4h", "5b4b", "4h5h", "4b5b"],
  "results": [
    { "depth": 1, "expected_nodes": 78 }
  ]
}
```

The field is **optional**. Existing fixtures (startpos, drop-heavy,
mid-game-tactical, check-evasion, promotion-zone-edges) omit it; the loader
treats absence as "no prefix". `cargo xtask capture-perft` writes the field
iff `--moves` is non-empty, so existing fixtures regenerate byte-identically.

The schema is per-`sfen` rather than a single monolithic file so future fixture
classes (mid-game tactical, drops, promotions, nifu / uchifuzume / sennichite)
can land alongside without touching this file. One file per `(category, sfen)`.

## Regenerating

```sh
# 1. Build the reference binary (tournament target).
cargo xtask build-reference

# 2. Place a YaneuraOu-compatible eval network at:
#      eval/nn.bin
#    The network is obtained out-of-band
#    and must NOT be committed. The reference engine refuses to proceed
#    past `isready` without it; perft itself does not consume the network,
#    but the upstream code path forces the load.

# 3. Capture fixtures.
cargo xtask capture-perft

# 4. (Optional) Capture a history-prefixed fixture (e.g. sennichite).
cargo xtask capture-perft \
  --sfen "9/4k4/9/9/9/9/9/4K4/9 b 9P9p 1" \
  --moves "5h4h 5b4b 4h5h 4b5b 5h4h 5b4b 4h5h 4b5b 5h4h 5b4b 4h5h 4b5b" \
  --fixture tests/fixtures/perft/sennichite.json \
  --max-depth 3
```

`capture-perft` is deterministic: rerunning against the same submodule pin
produces a byte-identical file.
