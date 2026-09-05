# eval fixtures

Captured static NNUE evaluation ground truth for `eval(sfen)`. Each file pins
the reference engine's evaluation for a single starting `sfen`; the value is
the raw integer returned by the reference engine's static evaluation function
(`Eval::evaluate`) before any search.

## Schema

```json
{
  "sfen": "<position in SFEN>",
  "eval": -103
}
```

`eval` is the static NNUE evaluation as printed by the reference engine's `e`
command (`eval = <integer>`). The value is from the **side-to-move perspective**
— positive means the side to move is better, negative means worse. The raw
value is recorded verbatim without rescaling or sign flip.

### Optional `moves` prefix

Fixtures that depend on position history accept an optional `moves` field.
When present, the listed USI moves are played after parsing the SFEN before
the static evaluation is captured. This mirrors USI's
`position sfen <SFEN> moves <m1> <m2> ...` shape.

```json
{
  "sfen": "9/4k4/9/9/9/9/9/4K4/9 b 9P9p 1",
  "moves": ["5h4h", "5b4b"],
  "eval": 42
}
```

The field is **optional**. `cargo xtask capture-eval` writes the field iff
`--moves` is non-empty.

## Build target

Fixtures are captured using the **`tournament`** build of the reference binary
(the default target for `cargo xtask build-reference`). The `tournament` target
loads `nn.bin` on `isready` and exposes the `e` command (YaneuraOu-specific
non-Stockfish extension in `usi.cpp`) which calls `Eval::evaluate(pos)` and
prints `eval = <integer>` to stdout.

The `e` command is distinct from the `eval` command. `eval` calls `trace_eval()`
which is a TODO stub in `YaneuraOuEngine` and produces no output.

## Regenerating

```sh
# 1. Build the reference binary (tournament target, default).
cargo xtask build-reference

# 2. Place a YaneuraOu-compatible eval network at:
#      eval/nn.bin
#    The network is obtained out-of-band
#    and must NOT be committed.

# 3. Capture the startpos fixture.
cargo xtask capture-eval

# 4. (Optional) Capture a fixture for a different position or with moves.
cargo xtask capture-eval \
  --sfen "9/4k4/9/9/9/9/9/4K4/9 b 9P9p 1" \
  --moves "5h4h 5b4b" \
  --fixture tests/fixtures/eval/my-position.json
```

`capture-eval` is deterministic: rerunning against the same reference build and
the same `nn.bin` produces a byte-identical file.
