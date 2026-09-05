# ybb opening-book fixtures

Ground truth for the `.ybb` opening-book reader parity test
(`crates/attic-search/tests/book_probe.rs`). The `.ybb` format is the
YaneuraOu "YANE-BINBOOK-V1" binary book, derived from the reference build in
`upstream YaneuraOu` (`book/book.cpp`).

## Files

- `book.db` — hand-authored text book (the reference's `.db` grammar). Positions
  are drawn from the existing parity-fixture SFENs (`tests/fixtures/perft/*.json`):
  startpos, check-evasion, drop-heavy, sennichite. It includes a position with
  several moves carrying distinct value/depth (startpos and check-evasion) and
  drop moves (`G*5f`, `P*5e`).
- `sample.ybb` — the binary book (256 bytes: 32-byte header + 4×44-byte index +
  8×6-byte move records; the depth flag is set, so records are 6 bytes).
- `expected.json` — the expected probe results, derived from `book.db`. Note the
  `.ybb` move record has no per-move count field, so `count` is always 0 after
  conversion, and no ponder move is stored.

## How `sample.ybb` was produced (and anchored to the reference)

There is **no `.db → .ybb` converter in the reference** (the only ybb
writers are the low-level `write_ybb_*` helpers in `makebook2025.cpp`, and
`makebook peta_shock` only does `.db → .db` or `.ybb → .ybb`). So the sample was
**hand-constructed** per the derived binary format and then **validated through
the reference engine**:

1. `cargo xtask capture-book` reads `book.db`, packs each position with the
   workspace PackedSfen encoder (`attic_state::sfen_pack`) and encodes each move
   as a YaneuraOu `Move16` (`attic_state::Move::move16`), sorts the index by the
   packed key, and writes `sample.ybb` + `expected.json`.
2. The PackedSfen encoder is independently pinned bit-for-bit against the
   reference's own cshogi-produced vectors
   (`source/position.cpp`, reproduced in
   `attic-state/src/packed_sfen.rs`).
3. The whole `sample.ybb` was then loaded into the reference `YaneuraOu-by-gcc`
   binary as a book and probed. With the sample staged as `user_book1.ybb` (the
   `.db → .ybb` sibling fallback resolves `user_book1.db → user_book1.ybb`) and
   the depth/eval filters relaxed:

   ```
   setoption name BookDir value <abs path to a staging dir>
   setoption name BookFile value user_book1.db
   setoption name BookDepthLimit value 0
   setoption name BookEvalBlackLimit value -99999
   isready
   position sfen <each book SFEN>
   go depth 1
   ```

   the reference reported `read book done. number of positions = 4` and returned
   the stored best book move for every position:

   | SFEN (fixture)      | reference `bestmove` |
   |---------------------|----------------------|
   | startpos            | `7g7f`               |
   | check-evasion       | `1g5c+`              |
   | drop-heavy          | `5d5c+`              |
   | sennichite          | `P*5e`               |

   The reference computes the query key with its **own** `sfen_pack`; that it
   binary-searches our index and hits every position confirms our PackedSfen (and
   Move16) encoding matches the reference to the bit — a single-bit divergence
   would miss.

## Regenerating

```sh
cargo run -p xtask -- capture-book
```

Regenerates `sample.ybb` and `expected.json` from `book.db` byte-identically
(the encoder is deterministic).
