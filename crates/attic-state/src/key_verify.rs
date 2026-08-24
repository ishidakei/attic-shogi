//! Formal, independent verification that the port's Zobrist tables and composed
//! position keys are bit-identical to the pinned reference
//! (`upstream YaneuraOu` @ `76d58ef`, `source/position.cpp`).
//!
//! This module is **test-only** and deliberately re-derives the reference's
//! Zobrist generation *from scratch* — it does **not** call, read, or copy any
//! of [`crate::key`]'s generation internals (`rand64`, `set_rand`, `build`,
//! `ref_code_to_slot`). The only things it touches from the port are the tables
//! *under test*, read through their public-in-crate accessors
//! ([`crate::key::psq`], [`crate::key::hand_step`], [`crate::key::side`],
//! [`crate::key::NO_PAWNS_SEED`]). The whole point is that a future transcription
//! regression in `key.rs` — a shifted draw, a mis-mapped piece code, a dropped
//! `noPawns` draw — is caught here by an *independent* second opinion rather than
//! passing silently just because both sides share a bug.
//!
//! The reference algorithm (`Position::init`, verified against the pin's actual
//! loop structure and draw counts, `position.cpp`, `misc.h`,
//! `extra/key128.h`, `types.h`):
//!
//! - `PRNG` is `xorshift64*` seeded `20151225`. One draw advances the state with
//!   `x ^= x>>12; x ^= x<<25; x ^= x>>27` and returns `x * 2685821657736338717`.
//! - `set_rand` draws **four** words per entry; the 64-bit `SET_HASH` config
//!   keeps the **first** and discards the other three (they still advance the
//!   stream).
//! - Fill order: `Zobrist::zero` (no draw) → `side` → `noPawns`
//!   (`USE_PARTIAL_KEY` is defined for this config) → `psq[pc][sq]` over
//!   reference `Piece()` `pc = 1..=31` (`if (pc)` skips `0`) × `SQ` `sq = 0..=80`
//!   → `hand[c][pr]` over `COLOR` × `pr = 1..=7` (`if (pr)` skips
//!   `NO_PIECE_TYPE`, loop stops before `PIECE_HAND_NB == KING == 8`).

#![cfg(test)]

use crate::color::Color;
use crate::piece::{Piece, PieceKind};
use crate::position::Position;
use crate::square::Square;

/// The reference PRNG seed (`position.cpp`): 電王トーナメント 2015 の開発開始日.
const SEED: u64 = 20151225;
/// The `xorshift64*` output multiplier (`misc.h`).
const MULT: u64 = 2685821657736338717;

/// Reference `PieceType` order (`types.h`): `NO_PIECE_TYPE, PAWN, LANCE,
/// KNIGHT, SILVER, BISHOP, ROOK, GOLD, KING`. This is the reference's numbering,
/// **not** the port's [`PieceKind`] numbering (which puts `Gold` before
/// `Bishop`) — kept separate on purpose so the mapping is exercised.
const ALL_KINDS: [PieceKind; PieceKind::COUNT] = [
    PieceKind::Pawn,
    PieceKind::Lance,
    PieceKind::Knight,
    PieceKind::Silver,
    PieceKind::Gold,
    PieceKind::Bishop,
    PieceKind::Rook,
    PieceKind::King,
];

/// The reference `PRNG` — a standalone stepping state, independent of
/// [`crate::key`]'s `const fn rand64`.
struct Prng {
    state: u64,
}

impl Prng {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// One `PRNG::rand64()` step (`misc.h`): mutate the state, return
    /// `state * MULT`.
    fn rand64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(MULT)
    }

    /// The reference `set_rand` (`position.cpp`, `SET_HASH` at
    /// `extra/key128.h`): draw four words, keep the first.
    fn set_rand(&mut self) -> u64 {
        let r1 = self.rand64();
        let _r2 = self.rand64();
        let _r3 = self.rand64();
        let _r4 = self.rand64();
        r1
    }
}

/// Independently re-derived reference tables, indexed by the *reference's* own
/// conventions (reference `Piece` code, reference `PieceType`, reference `SQ`).
struct RefTables {
    side: u64,
    no_pawns: u64,
    /// `psq[pc][sq]` — `pc` is the reference `Piece` code `0..=31` (`0` unused),
    /// `sq` the reference `SQ` index `0..=80`.
    psq: [[u64; Square::COUNT]; 32],
    /// `hand[c][pr]` — `pr` is the reference `PieceType` `0..=7` (`0` unused;
    /// `1..=7` = `PAWN..GOLD`).
    hand: [[u64; 8]; Color::COUNT],
}

/// Decode a reference `Piece` code (`types.h`) into `(promoted, color,
/// kind)`, or `None` for a code that never lands on a board — `NO_PIECE` (`0`),
/// the unnamed gap at `16`, and the `B_GOLDS` / `W_GOLDS` gold-equivalent meta
/// pieces (`15` / `31`). `None` codes are still *drawn* (to keep the PRNG stream
/// aligned) but carry no port entry to compare against.
///
/// Re-derived here from the reference enum layout — deliberately *not* sharing
/// code with [`crate::key`]'s `ref_code_to_slot`.
fn decode_ref_piece(pc: usize) -> Option<(bool, Color, PieceKind)> {
    if pc == 0 || pc == 16 {
        return None; // NO_PIECE and the 16 gap between B_GOLDS and W_PAWN.
    }
    let (color, local) = if pc < 16 {
        (Color::Black, pc)
    } else {
        (Color::White, pc - 16)
    };
    // `local` is now 1..=15 within one colour's block.
    if local == 15 {
        return None; // B_GOLDS / W_GOLDS meta piece.
    }
    // 1..=8 are the unpromoted PieceTypes (PAWN..KING); 9..=14 are the promoted
    // forms PRO_PAWN..DRAGON, i.e. PieceType (local - 8) = PAWN..ROOK promoted.
    let (promoted, ref_pt) = if local <= 8 {
        (false, local)
    } else {
        (true, local - 8)
    };
    let kind = ref_piece_type_to_kind(ref_pt)?;
    Some((promoted, color, kind))
}

/// Map a reference `PieceType` (`1..=8`) to this port's [`PieceKind`]. The
/// reference order is `PAWN, LANCE, KNIGHT, SILVER, BISHOP, ROOK, GOLD, KING`
/// (`types.h`); the port puts `Gold` at index 4 and `Bishop`/`Rook` after
/// it, so the remap is not the identity.
fn ref_piece_type_to_kind(pt: usize) -> Option<PieceKind> {
    Some(match pt {
        1 => PieceKind::Pawn,
        2 => PieceKind::Lance,
        3 => PieceKind::Knight,
        4 => PieceKind::Silver,
        5 => PieceKind::Bishop,
        6 => PieceKind::Rook,
        7 => PieceKind::Gold,
        8 => PieceKind::King,
        _ => return None,
    })
}

/// Re-derive every reference Zobrist table from the pinned algorithm, drawing in
/// the exact reference order so the stream stays aligned entry-for-entry.
fn derive_reference_tables() -> RefTables {
    let mut rng = Prng::new(SEED);

    // `Zobrist::zero` — literal zeros, no draw (`position.cpp`).
    // First live draw: `side`. Second: `noPawns` (USE_PARTIAL_KEY).
    let side = rng.set_rand();
    let no_pawns = rng.set_rand();

    // psq: `for (pc : Piece()) for (sq : SQ) if (pc) set_rand(...)`. `Piece()`
    // runs `0..PIECE_NB` (`PIECE_NB == 32`); `if (pc)` skips `0`, so every code
    // `1..=31` draws once per square — even the never-realised gap / meta codes,
    // which keeps the stream aligned.
    let mut psq = [[0u64; Square::COUNT]; 32];
    for row in psq.iter_mut().skip(1) {
        for sq in row.iter_mut() {
            *sq = rng.set_rand();
        }
    }

    // hand: `for (c : COLOR) for (pr = NO_PIECE_TYPE; pr < PIECE_HAND_NB; ++pr)
    // if (pr) set_rand(...)`. `PIECE_HAND_NB == KING == 8`, so `pr` draws for
    // `1..=7` (PAWN..GOLD); no king in hand.
    let mut hand = [[0u64; 8]; Color::COUNT];
    for hand_c in hand.iter_mut() {
        for pr in hand_c.iter_mut().take(8).skip(1) {
            *pr = rng.set_rand();
        }
    }

    RefTables {
        side,
        no_pawns,
        psq,
        hand,
    }
}

/// Read the port's `psq` value for a `(promoted, color, kind, sq)`, returning `0`
/// for the never-realised promoted Gold / King combinations the port leaves
/// unset (and never reads).
fn port_psq(promoted: bool, color: Color, kind: PieceKind, sq: Square) -> u64 {
    let piece = if promoted {
        Piece::promoted(kind, color)
    } else {
        Some(Piece::new(kind, color))
    };
    piece.map_or(0, |p| crate::key::psq(p, sq))
}

/// Every table the port exposes, compared entry-by-entry against the independent
/// re-derivation. Any mismatch fails loudly with the table name and index; the
/// constants are never adjusted to make this pass.
#[test]
fn tables_match_independent_reference_derivation() {
    let refs = derive_reference_tables();

    assert_eq!(
        crate::key::side(),
        refs.side,
        "side key mismatch: port {:#018x} vs reference {:#018x}",
        crate::key::side(),
        refs.side
    );
    assert_eq!(
        crate::key::NO_PAWNS_SEED,
        refs.no_pawns,
        "noPawns seed mismatch: port {:#018x} vs reference {:#018x}",
        crate::key::NO_PAWNS_SEED,
        refs.no_pawns
    );

    // psq: walk every reference code that maps to a realised port piece and
    // compare all 81 squares. Codes with no port piece (NO_PIECE, the 16 gap,
    // the GOLDS meta pieces) are drawn by the reference to keep the stream
    // aligned but carry nothing to compare — the port never stores them.
    for pc in 1..32usize {
        let Some((promoted, color, kind)) = decode_ref_piece(pc) else {
            continue;
        };
        for sq_idx in 0..Square::COUNT {
            let sq = Square::from_index(sq_idx as u8).unwrap();
            let port = port_psq(promoted, color, kind, sq);
            let reference = refs.psq[pc][sq_idx];
            assert_eq!(
                port, reference,
                "psq mismatch at reference code {pc} ({promoted:?} {color:?} {kind:?}), \
                 square {sq_idx}: port {port:#018x} vs reference {reference:#018x}"
            );
        }
    }

    // hand: reference PieceType 1..=7 (PAWN..GOLD) per colour.
    for color in [Color::Black, Color::White] {
        for pr in 1..=7usize {
            let kind = ref_piece_type_to_kind(pr).unwrap();
            let port = crate::key::hand_step(color, kind);
            let reference = refs.hand[color.index()][pr];
            assert_eq!(
                port, reference,
                "hand mismatch at {color:?} reference PieceType {pr} ({kind:?}): \
                 port {port:#018x} vs reference {reference:#018x}"
            );
        }
    }
}

/// Compose a full position key directly from the independently re-derived raw
/// tables, mirroring the reference's `key = board_key ^ hand_key` split
/// (`position.cpp` / `position.h`): `board_key` is the XOR of `psq[piece][sq]`
/// over occupied squares plus the `side` term while White is to move; `hand_key`
/// is the wrapping sum of `hand[color][kind]` once per held copy.
fn compose_key(pos: &Position, refs: &RefTables) -> u64 {
    let mut board_key = 0u64;
    for sq_idx in 0..Square::COUNT as u8 {
        let sq = Square::from_index(sq_idx).unwrap();
        if let Some(piece) = pos.board().get(sq) {
            // Re-encode the port piece into its reference code, then index the
            // independent table — no reliance on the port's own psq storage.
            let pc = ref_code_for_port_piece(piece);
            board_key ^= refs.psq[pc][sq_idx as usize];
        }
    }
    if pos.side_to_move() == Color::White {
        board_key ^= refs.side;
    }

    let mut hand_key = 0u64;
    for color in [Color::Black, Color::White] {
        for pr in 1..=7usize {
            let kind = ref_piece_type_to_kind(pr).unwrap();
            let n = pos.hand(color).count(kind) as u64;
            hand_key = hand_key.wrapping_add(refs.hand[color.index()][pr].wrapping_mul(n));
        }
    }

    board_key ^ hand_key
}

/// Encode a port [`Piece`] into its reference `Piece` code — the inverse of
/// [`decode_ref_piece`], derived independently from the reference enum layout
/// (`types.h`, `PIECE_WHITE == 16`).
fn ref_code_for_port_piece(piece: Piece) -> usize {
    let ref_pt = match piece.kind {
        PieceKind::Pawn => 1,
        PieceKind::Lance => 2,
        PieceKind::Knight => 3,
        PieceKind::Silver => 4,
        PieceKind::Bishop => 5,
        PieceKind::Rook => 6,
        PieceKind::Gold => 7,
        PieceKind::King => 8,
    };
    // Promoted forms are PRO_PAWN..DRAGON = PieceType + 8 (PAWN promoted -> 9).
    let local = if piece.promoted { ref_pt + 8 } else { ref_pt };
    let color_offset = if piece.color == Color::White { 16 } else { 0 };
    color_offset + local
}

/// Composed-key cross-check on real positions: for a spread of positions
/// (startpos, promotions on board, drops in hand, both sides to move), the key
/// recomposed from the raw tables must equal the incrementally / recompute
/// maintained `pos.key()`.
#[test]
fn composed_keys_match_maintained_keys() {
    let refs = derive_reference_tables();

    // SFEN positions covering promotions, drops, and both sides to move. Parsed
    // via `parse_sfen`, whose direct board/hand mutations reseed the key through
    // `refresh_keys` (the recompute path).
    let sfens = [
        // Startpos, Black to move, empty hands.
        crate::sfen::STARTPOS_SFEN,
        // Promoted pieces of both colours on the board, Black to move.
        "+P+L+N+S1+p+l+n+s/9/9/9/4k4/9/9/9/4K4 b - 1",
        // Hand pieces for both colours (drops available), White to move.
        "4k4/9/9/9/9/9/9/9/4K4 w RBGSNLP2r2b3p 1",
        // Horse and dragon (promoted bishop/rook) on the board, with hands,
        // White to move.
        "+r4+b2k/9/9/9/9/9/9/9/K3+R3+B b Gg5P 1",
    ];
    for sfen in sfens {
        let pos = crate::sfen::parse_sfen(sfen).unwrap();
        assert_eq!(
            compose_key(&pos, &refs),
            pos.key(),
            "composed key disagrees with maintained key for SFEN `{sfen}`"
        );
    }

    // Incremental path: from startpos, walk a real opening line that exercises a
    // capture-with-promotion (8h2b+), a recapture (3a2b) and a drop (B*5e), each
    // ply cross-checked against the independent composition.
    let mut pos = crate::sfen::parse_sfen(crate::sfen::STARTPOS_SFEN).unwrap();
    for usi in ["7g7f", "3c3d", "8h2b+", "3a2b", "B*5e"] {
        let mv = crate::move_::parse_usi_move(usi, &pos).unwrap();
        pos.do_move(mv);
        assert_eq!(
            compose_key(&pos, &refs),
            pos.key(),
            "composed key disagrees with maintained key after `{usi}`"
        );
    }
}

/// Minimal, self-contained SHA-256 (FIPS 180-4). Test-only; avoids pulling a
/// crypto dependency into the crate just to anchor a golden digest.
fn sha256(data: &[u8]) -> [u8; 32] {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];
    let mut h: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];

    let bit_len = (data.len() as u64).wrapping_mul(8);
    let mut msg = data.to_vec();
    msg.push(0x80);
    while msg.len() % 64 != 56 {
        msg.push(0);
    }
    msg.extend_from_slice(&bit_len.to_be_bytes());

    // Padding above makes the length an exact multiple of 64, so the remainder
    // half of `as_chunks` is always empty.
    for chunk in msg.as_chunks::<64>().0 {
        let mut w = [0u32; 64];
        for (i, word) in w.iter_mut().enumerate().take(16) {
            let b = i * 4;
            *word = u32::from_be_bytes([chunk[b], chunk[b + 1], chunk[b + 2], chunk[b + 3]]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }

        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh] = h;
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let t1 = hh
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let t2 = s0.wrapping_add(maj);
            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(t1);
            d = c;
            c = b;
            b = a;
            a = t1.wrapping_add(t2);
        }
        h[0] = h[0].wrapping_add(a);
        h[1] = h[1].wrapping_add(b);
        h[2] = h[2].wrapping_add(c);
        h[3] = h[3].wrapping_add(d);
        h[4] = h[4].wrapping_add(e);
        h[5] = h[5].wrapping_add(f);
        h[6] = h[6].wrapping_add(g);
        h[7] = h[7].wrapping_add(hh);
    }

    let mut out = [0u8; 32];
    for (i, word) in h.iter().enumerate() {
        out[i * 4..i * 4 + 4].copy_from_slice(&word.to_be_bytes());
    }
    out
}

fn to_hex(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

/// Canonical byte serialisation of every table the port exposes, in a fixed
/// order: `side`, `noPawns`, then `psq` in port piece-code order (each entry's
/// 81 squares), then `hand` in `(color, kind)` order. Every `u64` is emitted
/// little-endian. Reading through the port accessors means an accidental
/// regeneration of `key.rs` changes these bytes and thus the digest.
fn serialize_port_tables() -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&crate::key::side().to_le_bytes());
    bytes.extend_from_slice(&crate::key::NO_PAWNS_SEED.to_le_bytes());

    // The `(promoted, color, kind)` nesting matches the port's `piece_code`
    // index arithmetic (`(promoted*2 + color)*8 + kind`), so this walks the psq
    // table in ascending port code order.
    for promoted in [false, true] {
        for color in [Color::Black, Color::White] {
            for kind in ALL_KINDS {
                for sq_idx in 0..Square::COUNT {
                    let sq = Square::from_index(sq_idx as u8).unwrap();
                    bytes.extend_from_slice(&port_psq(promoted, color, kind, sq).to_le_bytes());
                }
            }
        }
    }

    for color in [Color::Black, Color::White] {
        for kind in ALL_KINDS {
            bytes.extend_from_slice(&crate::key::hand_step(color, kind).to_le_bytes());
        }
    }

    bytes
}

/// Golden checksum. A SHA-256 over the canonical serialisation of all port
/// tables, pinned to the reference `upstream YaneuraOu` @ `76d58ef`
/// (`source/position.cpp`). Any accidental regeneration of the Zobrist
/// tables changes this digest and fails loudly — do **not** edit the constant to
/// make the test pass; investigate the table change instead. The value is
/// anchored by [`tables_match_independent_reference_derivation`], which proves
/// the serialised tables equal the independent reference derivation.
const GOLDEN_TABLES_SHA256: &str =
    "5f5c5937aadc1b12c824824d2d595fa2fd893b956c4bfb54aa1bcd02a53502a0";

#[test]
fn golden_table_checksum_is_stable() {
    let digest = to_hex(&sha256(&serialize_port_tables()));
    assert_eq!(
        digest, GOLDEN_TABLES_SHA256,
        "Zobrist table golden checksum changed — the tables were regenerated. \
         Do NOT update the constant to silence this; verify the tables against \
         the reference (upstream YaneuraOu @ 76d58ef) first."
    );
}

#[cfg(test)]
mod sha256_self_test {
    use super::sha256;
    use super::to_hex;

    #[test]
    fn known_vectors() {
        // FIPS 180-4 / RFC 6234 reference vectors.
        assert_eq!(
            to_hex(&sha256(b"")),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            to_hex(&sha256(b"abc")),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        assert_eq!(
            to_hex(&sha256(
                b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"
            )),
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
        );
    }
}
