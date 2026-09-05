//! PackedSfen — the 256-bit (32-byte) Huffman position encoding, ported
//! bit-for-bit from `SfenPacker::pack` (`extra/sfen_packer.cpp`).
//!
//! The `.ybb` opening-book index keys positions by this exact encoding, so any
//! single-bit divergence from the reference writer makes the index binary
//! search miss.
//!
//! Bits are packed least-significant-first within each byte, bytes ascending:
//! stream bit 0 is bit 0 (LSB) of `data[0]` (`sfen_packer.cpp`). The
//! trailing "piece box" pass — every piece neither on the board nor in a hand —
//! pads the stream so that any position within the standard piece complement
//! lands on exactly 256 bits.

use crate::color::Color;
use crate::piece::{Piece, PieceKind};
use crate::position::Position;
use crate::square::Square;

/// Length in bytes of a [`PackedSfen`].
pub const PACKED_SFEN_LEN: usize = 32;

/// A shogi position packed into 32 bytes, identical to YaneuraOu's `PackedSfen`.
///
/// The encoding covers side-to-move, both king squares, the board, and both
/// hands — but **not** the game ply. Two positions that differ only in ply pack
/// to identical bytes (the `.ybb` index carries ply separately).
pub type PackedSfen = [u8; PACKED_SFEN_LEN];

/// Piece kinds in the reference's "Apery" enumeration order, used for both the
/// hand and the piece-box passes (`to_apery_pieces[]`, `sfen_packer.cpp`).
const HAND_ORDER: [PieceKind; 7] = [
    PieceKind::Pawn,
    PieceKind::Lance,
    PieceKind::Knight,
    PieceKind::Silver,
    PieceKind::Gold,
    PieceKind::Bishop,
    PieceKind::Rook,
];

/// Starting piece-box counts, indexed by `PieceKind::index()` for `Pawn..=Rook`
/// (`sfen_packer.cpp`). King is never boxed.
const PIECE_BOX_START: [i32; 7] = [18, 4, 4, 4, 4, 2, 2];

/// LSB-first bit writer over the fixed 32-byte buffer.
struct BitWriter {
    data: PackedSfen,
    cursor: usize,
}

impl BitWriter {
    fn new() -> Self {
        Self {
            data: [0; PACKED_SFEN_LEN],
            cursor: 0,
        }
    }

    #[inline]
    fn write_one_bit(&mut self, b: bool) {
        let byte = self.cursor / 8;
        // An over-populated (illegal) position runs past 256 bits; dropping the
        // overflow keeps the encoder total, and its output is only meaningful
        // for legal inputs anyway.
        if b && byte < PACKED_SFEN_LEN {
            self.data[byte] |= 1 << (self.cursor & 7);
        }
        self.cursor += 1;
    }

    #[inline]
    fn write_n_bit(&mut self, d: u32, n: u32) {
        for i in 0..n {
            self.write_one_bit((d & (1 << i)) != 0);
        }
    }
}

/// `(code, bits)` for the on-board / hand Huffman code of a raw piece kind
/// (`huffman_table[]`, `sfen_packer.cpp`).
fn huffman_board(kind: PieceKind) -> (u32, u32) {
    match kind {
        PieceKind::Pawn => (0x01, 2),
        PieceKind::Lance => (0x03, 4),
        PieceKind::Knight => (0x0b, 4),
        PieceKind::Silver => (0x07, 4),
        PieceKind::Gold => (0x0f, 5),
        PieceKind::Bishop => (0x1f, 6),
        PieceKind::Rook => (0x3f, 6),
        PieceKind::King => unreachable!("king is not Huffman-coded as a board/hand piece"),
    }
}

/// `(code, bits)` for the piece-box Huffman code of a raw piece kind
/// (`huffman_table_piecebox[]`, `sfen_packer.cpp`).
fn huffman_piece_box(kind: PieceKind) -> (u32, u32) {
    match kind {
        PieceKind::Pawn => (0x02, 2),
        PieceKind::Lance => (0x09, 4),
        PieceKind::Knight => (0x0d, 4),
        PieceKind::Silver => (0x0b, 4),
        PieceKind::Gold => (0x1b, 5),
        PieceKind::Bishop => (0x2f, 6),
        PieceKind::Rook => (0x3f, 6),
        PieceKind::King => unreachable!("king is never in the piece box"),
    }
}

/// A board piece, Huffman-coded (`write_board_piece_to_stream`,
/// `sfen_packer.cpp`): code, then a promote bit (except Gold), then a
/// color bit.
fn write_board_piece(w: &mut BitWriter, piece: Piece) {
    let (code, bits) = huffman_board(piece.kind);
    w.write_n_bit(code, bits);
    if piece.kind != PieceKind::Gold {
        w.write_one_bit(piece.promoted);
    }
    w.write_one_bit(piece.color == Color::White);
}

/// A hand piece, Huffman-coded (`write_hand_piece_to_stream`,
/// `sfen_packer.cpp`): the board code with its low bit dropped, then a
/// forced-unpromoted bit (except Gold), then a color bit.
fn write_hand_piece(w: &mut BitWriter, kind: PieceKind, color: Color) {
    let (code, bits) = huffman_board(kind);
    w.write_n_bit(code >> 1, bits - 1);
    if kind != PieceKind::Gold {
        w.write_one_bit(false);
    }
    w.write_one_bit(color == Color::White);
}

/// A piece-box piece, Huffman-coded (`write_piecebox_piece_to_stream`,
/// `sfen_packer.cpp`): the piece-box code, then a zero color bit
/// (except Gold, which encodes its color implicitly).
fn write_piece_box_piece(w: &mut BitWriter, kind: PieceKind) {
    let (code, bits) = huffman_piece_box(kind);
    w.write_n_bit(code, bits);
    if kind != PieceKind::Gold {
        w.write_one_bit(false);
    }
}

/// Locate a color's king by scanning the board in index order; `SQ_NB` (81) if
/// the king is absent (mirrors `pos.square<KING>(c)` returning `SQ_NB`).
fn king_square(pos: &Position, color: Color) -> u32 {
    for index in 0..Square::COUNT as u8 {
        let sq = Square::from_index(index).expect("index < COUNT");
        if let Some(p) = pos.board().get(sq)
            && p.kind == PieceKind::King
            && p.color == color
        {
            return sq.index() as u32;
        }
    }
    Square::COUNT as u32
}

/// Pack a [`Position`] into its 32-byte [`PackedSfen`], bit-identical to the
/// reference `Position::sfen_pack`.
pub fn sfen_pack(pos: &Position) -> PackedSfen {
    let mut w = BitWriter::new();

    w.write_one_bit(pos.side_to_move() == Color::White);

    w.write_n_bit(king_square(pos, Color::Black), 7);
    w.write_n_bit(king_square(pos, Color::White), 7);

    // Kings carry their square above, so the board pass skips them.
    let mut box_count = PIECE_BOX_START;
    for index in 0..Square::COUNT as u8 {
        let sq = Square::from_index(index).expect("index < COUNT");
        match pos.board().get(sq) {
            Some(p) if p.kind == PieceKind::King => {}
            Some(p) => {
                write_board_piece(&mut w, p);
                box_count[p.kind.index()] -= 1;
            }
            None => w.write_one_bit(false),
        }
    }

    for color in [Color::Black, Color::White] {
        for kind in HAND_ORDER {
            let n = pos.hand(color).count(kind);
            for _ in 0..n {
                write_hand_piece(&mut w, kind, color);
            }
            box_count[kind.index()] -= i32::from(n);
        }
    }

    for kind in HAND_ORDER {
        let leftover = box_count[kind.index()].max(0);
        for _ in 0..leftover {
            write_piece_box_piece(&mut w, kind);
        }
    }

    debug_assert_eq!(
        w.cursor, 256,
        "packed sfen must be exactly 256 bits for a legal position"
    );
    w.data
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sfen::parse_sfen;

    /// Parse an SFEN that may omit the trailing ply field (the reference's own
    /// PackedSfen test vectors are 3-field). Ply does not affect packing.
    fn parse(sfen: &str) -> Position {
        let with_ply = if sfen.split(' ').count() == 3 {
            format!("{sfen} 1")
        } else {
            sfen.to_string()
        };
        parse_sfen(&with_ply).expect("valid sfen")
    }

    /// Vectors transcribed from the reference's PackedSfen unit test
    /// (`position.cpp`), whose bytes were produced by cshogi
    /// (`board.to_psfen`) — an implementation independent of both.
    #[test]
    fn matches_reference_cshogi_vectors() {
        let cases: [(&str, [u8; 32]); 4] = [
            (
                "lnsgkgsnl/9/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL w -",
                [
                    89, 164, 81, 34, 12, 171, 68, 252, 44, 167, 68, 56, 94, 137, 240, 72, 132, 87,
                    34, 60, 167, 68, 56, 86, 137, 248, 88, 70, 137, 48, 188, 126,
                ],
            ),
            (
                "lns1kgsnl/9/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL w -",
                [
                    89, 164, 81, 34, 12, 171, 68, 252, 44, 167, 68, 56, 94, 137, 240, 72, 4, 18,
                    225, 57, 37, 194, 177, 74, 196, 199, 50, 74, 132, 97, 191, 126,
                ],
            ),
            (
                "lnsgkgsnl/9/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGK4 w -",
                [
                    89, 164, 81, 34, 88, 37, 226, 199, 41, 17, 188, 18, 129, 68, 120, 37, 194, 115,
                    74, 132, 99, 149, 136, 143, 101, 148, 8, 67, 106, 107, 191, 126,
                ],
            ),
            (
                "lnsgk4/9/ppppppppp/9/9/9/PPPPPPPPP/9/LNSGK4 w GBRgbr",
                [
                    89, 36, 18, 1, 137, 128, 68, 64, 34, 144, 8, 175, 68, 120, 78, 137, 112, 172,
                    18, 97, 25, 37, 194, 112, 30, 159, 251, 252, 166, 212, 218, 90,
                ],
            ),
        ];

        for (sfen, expected) in cases {
            let pos = parse(sfen);
            let packed = sfen_pack(&pos);
            assert_eq!(packed, expected, "packed sfen mismatch for {sfen}");
        }
    }

    #[test]
    fn ply_does_not_affect_packing() {
        let a =
            parse_sfen("lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1").unwrap();
        let b =
            parse_sfen("lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 99").unwrap();
        assert_eq!(sfen_pack(&a), sfen_pack(&b));
    }

    #[test]
    fn side_to_move_flips_bit0() {
        let black =
            parse_sfen("lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1").unwrap();
        let white =
            parse_sfen("lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL w - 1").unwrap();
        assert_eq!(sfen_pack(&black)[0] & 1, 0);
        assert_eq!(sfen_pack(&white)[0] & 1, 1);
    }

    #[test]
    fn fixture_sfens_pack_to_full_width() {
        // Promoted board pieces, both-color hands and sparse boards must pack
        // without tripping the 256-bit debug assert; the cshogi vectors above
        // do not reach them.
        for sfen in [
            "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1",
            "4k4/9/4r4/9/9/9/4K3B/9/9 b RG2gs2n3p 1",
            "k8/1P7/G8/1N2P4/9/9/9/9/8K b 2PG2pg 1",
            "l7l/1r1sg2k1/2nppgsp1/p1p3p1p/1p2N4/2P1P1P2/PPSP1PB1P/3GG1SR1/LN2K3L b BNPp 1",
            "4k4/3P3+PL/2N2PR2/1L2BNS2/4N4/9/9/9/4K4 b - 1",
            "9/4k4/9/9/9/9/9/4K4/9 b 9P9p 1",
        ] {
            let pos = parse_sfen(sfen).unwrap();
            let packed = sfen_pack(&pos);
            assert_eq!(packed.len(), 32, "sfen {sfen}");
        }
    }
}
