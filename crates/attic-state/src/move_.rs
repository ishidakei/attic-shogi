//! Packed 32-bit move encoding.
//!
//! The bit layout matches the reference engine (`types.h`, `916-928`)
//! exactly, so that move encodings round-trip through TT entries unchanged.
//!
//! ```text
//! bit:  31         21 20    16 15 14 13         7 6          0
//!        │   zero   │ piece  │ P│ D│  from / pt  │     to     │
//!        └──────────┴────────┴──┴──┴─────────────┴────────────┘
//!
//! Bits  0..6   (7 bits): to-square, 0..80 (Square::index)
//! Bits  7..13  (7 bits): from-square 0..80 for board moves;
//!                        OR YaneuraOu PieceType code (1..7) for drops
//! Bit   14            : DROP flag (1 = drop, 0 = board move)
//! Bit   15            : PROMOTE flag (1 = promote, 0 = no promote)
//! Bits 16..20  (5 bits): YaneuraOu Piece code of the piece at `to`
//!                        AFTER the move (the promoted form when PROMOTE=1).
//!                        PAWN=1 .. KING=8; +PIECE_PROMOTE(8); +PIECE_WHITE(16).
//! Bits 21..31         : zero.
//! ```
//!
//! [`PieceKind`] disagrees with the reference's `PieceType` ordering (`Gold=4`
//! here between `Silver` and `Bishop`; `BISHOP=5, ROOK=6, GOLD=7` there), so
//! every code crossing this boundary goes through [`PIECE_KIND_TO_REF`] or its
//! inverse [`REF_TO_PIECE_KIND`].

use core::fmt;

use crate::color::Color;
use crate::piece::{Piece, PieceKind};
use crate::position::Position;
use crate::square::Square;

/// Reference flag: drop (`MOVE_DROP` in `types.h`).
const FLAG_DROP: u32 = 1 << 14;
/// Reference flag: promote (`MOVE_PROMOTE` in `types.h`).
const FLAG_PROMOTE: u32 = 1 << 15;
/// Reference offset: PIECE_PROMOTE (added to a `PieceType` code to mark it promoted).
const PIECE_PROMOTE: u32 = 8;
/// Reference offset: PIECE_WHITE (added to a `Piece` code to mark it white).
const PIECE_WHITE: u32 = 16;

/// Map our `PieceKind` discriminant to the reference's `PieceType` code (1..8).
const PIECE_KIND_TO_REF: [u32; 8] = [
    1, // Pawn
    2, // Lance
    3, // Knight
    4, // Silver
    7, // Gold
    5, // Bishop
    6, // Rook
    8, // King
];

/// Inverse of [`PIECE_KIND_TO_REF`], indexed by the reference 4-bit `PieceType`
/// code (`pc & 0x0F`). Yields `(PieceKind, promoted)`. `None` for unused codes
/// (`0` = NO_PIECE_TYPE; `15` = a non-move special value in the reference).
const REF_TO_PIECE_KIND: [Option<(PieceKind, bool)>; 16] = [
    None,                             // 0  NO_PIECE_TYPE
    Some((PieceKind::Pawn, false)),   // 1  PAWN
    Some((PieceKind::Lance, false)),  // 2  LANCE
    Some((PieceKind::Knight, false)), // 3  KNIGHT
    Some((PieceKind::Silver, false)), // 4  SILVER
    Some((PieceKind::Bishop, false)), // 5  BISHOP
    Some((PieceKind::Rook, false)),   // 6  ROOK
    Some((PieceKind::Gold, false)),   // 7  GOLD
    Some((PieceKind::King, false)),   // 8  KING
    Some((PieceKind::Pawn, true)),    // 9  PRO_PAWN
    Some((PieceKind::Lance, true)),   // 10 PRO_LANCE
    Some((PieceKind::Knight, true)),  // 11 PRO_KNIGHT
    Some((PieceKind::Silver, true)),  // 12 PRO_SILVER
    Some((PieceKind::Bishop, true)),  // 13 HORSE
    Some((PieceKind::Rook, true)),    // 14 DRAGON
    None,                             // 15 (B_GOLDS / W_GOLDS — never on a Move)
];

const fn piece_to_ref_code(piece: Piece) -> u32 {
    let pt = PIECE_KIND_TO_REF[piece.kind.index()];
    let promote_bit = if piece.promoted { PIECE_PROMOTE } else { 0 };
    let color_bit = match piece.color {
        Color::Black => 0,
        Color::White => PIECE_WHITE,
    };
    pt | promote_bit | color_bit
}

/// Packed 32-bit move whose bit layout matches the YaneuraOu reference.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Move(u32);

impl Move {
    /// `MOVE_NONE` — invalid / unset move.
    pub const fn none() -> Self {
        Self(0)
    }

    /// `MOVE_NULL` — null-move sentinel (`(1 << 7) + 1`).
    pub const fn null() -> Self {
        Self((1 << 7) + 1)
    }

    /// `MOVE_RESIGN` — resignation sentinel (`(2 << 7) + 2`).
    pub const fn resign() -> Self {
        Self((2 << 7) + 2)
    }

    /// `MOVE_WIN` — declaration-of-win sentinel (`(3 << 7) + 3`).
    pub const fn win() -> Self {
        Self((3 << 7) + 3)
    }

    /// Construct a board move (no promotion). `piece` is the piece on `from`.
    pub fn make(from: Square, to: Square, piece: Piece) -> Self {
        let bits =
            (to.index() as u32) | ((from.index() as u32) << 7) | (piece_to_ref_code(piece) << 16);
        Self(bits)
    }

    /// Construct a board move with promotion. `piece` is the unpromoted piece
    /// on `from`; the upper-bits Piece is stored as the promoted form.
    pub fn make_promote(from: Square, to: Square, piece: Piece) -> Self {
        let promoted_code = piece_to_ref_code(piece) | PIECE_PROMOTE;
        let bits = (to.index() as u32)
            | ((from.index() as u32) << 7)
            | FLAG_PROMOTE
            | (promoted_code << 16);
        Self(bits)
    }

    /// Construct a drop. King has no legal drop in shogi, but the encoding does
    /// not enforce it.
    pub fn make_drop(kind: PieceKind, color: Color, to: Square) -> Self {
        let pt = PIECE_KIND_TO_REF[kind.index()];
        let color_bit = match color {
            Color::Black => 0,
            Color::White => PIECE_WHITE,
        };
        let bits = (to.index() as u32) | (pt << 7) | FLAG_DROP | ((pt | color_bit) << 16);
        Self(bits)
    }

    /// Wrap a raw u32 (e.g. one read from a TT entry) without translation.
    pub const fn from_bits(bits: u32) -> Self {
        Self(bits)
    }

    /// Raw 32-bit representation.
    pub const fn to_bits(self) -> u32 {
        self.0
    }

    /// The 16-bit move fragment (`Move16`): the packed move without its upper
    /// piece-code bits. This is the value the `.ybb` opening book stores.
    pub const fn move16(self) -> u16 {
        (self.0 & 0xFFFF) as u16
    }

    /// Drop flag (bit 14).
    pub const fn is_drop(self) -> bool {
        (self.0 & FLAG_DROP) != 0
    }

    /// Promote flag (bit 15).
    pub const fn is_promote(self) -> bool {
        (self.0 & FLAG_PROMOTE) != 0
    }

    /// Reference's `is_ok` predicate. False for `MOVE_NONE/NULL/RESIGN/WIN`,
    /// true for any move constructed via [`make`](Self::make),
    /// [`make_promote`](Self::make_promote), or [`make_drop`](Self::make_drop).
    pub const fn is_ok(self) -> bool {
        (self.0 >> 7) != (self.0 & 0x7f)
    }

    /// Destination square.
    ///
    /// # Panics
    /// Panics if the encoded `to` index is out of range; this cannot happen
    /// for a `Move` produced by the constructors in this module.
    pub fn to_sq(self) -> Square {
        Square::from_index((self.0 & 0x7f) as u8).expect("Move::to_sq: malformed move data")
    }

    /// Origin square. Only valid for non-drop moves.
    ///
    /// # Panics
    /// Panics if `self.is_drop()` (the bits hold a `PieceType` code, not a
    /// square) or if the encoded `from` index is out of range.
    pub fn from_sq(self) -> Square {
        debug_assert!(!self.is_drop(), "Move::from_sq called on a drop");
        Square::from_index(((self.0 >> 7) & 0x7f) as u8)
            .expect("Move::from_sq: malformed move data")
    }

    /// Piece kind being dropped if `self` is a well-formed drop whose piece
    /// field is a bare droppable `PieceType` (reference codes `1..=7`); `None`
    /// for a non-drop or an out-of-range field.
    ///
    /// Total over every bit pattern, so [`Position::to_move`] can hand it a
    /// possibly-torn 16-bit TT fragment.
    pub fn dropped_piece_kind_checked(self) -> Option<PieceKind> {
        if !self.is_drop() {
            return None;
        }
        let field = ((self.0 >> 7) & 0x7f) as usize;
        if !(1..=7).contains(&field) {
            return None;
        }
        REF_TO_PIECE_KIND[field].map(|(kind, _)| kind)
    }

    /// Piece kind being dropped. Only valid for drops.
    ///
    /// # Panics
    /// Panics if `!self.is_drop()` or if the encoded code is not 1..=7.
    pub fn dropped_piece_kind(self) -> PieceKind {
        debug_assert!(
            self.is_drop(),
            "Move::dropped_piece_kind called on a non-drop"
        );
        let code = ((self.0 >> 7) & 0x7f) as usize;
        REF_TO_PIECE_KIND[code & 0x0F]
            .expect("Move::dropped_piece_kind: malformed move data")
            .0
    }

    /// Piece occupying `to` *after* the move (promoted form when
    /// `is_promote()`).
    ///
    /// # Panics
    /// Panics if the upper 5 bits encode an invalid piece (e.g. on
    /// `MOVE_NONE`, where they are zero).
    pub fn moved_piece_after(self) -> Piece {
        let code = (self.0 >> 16) & 0x1F;
        let color = if (code & PIECE_WHITE) != 0 {
            Color::White
        } else {
            Color::Black
        };
        let pt_bits = (code & 0x0F) as usize;
        let (kind, promoted) =
            REF_TO_PIECE_KIND[pt_bits].expect("Move::moved_piece_after: malformed move data");
        Piece {
            kind,
            color,
            promoted,
        }
    }
}

/// Flip a 16-bit move fragment to the one that plays the identical move on the
/// board rotated 180° (`flip_move`, `types.h`), where
/// `Flip(sq) = 80 - sq`.
///
/// A `.ybb` entry for the color-flipped position stores moves for the rotated
/// board, and this maps each back onto the real position. The result is a raw
/// fragment that must still be widened against the real position's legal moves
/// before use; fields are masked to 7 bits, so a malformed input produces
/// nothing outside the `move16` layout and simply fails to widen.
pub const fn flip_move16(m: u16) -> u16 {
    const TO_MASK: u16 = 0x7f;
    let to = m & TO_MASK;
    let flip_to = (80u16.wrapping_sub(to)) & TO_MASK;
    if (m & (FLAG_DROP as u16)) != 0 {
        // Bits 7..13 hold the dropped-piece code, not a square, so they survive
        // the rotation unchanged.
        let pt = (m >> 7) & TO_MASK;
        flip_to | (pt << 7) | (FLAG_DROP as u16)
    } else {
        let from = (m >> 7) & TO_MASK;
        let flip_from = (80u16.wrapping_sub(from)) & TO_MASK;
        let promote = m & (FLAG_PROMOTE as u16);
        flip_to | (flip_from << 7) | promote
    }
}

/// Parse a USI move string (e.g. `7g7f`, `8h2b+`, `P*5e`) into a [`Move`].
///
/// USI files are `1..=9` with file `1` on the right from Black's view and ranks
/// are `a..=i` with rank `a` at the top, so internal coordinates are
/// `usi_file - 1` and `usi_rank - 1`.
///
/// The move is not checked for legality: a syntactically well-formed string
/// always yields an encoded `Move`.
pub fn parse_usi_move(s: &str, pos: &Position) -> Result<Move, UsiMoveParseError> {
    let bytes = s.as_bytes();
    if bytes.is_empty() {
        return Err(UsiMoveParseError::Empty);
    }

    if bytes.len() == 4 && bytes[1] == b'*' {
        let kind = parse_drop_piece(bytes[0])?;
        let to = parse_square(bytes[2], bytes[3])?;
        return Ok(Move::make_drop(kind, pos.side_to_move(), to));
    }

    if bytes.len() != 4 && bytes.len() != 5 {
        return Err(UsiMoveParseError::InvalidLength(bytes.len()));
    }

    let from = parse_square(bytes[0], bytes[1])?;
    let to = parse_square(bytes[2], bytes[3])?;
    let promote = match bytes.get(4) {
        None => false,
        Some(b'+') => true,
        Some(&c) => return Err(UsiMoveParseError::InvalidPromotionMarker(c as char)),
    };

    let piece = pos
        .board()
        .get(from)
        .ok_or(UsiMoveParseError::EmptyFromSquare)?;

    if promote {
        if piece.promoted {
            return Err(UsiMoveParseError::PromoteAlreadyPromoted);
        }
        Ok(Move::make_promote(from, to, piece))
    } else {
        Ok(Move::make(from, to, piece))
    }
}

/// Format a [`Move`] as a USI move string. Inverse of [`parse_usi_move`].
///
/// Behaviour on the non-move sentinels (`Move::none()`, `Move::null()`,
/// `Move::resign()`, `Move::win()`) is unspecified: those values do not encode
/// a square or piece in the layout this decodes.
pub fn format_usi_move(m: Move) -> String {
    if m.is_drop() {
        let kind = m.dropped_piece_kind();
        let letter = drop_letter(kind);
        let to = m.to_sq();
        format!(
            "{letter}*{}{}",
            file_to_usi(to.file()),
            rank_to_usi(to.rank()),
        )
    } else {
        let from = m.from_sq();
        let to = m.to_sq();
        let mut s = String::with_capacity(5);
        s.push(file_to_usi(from.file()));
        s.push(rank_to_usi(from.rank()));
        s.push(file_to_usi(to.file()));
        s.push(rank_to_usi(to.rank()));
        if m.is_promote() {
            s.push('+');
        }
        s
    }
}

fn drop_letter(kind: PieceKind) -> char {
    match kind {
        PieceKind::Pawn => 'P',
        PieceKind::Lance => 'L',
        PieceKind::Knight => 'N',
        PieceKind::Silver => 'S',
        PieceKind::Gold => 'G',
        PieceKind::Bishop => 'B',
        PieceKind::Rook => 'R',
        // USI has no King-drop notation, and no legal sequence reaches this
        // bits pattern even though `make_drop` accepts King.
        PieceKind::King => panic!("format_usi_move: King drops have no USI representation"),
    }
}

fn file_to_usi(file: u8) -> char {
    debug_assert!(file < Square::FILES, "file out of range");
    (b'1' + file) as char
}

fn rank_to_usi(rank: u8) -> char {
    debug_assert!(rank < Square::RANKS, "rank out of range");
    (b'a' + rank) as char
}

fn parse_square(file_byte: u8, rank_byte: u8) -> Result<Square, UsiMoveParseError> {
    if !(b'1'..=b'9').contains(&file_byte) {
        return Err(UsiMoveParseError::InvalidFile(file_byte as char));
    }
    if !(b'a'..=b'i').contains(&rank_byte) {
        return Err(UsiMoveParseError::InvalidRank(rank_byte as char));
    }
    let file = file_byte - b'1';
    let rank = rank_byte - b'a';
    Square::new(file, rank).ok_or(UsiMoveParseError::InvalidFile(file_byte as char))
}

fn parse_drop_piece(byte: u8) -> Result<PieceKind, UsiMoveParseError> {
    match byte {
        b'P' => Ok(PieceKind::Pawn),
        b'L' => Ok(PieceKind::Lance),
        b'N' => Ok(PieceKind::Knight),
        b'S' => Ok(PieceKind::Silver),
        b'G' => Ok(PieceKind::Gold),
        b'B' => Ok(PieceKind::Bishop),
        b'R' => Ok(PieceKind::Rook),
        c => Err(UsiMoveParseError::InvalidDropPiece(c as char)),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UsiMoveParseError {
    Empty,
    InvalidLength(usize),
    InvalidFile(char),
    InvalidRank(char),
    InvalidDropPiece(char),
    InvalidPromotionMarker(char),
    EmptyFromSquare,
    PromoteAlreadyPromoted,
}

impl fmt::Display for UsiMoveParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty => f.write_str("usi-move: empty input"),
            Self::InvalidLength(n) => write!(f, "usi-move: invalid length {n} (expected 4 or 5)"),
            Self::InvalidFile(c) => write!(f, "usi-move: invalid file {c:?}"),
            Self::InvalidRank(c) => write!(f, "usi-move: invalid rank {c:?}"),
            Self::InvalidDropPiece(c) => write!(f, "usi-move: invalid drop piece {c:?}"),
            Self::InvalidPromotionMarker(c) => {
                write!(f, "usi-move: invalid promotion marker {c:?}")
            }
            Self::EmptyFromSquare => f.write_str("usi-move: from-square is empty on the board"),
            Self::PromoteAlreadyPromoted => {
                f.write_str("usi-move: cannot promote an already-promoted piece")
            }
        }
    }
}

impl std::error::Error for UsiMoveParseError {}

#[cfg(test)]
mod tests {
    use super::*;

    /// Each row's `bits` is derived by hand from `types.h`.
    struct Fixture {
        bits: u32,
        from: Option<(u8, u8)>,
        to: (u8, u8),
        piece_after: Piece,
        is_drop: bool,
        is_promote: bool,
        dropped: Option<PieceKind>,
    }

    fn p(kind: PieceKind, color: Color, promoted: bool) -> Piece {
        Piece {
            kind,
            color,
            promoted,
        }
    }

    fn fixtures() -> Vec<Fixture> {
        vec![
            // 7g7f: B pawn pushes from (file=6,rank=6)=60 to (file=6,rank=5)=59.
            Fixture {
                bits: 0x0001_1E3B,
                from: Some((6, 6)),
                to: (6, 5),
                piece_after: p(PieceKind::Pawn, Color::Black, false),
                is_drop: false,
                is_promote: false,
                dropped: None,
            },
            // 8c8d: W pawn from (file=7,rank=2)=65 to (file=7,rank=3)=66.
            Fixture {
                bits: 0x0011_20C2,
                from: Some((7, 2)),
                to: (7, 3),
                piece_after: p(PieceKind::Pawn, Color::White, false),
                is_drop: false,
                is_promote: false,
                dropped: None,
            },
            // 2h3h: B rook from (file=1,rank=7)=16 to (file=2,rank=7)=25.
            Fixture {
                bits: 0x0006_0819,
                from: Some((1, 7)),
                to: (2, 7),
                piece_after: p(PieceKind::Rook, Color::Black, false),
                is_drop: false,
                is_promote: false,
                dropped: None,
            },
            // 8h2b+: B bishop from (file=7,rank=7)=70 to (file=1,rank=1)=10,
            // promoted to horse (B_HORSE = 13).
            Fixture {
                bits: 0x000D_A30A,
                from: Some((7, 7)),
                to: (1, 1),
                piece_after: p(PieceKind::Bishop, Color::Black, true),
                is_drop: false,
                is_promote: true,
                dropped: None,
            },
            // P*5e (Black): drop pawn at (file=4,rank=4)=40.
            Fixture {
                bits: 0x0001_40A8,
                from: None,
                to: (4, 4),
                piece_after: p(PieceKind::Pawn, Color::Black, false),
                is_drop: true,
                is_promote: false,
                dropped: Some(PieceKind::Pawn),
            },
            // P*5e (White): drop pawn at (file=4,rank=4)=40.
            Fixture {
                bits: 0x0011_40A8,
                from: None,
                to: (4, 4),
                piece_after: p(PieceKind::Pawn, Color::White, false),
                is_drop: true,
                is_promote: false,
                dropped: Some(PieceKind::Pawn),
            },
        ]
    }

    fn build(fx: &Fixture) -> Move {
        let to = Square::new(fx.to.0, fx.to.1).unwrap();
        match (fx.from, fx.is_promote, fx.is_drop) {
            (Some((ff, fr)), false, false) => {
                Move::make(Square::new(ff, fr).unwrap(), to, fx.piece_after)
            }
            (Some((ff, fr)), true, false) => {
                let unpromoted = Piece {
                    promoted: false,
                    ..fx.piece_after
                };
                Move::make_promote(Square::new(ff, fr).unwrap(), to, unpromoted)
            }
            (None, false, true) => Move::make_drop(fx.dropped.unwrap(), fx.piece_after.color, to),
            other => panic!("unsupported fixture shape: {other:?}"),
        }
    }

    #[test]
    fn fixture_encodes_to_reference_bits() {
        for fx in fixtures() {
            let m = build(&fx);
            assert_eq!(
                m.to_bits(),
                fx.bits,
                "encode mismatch for fixture bits=0x{:08X}: got 0x{:08X}",
                fx.bits,
                m.to_bits(),
            );
        }
    }

    #[test]
    fn fixture_decodes_to_components() {
        for fx in fixtures() {
            let m = Move::from_bits(fx.bits);
            assert_eq!(
                m.is_drop(),
                fx.is_drop,
                "is_drop mismatch for 0x{:08X}",
                fx.bits
            );
            assert_eq!(
                m.is_promote(),
                fx.is_promote,
                "is_promote mismatch for 0x{:08X}",
                fx.bits
            );
            assert_eq!(
                m.to_sq(),
                Square::new(fx.to.0, fx.to.1).unwrap(),
                "to_sq mismatch for 0x{:08X}",
                fx.bits
            );
            if let Some((ff, fr)) = fx.from {
                assert_eq!(
                    m.from_sq(),
                    Square::new(ff, fr).unwrap(),
                    "from_sq mismatch for 0x{:08X}",
                    fx.bits
                );
            }
            if let Some(kind) = fx.dropped {
                assert_eq!(
                    m.dropped_piece_kind(),
                    kind,
                    "dropped_piece_kind mismatch for 0x{:08X}",
                    fx.bits
                );
            }
            assert_eq!(
                m.moved_piece_after(),
                fx.piece_after,
                "moved_piece_after mismatch for 0x{:08X}",
                fx.bits
            );
        }
    }

    #[test]
    fn round_trip_normal_moves_over_all_kinds() {
        let kinds = [
            PieceKind::Pawn,
            PieceKind::Lance,
            PieceKind::Knight,
            PieceKind::Silver,
            PieceKind::Gold,
            PieceKind::Bishop,
            PieceKind::Rook,
            PieceKind::King,
        ];
        let from = Square::new(4, 4).unwrap();
        let to = Square::new(4, 5).unwrap();
        for kind in kinds {
            for color in [Color::Black, Color::White] {
                let piece = Piece::new(kind, color);
                let m = Move::make(from, to, piece);
                assert_eq!(m.from_sq(), from);
                assert_eq!(m.to_sq(), to);
                assert_eq!(m.moved_piece_after(), piece);
                assert!(!m.is_drop());
                assert!(!m.is_promote());
                assert!(m.is_ok());
            }
        }
    }

    #[test]
    fn round_trip_promote_moves_over_promotable_kinds() {
        let promotable = [
            PieceKind::Pawn,
            PieceKind::Lance,
            PieceKind::Knight,
            PieceKind::Silver,
            PieceKind::Bishop,
            PieceKind::Rook,
        ];
        let from = Square::new(7, 7).unwrap();
        let to = Square::new(1, 1).unwrap();
        for kind in promotable {
            for color in [Color::Black, Color::White] {
                let unpromoted = Piece::new(kind, color);
                let m = Move::make_promote(from, to, unpromoted);
                assert_eq!(m.from_sq(), from);
                assert_eq!(m.to_sq(), to);
                assert!(m.is_promote());
                assert!(!m.is_drop());
                let after = m.moved_piece_after();
                assert_eq!(after.kind, kind);
                assert_eq!(after.color, color);
                assert!(after.promoted);
            }
        }
    }

    #[test]
    fn round_trip_drops_over_handable_kinds() {
        let handable = [
            PieceKind::Pawn,
            PieceKind::Lance,
            PieceKind::Knight,
            PieceKind::Silver,
            PieceKind::Gold,
            PieceKind::Bishop,
            PieceKind::Rook,
        ];
        let to = Square::new(4, 4).unwrap();
        for kind in handable {
            for color in [Color::Black, Color::White] {
                let m = Move::make_drop(kind, color, to);
                assert!(m.is_drop());
                assert!(!m.is_promote());
                assert_eq!(m.to_sq(), to);
                assert_eq!(m.dropped_piece_kind(), kind);
                let after = m.moved_piece_after();
                assert_eq!(after.kind, kind);
                assert_eq!(after.color, color);
                assert!(!after.promoted);
                assert!(m.is_ok());
            }
        }
    }

    #[test]
    fn round_trip_boundary_squares() {
        let zero = Square::from_index(0).unwrap();
        let last = Square::from_index(80).unwrap();
        let pawn = Piece::new(PieceKind::Pawn, Color::Black);
        let m = Move::make(zero, last, pawn);
        assert_eq!(m.from_sq(), zero);
        assert_eq!(m.to_sq(), last);
        let m = Move::make(last, zero, pawn);
        assert_eq!(m.from_sq(), last);
        assert_eq!(m.to_sq(), zero);
    }

    #[test]
    fn raw_round_trip() {
        let from = Square::new(2, 3).unwrap();
        let to = Square::new(5, 6).unwrap();
        let piece = Piece::new(PieceKind::Silver, Color::White);
        let m = Move::make(from, to, piece);
        let bits = m.to_bits();
        assert_eq!(Move::from_bits(bits), m);
    }

    #[test]
    fn is_ok_rejects_sentinels_and_accepts_normal_moves() {
        assert!(!Move::none().is_ok());
        assert!(!Move::null().is_ok());
        assert!(!Move::resign().is_ok());
        assert!(!Move::win().is_ok());

        let normal = Move::make(
            Square::new(6, 6).unwrap(),
            Square::new(6, 5).unwrap(),
            Piece::new(PieceKind::Pawn, Color::Black),
        );
        assert!(normal.is_ok());

        let drop = Move::make_drop(PieceKind::Pawn, Color::Black, Square::new(4, 4).unwrap());
        assert!(drop.is_ok());

        let promote = Move::make_promote(
            Square::new(7, 7).unwrap(),
            Square::new(1, 1).unwrap(),
            Piece::new(PieceKind::Bishop, Color::Black),
        );
        assert!(promote.is_ok());
    }

    #[test]
    fn sentinel_bit_values_match_reference() {
        assert_eq!(Move::none().to_bits(), 0);
        assert_eq!(Move::null().to_bits(), (1 << 7) + 1);
        assert_eq!(Move::resign().to_bits(), (2 << 7) + 2);
        assert_eq!(Move::win().to_bits(), (3 << 7) + 3);
    }

    mod flip_move16 {
        use super::super::flip_move16;
        use crate::Color;
        use crate::move_::Move;
        use crate::piece::{Piece, PieceKind};
        use crate::square::Square;

        fn sq(file: u8, rank: u8) -> Square {
            Square::new(file, rank).unwrap()
        }

        /// `Flip(sq) = 80 - sq` — the 180° rotation the reference `Flip` applies.
        fn flip_sq(s: Square) -> Square {
            Square::from_index(80 - s.index()).unwrap()
        }

        #[test]
        fn normal_move_flips_from_and_to() {
            let m = Move::make(
                sq(6, 6),
                sq(6, 5),
                Piece::new(PieceKind::Pawn, Color::Black),
            );
            let f = flip_move16(m.move16());
            let expected = Move::make(
                flip_sq(sq(6, 6)),
                flip_sq(sq(6, 5)),
                Piece::new(PieceKind::Pawn, Color::Black),
            );
            assert_eq!(f, expected.move16());
        }

        #[test]
        fn promote_move_flips_and_keeps_promote_flag() {
            let m = Move::make_promote(
                sq(7, 7),
                sq(1, 1),
                Piece::new(PieceKind::Bishop, Color::Black),
            );
            let f = flip_move16(m.move16());
            let expected = Move::make_promote(
                flip_sq(sq(7, 7)),
                flip_sq(sq(1, 1)),
                Piece::new(PieceKind::Bishop, Color::Black),
            );
            assert_eq!(f, expected.move16());
            assert_ne!(f & (1 << 15), 0);
        }

        #[test]
        fn drop_flips_to_and_keeps_piece_code() {
            let m = Move::make_drop(PieceKind::Pawn, Color::Black, sq(4, 4));
            let f = flip_move16(m.move16());
            // The dropped-piece code and drop flag are color-independent, so the
            // flipped fragment equals a same-kind drop on the flipped square.
            let expected = Move::make_drop(PieceKind::Pawn, Color::White, flip_sq(sq(4, 4)));
            assert_eq!(f, expected.move16());
            assert_ne!(f & (1 << 14), 0);
        }

        #[test]
        fn flip_is_an_involution() {
            for m in [
                Move::make(
                    sq(0, 0),
                    sq(8, 8),
                    Piece::new(PieceKind::Rook, Color::Black),
                )
                .move16(),
                Move::make_promote(
                    sq(2, 6),
                    sq(2, 8),
                    Piece::new(PieceKind::Lance, Color::White),
                )
                .move16(),
                Move::make_drop(PieceKind::Silver, Color::Black, sq(3, 3)).move16(),
            ] {
                assert_eq!(flip_move16(flip_move16(m)), m);
            }
        }
    }

    mod parse_usi_move {
        use super::*;
        use crate::sfen::{STARTPOS_SFEN, parse_sfen};

        const SENNICHITE_SFEN: &str = "9/4k4/9/9/9/9/9/4K4/9 b 9P9p 1";

        #[test]
        fn board_move_no_promotion_matches_make() {
            let pos = parse_sfen(STARTPOS_SFEN).unwrap();
            let parsed = parse_usi_move("7g7f", &pos).unwrap();
            let expected = Move::make(
                Square::new(6, 6).unwrap(),
                Square::new(6, 5).unwrap(),
                Piece::new(PieceKind::Pawn, Color::Black),
            );
            assert_eq!(parsed, expected);
            assert_eq!(parsed.to_bits(), 0x0001_1E3B);
        }

        #[test]
        fn sennichite_king_shuffle_parses() {
            let pos = parse_sfen(SENNICHITE_SFEN).unwrap();
            let parsed = parse_usi_move("5h4h", &pos).unwrap();
            let expected = Move::make(
                Square::new(4, 7).unwrap(),
                Square::new(3, 7).unwrap(),
                Piece::new(PieceKind::King, Color::Black),
            );
            assert_eq!(parsed, expected);
        }

        #[test]
        fn promote_parses_and_round_trips() {
            let pos = parse_sfen(STARTPOS_SFEN).unwrap();
            let parsed = parse_usi_move("8h2b+", &pos).unwrap();
            assert!(parsed.is_promote());
            assert!(!parsed.is_drop());
            assert_eq!(parsed.from_sq(), Square::new(7, 7).unwrap());
            assert_eq!(parsed.to_sq(), Square::new(1, 1).unwrap());
            let after = parsed.moved_piece_after();
            assert_eq!(after.kind, PieceKind::Bishop);
            assert_eq!(after.color, Color::Black);
            assert!(after.promoted);
        }

        #[test]
        fn drop_uses_side_to_move_color() {
            let pos = parse_sfen(SENNICHITE_SFEN).unwrap();
            let parsed = parse_usi_move("P*5e", &pos).unwrap();
            assert!(parsed.is_drop());
            assert_eq!(parsed.to_sq(), Square::new(4, 4).unwrap());
            assert_eq!(parsed.dropped_piece_kind(), PieceKind::Pawn);
            assert_eq!(parsed.moved_piece_after().color, Color::Black);
        }

        #[test]
        fn drop_uses_white_when_white_to_move() {
            let mut pos = parse_sfen(SENNICHITE_SFEN).unwrap();
            pos.set_side_to_move(Color::White);
            let parsed = parse_usi_move("P*5e", &pos).unwrap();
            assert_eq!(parsed.moved_piece_after().color, Color::White);
        }

        #[test]
        fn empty_input_errors() {
            let pos = parse_sfen(STARTPOS_SFEN).unwrap();
            assert_eq!(parse_usi_move("", &pos), Err(UsiMoveParseError::Empty));
        }

        #[test]
        fn wrong_length_errors() {
            let pos = parse_sfen(STARTPOS_SFEN).unwrap();
            assert_eq!(
                parse_usi_move("7g7", &pos),
                Err(UsiMoveParseError::InvalidLength(3))
            );
            assert_eq!(
                parse_usi_move("7g7f7f", &pos),
                Err(UsiMoveParseError::InvalidLength(6))
            );
        }

        #[test]
        fn invalid_file_or_rank_errors() {
            let pos = parse_sfen(STARTPOS_SFEN).unwrap();
            assert_eq!(
                parse_usi_move("0a1a", &pos),
                Err(UsiMoveParseError::InvalidFile('0'))
            );
            assert_eq!(
                parse_usi_move("1j1a", &pos),
                Err(UsiMoveParseError::InvalidRank('j'))
            );
        }

        #[test]
        fn fifth_byte_must_be_plus() {
            let pos = parse_sfen(STARTPOS_SFEN).unwrap();
            assert_eq!(
                parse_usi_move("7g7fx", &pos),
                Err(UsiMoveParseError::InvalidPromotionMarker('x'))
            );
        }

        #[test]
        fn empty_from_square_errors() {
            let pos = parse_sfen(STARTPOS_SFEN).unwrap();
            assert_eq!(
                parse_usi_move("5e5d", &pos),
                Err(UsiMoveParseError::EmptyFromSquare)
            );
        }

        #[test]
        fn promote_already_promoted_errors() {
            // An SFEN rank reads file 8 → 0, so `1+B6K` puts the promoted bishop
            // at file 7 — USI 8h.
            let sfen = "9/9/9/9/9/9/9/1+B6K/9 b - 1";
            let pos = parse_sfen(sfen).unwrap();
            assert_eq!(
                parse_usi_move("8h2b+", &pos),
                Err(UsiMoveParseError::PromoteAlreadyPromoted)
            );
        }

        #[test]
        fn invalid_drop_piece_errors() {
            let pos = parse_sfen(STARTPOS_SFEN).unwrap();
            assert_eq!(
                parse_usi_move("K*5e", &pos),
                Err(UsiMoveParseError::InvalidDropPiece('K'))
            );
        }
    }

    mod format_usi_move {
        use super::*;
        use crate::sfen::{STARTPOS_SFEN, parse_sfen};

        const SENNICHITE_SFEN: &str = "9/4k4/9/9/9/9/9/4K4/9 b 9P9p 1";
        const ALL_DROPS_SFEN: &str = "9/4k4/9/9/9/9/9/4K4/9 b RBGSNLP 1";

        #[test]
        fn board_move_at_startpos_round_trips() {
            let pos = parse_sfen(STARTPOS_SFEN).unwrap();
            let m = parse_usi_move("7g7f", &pos).unwrap();
            assert_eq!(format_usi_move(m), "7g7f");
        }

        #[test]
        fn promotion_round_trips() {
            let pos = parse_sfen(STARTPOS_SFEN).unwrap();
            let m = parse_usi_move("8h2b+", &pos).unwrap();
            assert_eq!(format_usi_move(m), "8h2b+");
        }

        #[test]
        fn pawn_drop_round_trips() {
            let pos = parse_sfen(SENNICHITE_SFEN).unwrap();
            let m = parse_usi_move("P*5e", &pos).unwrap();
            assert_eq!(format_usi_move(m), "P*5e");
        }

        #[test]
        fn every_drop_letter_round_trips() {
            let pos = parse_sfen(ALL_DROPS_SFEN).unwrap();
            let mut moves: Vec<Move> = Vec::new();
            pos.generate_legal_all(&mut moves);
            let mut seen_letters: std::collections::HashSet<char> = Default::default();
            for m in &moves {
                if !m.is_drop() {
                    continue;
                }
                let s = format_usi_move(*m);
                let letter = s.chars().next().unwrap();
                seen_letters.insert(letter);
                let reparsed = parse_usi_move(&s, &pos).unwrap();
                assert_eq!(reparsed, *m, "drop round-trip failed for {s}");
            }
            for expected in ['P', 'L', 'N', 'S', 'G', 'B', 'R'] {
                assert!(
                    seen_letters.contains(&expected),
                    "movegen did not produce a {expected}-drop from ALL_DROPS_SFEN",
                );
            }
        }

        #[test]
        fn every_legal_move_at_startpos_round_trips() {
            let pos = parse_sfen(STARTPOS_SFEN).unwrap();
            let mut moves: Vec<Move> = Vec::new();
            pos.generate_legal_all(&mut moves);
            assert!(!moves.is_empty(), "startpos has legal moves");
            for m in moves {
                let s = format_usi_move(m);
                let reparsed = parse_usi_move(&s, &pos).unwrap();
                assert_eq!(reparsed, m, "round-trip failed for {s}");
            }
        }

        #[test]
        fn corner_squares_format_correctly() {
            let pos = parse_sfen(STARTPOS_SFEN).unwrap();
            let m = Move::make(
                Square::new(0, 0).unwrap(),
                Square::new(8, 8).unwrap(),
                Piece::new(PieceKind::Pawn, Color::Black),
            );
            assert_eq!(format_usi_move(m), "1a9i");
            let m = Move::make(
                Square::new(8, 8).unwrap(),
                Square::new(0, 0).unwrap(),
                Piece::new(PieceKind::Pawn, Color::Black),
            );
            assert_eq!(format_usi_move(m), "9i1a");
            let m = Move::make_drop(PieceKind::Pawn, Color::Black, Square::new(0, 8).unwrap());
            assert_eq!(format_usi_move(m), "P*1i");
            let m = Move::make_drop(PieceKind::Rook, Color::White, Square::new(8, 0).unwrap());
            assert_eq!(format_usi_move(m), "R*9a");
            let _ = pos;
        }
    }
}
