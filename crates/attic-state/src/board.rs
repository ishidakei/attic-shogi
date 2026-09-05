use crate::bitboard::Bitboard;
use crate::color::Color;
use crate::piece::{Piece, PieceKind};
use crate::square::Square;

/// The number of distinct attack patterns tracked by [`Board`]'s per-colour
/// piece sets (see [`pattern_of`]).
pub(crate) const PATTERN_COUNT: usize = 10;

/// The attack-pattern slot indices, named. The four promoted minors collapse to
/// [`pat::GOLD`]; horse and dragon are distinct. This is the reference
/// `type_of(...)` index into `st->checkSquares[]`.
pub(crate) mod pat {
    pub const PAWN: usize = 0;
    pub const LANCE: usize = 1;
    pub const KNIGHT: usize = 2;
    pub const SILVER: usize = 3;
    pub const GOLD: usize = 4;
    pub const BISHOP: usize = 5;
    pub const ROOK: usize = 6;
    pub const HORSE: usize = 7;
    pub const DRAGON: usize = 8;
    pub const KING: usize = 9;
}

/// Map a concrete piece to its [`pat`] attack-pattern slot.
pub(crate) const fn pattern_of(piece: Piece) -> usize {
    match (piece.kind, piece.promoted) {
        (PieceKind::Pawn, false) => pat::PAWN,
        (PieceKind::Lance, false) => pat::LANCE,
        (PieceKind::Knight, false) => pat::KNIGHT,
        (PieceKind::Silver, false) => pat::SILVER,
        (PieceKind::Gold, _)
        | (PieceKind::Pawn | PieceKind::Lance | PieceKind::Knight | PieceKind::Silver, true) => {
            pat::GOLD
        }
        (PieceKind::Bishop, false) => pat::BISHOP,
        (PieceKind::Rook, false) => pat::ROOK,
        (PieceKind::Bishop, true) => pat::HORSE,
        (PieceKind::Rook, true) => pat::DRAGON,
        (PieceKind::King, _) => pat::KING,
    }
}

/// The 81-square board and the piece-set bitboards maintained incrementally
/// alongside it. `squares` is the source of truth; `occupied`, `by_color`, and
/// `by_pattern` are derived caches, and [`Board::set`] is the single mutation
/// funnel that keeps them in sync (the reference's `put_piece` / `remove_piece`
/// XOR maintenance of `byColorBB` / `byTypeBB`).
#[derive(Debug, Clone, Copy)]
pub struct Board {
    squares: [Option<Piece>; Square::COUNT],
    /// Every occupied square.
    occupied: Bitboard,
    /// Occupied squares per colour, indexed by [`Color::index`].
    by_color: [Bitboard; Color::COUNT],
    /// Occupied squares per `(colour, pattern)`; pattern is [`pattern_of`].
    by_pattern: [[Bitboard; PATTERN_COUNT]; Color::COUNT],
}

/// Equality compares only the piece placement: the derived sets are a pure
/// function of `squares`, so equal squares imply equal sets.
impl PartialEq for Board {
    fn eq(&self, other: &Self) -> bool {
        self.squares == other.squares
    }
}

impl Eq for Board {}

impl Board {
    pub const fn empty() -> Self {
        Self {
            squares: [None; Square::COUNT],
            occupied: Bitboard::EMPTY,
            by_color: [Bitboard::EMPTY; Color::COUNT],
            by_pattern: [[Bitboard::EMPTY; PATTERN_COUNT]; Color::COUNT],
        }
    }

    pub const fn get(&self, sq: Square) -> Option<Piece> {
        self.squares[sq.index() as usize]
    }

    pub fn set(&mut self, sq: Square, piece: Option<Piece>) {
        let i = sq.index() as usize;
        if let Some(old) = self.squares[i] {
            self.toggle_sets(old, sq);
        }
        self.squares[i] = piece;
        if let Some(new) = piece {
            self.toggle_sets(new, sq);
        }
    }

    /// XOR `piece`'s bit into the derived piece sets. Self-inverse: the same
    /// call both adds and removes.
    fn toggle_sets(&mut self, piece: Piece, sq: Square) {
        let bit = Bitboard::from_square(sq);
        self.occupied ^= bit;
        self.by_color[piece.color.index()] ^= bit;
        self.by_pattern[piece.color.index()][pattern_of(piece)] ^= bit;
    }

    pub(crate) fn occupied(&self) -> Bitboard {
        self.occupied
    }

    pub(crate) fn pieces_color(&self, color: Color) -> Bitboard {
        self.by_color[color.index()]
    }

    /// The `color` pieces whose attack pattern is `pattern` (a [`pattern_of`]
    /// slot).
    pub(crate) fn pieces_pattern(&self, color: Color, pattern: usize) -> Bitboard {
        self.by_pattern[color.index()][pattern]
    }
}

impl Default for Board {
    fn default() -> Self {
        Self::empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::color::Color;
    use crate::piece::PieceKind;

    #[test]
    fn empty_has_no_pieces() {
        let b = Board::empty();
        for index in 0..Square::COUNT as u8 {
            let sq = Square::from_index(index).unwrap();
            assert!(b.get(sq).is_none());
        }
    }

    #[test]
    fn set_then_get_round_trip_at_corner() {
        let mut b = Board::empty();
        let sq = Square::new(8, 8).unwrap();
        let piece = Piece::new(PieceKind::Rook, Color::White);
        b.set(sq, Some(piece));
        assert_eq!(b.get(sq), Some(piece));
    }

    #[test]
    fn set_does_not_disturb_other_squares() {
        let mut b = Board::empty();
        let sq_a = Square::new(0, 0).unwrap();
        let sq_b = Square::new(8, 8).unwrap();
        let piece = Piece::new(PieceKind::King, Color::Black);
        b.set(sq_a, Some(piece));
        assert!(b.get(sq_b).is_none());
    }

    /// The from-scratch oracle the incremental maintenance in `set` must agree
    /// with.
    fn sets_from_scan(
        b: &Board,
    ) -> (
        [Bitboard; Color::COUNT],
        [[Bitboard; PATTERN_COUNT]; 2],
        Bitboard,
    ) {
        let mut by_color = [Bitboard::EMPTY; Color::COUNT];
        let mut by_pattern = [[Bitboard::EMPTY; PATTERN_COUNT]; Color::COUNT];
        let mut occ = Bitboard::EMPTY;
        for index in 0..Square::COUNT as u8 {
            let sq = Square::from_index(index).unwrap();
            if let Some(p) = b.get(sq) {
                let bit = Bitboard::from_square(sq);
                occ |= bit;
                by_color[p.color.index()] |= bit;
                by_pattern[p.color.index()][pattern_of(p)] |= bit;
            }
        }
        (by_color, by_pattern, occ)
    }

    fn assert_sets_consistent(b: &Board) {
        let (by_color, by_pattern, occ) = sets_from_scan(b);
        assert_eq!(b.occupied(), occ, "occupied set drifted");
        for color in [Color::Black, Color::White] {
            assert_eq!(
                b.pieces_color(color),
                by_color[color.index()],
                "by_color {color:?}"
            );
            for (pat, &expected) in by_pattern[color.index()].iter().enumerate() {
                assert_eq!(
                    b.pieces_pattern(color, pat),
                    expected,
                    "by_pattern {color:?} {pat}",
                );
            }
        }
    }

    #[test]
    fn piece_sets_track_place_capture_and_clear() {
        let mut b = Board::empty();
        assert_sets_consistent(&b);
        let sq = Square::new(4, 4).unwrap();
        let horse = Piece::promoted(PieceKind::Bishop, Color::White).unwrap();
        b.set(sq, Some(Piece::new(PieceKind::Gold, Color::Black)));
        assert_sets_consistent(&b);
        b.set(sq, Some(horse));
        assert_sets_consistent(&b);
        assert!(b.pieces_pattern(Color::White, pattern_of(horse)).test(sq));
        assert!(!b.pieces_color(Color::Black).test(sq));
        b.set(sq, None);
        assert_sets_consistent(&b);
        assert!(b.occupied().is_empty());
    }
}
