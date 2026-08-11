use crate::color::Color;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum PieceKind {
    Pawn = 0,
    Lance = 1,
    Knight = 2,
    Silver = 3,
    Gold = 4,
    Bishop = 5,
    Rook = 6,
    King = 7,
}

impl PieceKind {
    pub const COUNT: usize = 8;

    pub const fn index(self) -> usize {
        self as usize
    }

    pub const fn can_promote(self) -> bool {
        matches!(
            self,
            PieceKind::Pawn
                | PieceKind::Lance
                | PieceKind::Knight
                | PieceKind::Silver
                | PieceKind::Bishop
                | PieceKind::Rook
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Piece {
    pub kind: PieceKind,
    pub color: Color,
    pub promoted: bool,
}

impl Piece {
    pub const fn new(kind: PieceKind, color: Color) -> Self {
        Self {
            kind,
            color,
            promoted: false,
        }
    }

    pub const fn promoted(kind: PieceKind, color: Color) -> Option<Self> {
        if kind.can_promote() {
            Some(Self {
                kind,
                color,
                promoted: true,
            })
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_is_unpromoted() {
        let p = Piece::new(PieceKind::Pawn, Color::Black);
        assert_eq!(p.kind, PieceKind::Pawn);
        assert_eq!(p.color, Color::Black);
        assert!(!p.promoted);
    }

    #[test]
    fn promoted_rejects_gold_and_king() {
        assert!(Piece::promoted(PieceKind::Gold, Color::Black).is_none());
        assert!(Piece::promoted(PieceKind::King, Color::White).is_none());
    }

    #[test]
    fn promoted_accepts_promotable_kinds() {
        for kind in [
            PieceKind::Pawn,
            PieceKind::Lance,
            PieceKind::Knight,
            PieceKind::Silver,
            PieceKind::Bishop,
            PieceKind::Rook,
        ] {
            let p = Piece::promoted(kind, Color::Black).unwrap();
            assert!(p.promoted);
            assert_eq!(p.kind, kind);
        }
    }
}
