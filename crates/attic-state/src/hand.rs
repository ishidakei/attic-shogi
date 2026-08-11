use crate::piece::PieceKind;

const HAND_KINDS: usize = 7;

const MAX_BY_KIND: [u8; HAND_KINDS] = [18, 4, 4, 4, 4, 2, 2];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Hand([u8; HAND_KINDS]);

impl Hand {
    pub const fn empty() -> Self {
        Self([0; HAND_KINDS])
    }

    pub const fn count(&self, kind: PieceKind) -> u8 {
        match Self::slot(kind) {
            Some(idx) => self.0[idx],
            None => 0,
        }
    }

    pub fn increment(&mut self, kind: PieceKind) {
        if let Some(idx) = Self::slot(kind) {
            let cap = MAX_BY_KIND[idx];
            if self.0[idx] < cap {
                self.0[idx] += 1;
            }
        }
    }

    pub fn decrement(&mut self, kind: PieceKind) {
        if let Some(idx) = Self::slot(kind)
            && self.0[idx] > 0
        {
            self.0[idx] -= 1;
        }
    }

    const fn slot(kind: PieceKind) -> Option<usize> {
        match kind {
            PieceKind::Pawn => Some(0),
            PieceKind::Lance => Some(1),
            PieceKind::Knight => Some(2),
            PieceKind::Silver => Some(3),
            PieceKind::Gold => Some(4),
            PieceKind::Bishop => Some(5),
            PieceKind::Rook => Some(6),
            PieceKind::King => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_has_zero_counts() {
        let h = Hand::empty();
        for kind in [
            PieceKind::Pawn,
            PieceKind::Lance,
            PieceKind::Knight,
            PieceKind::Silver,
            PieceKind::Gold,
            PieceKind::Bishop,
            PieceKind::Rook,
        ] {
            assert_eq!(h.count(kind), 0);
        }
    }

    #[test]
    fn increment_clamps_at_pawn_cap() {
        let mut h = Hand::empty();
        for _ in 0..30 {
            h.increment(PieceKind::Pawn);
        }
        assert_eq!(h.count(PieceKind::Pawn), 18);
    }

    #[test]
    fn increment_clamps_at_rook_cap() {
        let mut h = Hand::empty();
        for _ in 0..10 {
            h.increment(PieceKind::Rook);
        }
        assert_eq!(h.count(PieceKind::Rook), 2);
    }

    #[test]
    fn decrement_saturates_at_zero() {
        let mut h = Hand::empty();
        h.decrement(PieceKind::Lance);
        assert_eq!(h.count(PieceKind::Lance), 0);
    }

    #[test]
    fn king_is_not_held_in_hand() {
        let mut h = Hand::empty();
        h.increment(PieceKind::King);
        assert_eq!(h.count(PieceKind::King), 0);
    }
}
