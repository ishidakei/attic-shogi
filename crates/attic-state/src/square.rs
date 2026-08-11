#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Square(u8);

impl Square {
    pub const FILES: u8 = 9;
    pub const RANKS: u8 = 9;
    pub const COUNT: usize = 81;

    pub const fn new(file: u8, rank: u8) -> Option<Self> {
        if file < Self::FILES && rank < Self::RANKS {
            Some(Self(file * Self::RANKS + rank))
        } else {
            None
        }
    }

    pub const fn from_index(index: u8) -> Option<Self> {
        if (index as usize) < Self::COUNT {
            Some(Self(index))
        } else {
            None
        }
    }

    pub const fn index(self) -> u8 {
        self.0
    }

    pub const fn file(self) -> u8 {
        self.0 / Self::RANKS
    }

    pub const fn rank(self) -> u8 {
        self.0 % Self::RANKS
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn corner_round_trip() {
        let sq = Square::new(8, 8).unwrap();
        assert_eq!(sq.file(), 8);
        assert_eq!(sq.rank(), 8);
        assert_eq!(sq.index(), 80);
    }

    #[test]
    fn origin_round_trip() {
        let sq = Square::new(0, 0).unwrap();
        assert_eq!(sq.file(), 0);
        assert_eq!(sq.rank(), 0);
        assert_eq!(sq.index(), 0);
    }

    #[test]
    fn out_of_range_is_none() {
        assert!(Square::new(9, 0).is_none());
        assert!(Square::new(0, 9).is_none());
        assert!(Square::new(9, 9).is_none());
    }

    #[test]
    fn from_index_out_of_range_is_none() {
        assert!(Square::from_index(81).is_none());
        assert!(Square::from_index(255).is_none());
    }

    #[test]
    fn all_indices_round_trip() {
        for file in 0..Square::FILES {
            for rank in 0..Square::RANKS {
                let sq = Square::new(file, rank).unwrap();
                assert_eq!(sq.file(), file);
                assert_eq!(sq.rank(), rank);
                assert_eq!(Square::from_index(sq.index()).unwrap(), sq);
            }
        }
    }
}
