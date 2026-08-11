use attic_state::{
    Color, Hand, Piece, PieceKind, Position, STARTPOS_SFEN, Square, format_sfen, parse_sfen,
};
use proptest::prelude::*;

const ALL_KINDS: [PieceKind; 8] = [
    PieceKind::Pawn,
    PieceKind::Lance,
    PieceKind::Knight,
    PieceKind::Silver,
    PieceKind::Gold,
    PieceKind::Bishop,
    PieceKind::Rook,
    PieceKind::King,
];

const HAND_KINDS: [PieceKind; 7] = [
    PieceKind::Pawn,
    PieceKind::Lance,
    PieceKind::Knight,
    PieceKind::Silver,
    PieceKind::Gold,
    PieceKind::Bishop,
    PieceKind::Rook,
];

const HAND_CAPS: [u8; 7] = [18, 4, 4, 4, 4, 2, 2];

fn arb_color() -> impl Strategy<Value = Color> {
    prop_oneof![Just(Color::Black), Just(Color::White)]
}

fn arb_kind() -> impl Strategy<Value = PieceKind> {
    (0usize..ALL_KINDS.len()).prop_map(|i| ALL_KINDS[i])
}

fn arb_piece() -> impl Strategy<Value = Piece> {
    (arb_kind(), arb_color(), any::<bool>()).prop_map(|(kind, color, want_promote)| {
        if want_promote && kind.can_promote() {
            Piece::promoted(kind, color).unwrap()
        } else {
            Piece::new(kind, color)
        }
    })
}

fn arb_square_contents() -> impl Strategy<Value = Option<Piece>> {
    prop_oneof![5 => Just(None), 1 => arb_piece().prop_map(Some)]
}

fn arb_hand() -> impl Strategy<Value = Hand> {
    (
        0u8..=HAND_CAPS[0],
        0u8..=HAND_CAPS[1],
        0u8..=HAND_CAPS[2],
        0u8..=HAND_CAPS[3],
        0u8..=HAND_CAPS[4],
        0u8..=HAND_CAPS[5],
        0u8..=HAND_CAPS[6],
    )
        .prop_map(|cs| {
            let counts = [cs.0, cs.1, cs.2, cs.3, cs.4, cs.5, cs.6];
            let mut h = Hand::empty();
            for (idx, n) in counts.iter().enumerate() {
                for _ in 0..*n {
                    h.increment(HAND_KINDS[idx]);
                }
            }
            h
        })
}

fn arb_position() -> impl Strategy<Value = Position> {
    (
        proptest::collection::vec(arb_square_contents(), Square::COUNT),
        arb_hand(),
        arb_hand(),
        arb_color(),
        1u16..=u16::MAX,
    )
        .prop_map(|(squares, black_hand, white_hand, stm, ply)| {
            let mut p = Position::empty();
            for (i, contents) in squares.into_iter().enumerate() {
                let sq = Square::from_index(i as u8).unwrap();
                p.board_mut().set(sq, contents);
            }
            *p.hand_mut(Color::Black) = black_hand;
            *p.hand_mut(Color::White) = white_hand;
            p.set_side_to_move(stm);
            p.set_ply(ply);
            p
        })
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 256,
        ..ProptestConfig::default()
    })]

    #[cfg_attr(miri, ignore)]
    #[test]
    fn parse_format_round_trip(p in arb_position()) {
        let s = format_sfen(&p);
        let parsed = parse_sfen(&s).expect("format_sfen must produce a parseable sfen");
        prop_assert_eq!(parsed, p);
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn format_is_canonical(p in arb_position()) {
        let s1 = format_sfen(&p);
        let p2 = parse_sfen(&s1).unwrap();
        let s2 = format_sfen(&p2);
        prop_assert_eq!(s1, s2);
    }
}

#[test]
fn startpos_constant_round_trips_through_helper() {
    let p = Position::startpos();
    assert_eq!(format_sfen(&p), STARTPOS_SFEN);
}
