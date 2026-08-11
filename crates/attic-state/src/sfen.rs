use core::fmt;

use crate::color::Color;
use crate::piece::{Piece, PieceKind};
use crate::position::Position;
use crate::square::Square;

pub const STARTPOS_SFEN: &str = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SfenError {
    UnexpectedEnd,
    InvalidPiece(char),
    UnexpectedPromoteMarker,
    NonPromotablePromoted(char),
    BoardCursorOverflow,
    BoardCursorIncomplete,
    InvalidSideToMove(char),
    MissingHandPieceAfterCount,
    InvalidHandPiece(char),
    InvalidPly,
    UnexpectedTrailing,
}

impl fmt::Display for SfenError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SfenError::UnexpectedEnd => f.write_str("sfen: unexpected end of input"),
            SfenError::InvalidPiece(c) => write!(f, "sfen: invalid piece char {c:?}"),
            SfenError::UnexpectedPromoteMarker => f.write_str("sfen: unexpected '+' marker"),
            SfenError::NonPromotablePromoted(c) => {
                write!(f, "sfen: piece {c:?} cannot be promoted")
            }
            SfenError::BoardCursorOverflow => {
                f.write_str("sfen: board cursor advanced past file 1")
            }
            SfenError::BoardCursorIncomplete => {
                f.write_str("sfen: board section ended with cursor not at end")
            }
            SfenError::InvalidSideToMove(c) => {
                write!(f, "sfen: invalid side-to-move {c:?}")
            }
            SfenError::MissingHandPieceAfterCount => {
                f.write_str("sfen: hand count without following piece")
            }
            SfenError::InvalidHandPiece(c) => write!(f, "sfen: invalid hand piece {c:?}"),
            SfenError::InvalidPly => f.write_str("sfen: invalid ply"),
            SfenError::UnexpectedTrailing => f.write_str("sfen: unexpected trailing input"),
        }
    }
}

impl std::error::Error for SfenError {}

pub fn parse_sfen(s: &str) -> Result<Position, SfenError> {
    let mut fields = s.split(' ');
    let board_field = fields.next().ok_or(SfenError::UnexpectedEnd)?;
    let stm_field = fields.next().ok_or(SfenError::UnexpectedEnd)?;
    let hand_field = fields.next().ok_or(SfenError::UnexpectedEnd)?;
    let ply_field = fields.next().ok_or(SfenError::UnexpectedEnd)?;
    if fields.next().is_some() {
        return Err(SfenError::UnexpectedTrailing);
    }

    let mut pos = Position::empty();
    parse_board(board_field, &mut pos)?;
    parse_side_to_move(stm_field, &mut pos)?;
    parse_hands(hand_field, &mut pos)?;
    parse_ply(ply_field, &mut pos)?;
    // The board / hand / side mutations above go through the direct setters,
    // which bypass incremental key maintenance; seed the keys once here.
    pos.refresh_keys();
    Ok(pos)
}

fn parse_board(field: &str, pos: &mut Position) -> Result<(), SfenError> {
    let mut file: i8 = (Square::FILES as i8) - 1;
    let mut rank: i8 = 0;
    let mut promote = false;

    for ch in field.chars() {
        match ch {
            '/' => {
                if file != -1 {
                    return Err(SfenError::BoardCursorIncomplete);
                }
                if rank == (Square::RANKS as i8) - 1 {
                    return Err(SfenError::BoardCursorOverflow);
                }
                if promote {
                    return Err(SfenError::UnexpectedPromoteMarker);
                }
                file = (Square::FILES as i8) - 1;
                rank += 1;
            }
            '+' => {
                if promote {
                    return Err(SfenError::UnexpectedPromoteMarker);
                }
                promote = true;
            }
            '1'..='9' => {
                if promote {
                    return Err(SfenError::UnexpectedPromoteMarker);
                }
                let skip = (ch as u8 - b'0') as i8;
                if file - skip < -1 {
                    return Err(SfenError::BoardCursorOverflow);
                }
                file -= skip;
            }
            _ => {
                let (kind, color) = parse_piece_char(ch)?;
                if file < 0 || rank < 0 || rank >= Square::RANKS as i8 {
                    return Err(SfenError::BoardCursorOverflow);
                }
                let piece = if promote {
                    Piece::promoted(kind, color)
                        .ok_or(SfenError::NonPromotablePromoted(ch.to_ascii_uppercase()))?
                } else {
                    Piece::new(kind, color)
                };
                let sq = Square::new(file as u8, rank as u8).unwrap();
                pos.board_mut().set(sq, Some(piece));
                file -= 1;
                promote = false;
            }
        }
    }

    if promote {
        return Err(SfenError::UnexpectedPromoteMarker);
    }
    if rank != (Square::RANKS as i8) - 1 || file != -1 {
        return Err(SfenError::BoardCursorIncomplete);
    }
    Ok(())
}

fn parse_side_to_move(field: &str, pos: &mut Position) -> Result<(), SfenError> {
    let mut chars = field.chars();
    let token = chars.next().ok_or(SfenError::UnexpectedEnd)?;
    if chars.next().is_some() {
        return Err(SfenError::InvalidSideToMove(token));
    }
    let color = match token {
        'b' => Color::Black,
        'w' => Color::White,
        c => return Err(SfenError::InvalidSideToMove(c)),
    };
    pos.set_side_to_move(color);
    Ok(())
}

fn parse_hands(field: &str, pos: &mut Position) -> Result<(), SfenError> {
    if field.is_empty() {
        return Err(SfenError::UnexpectedEnd);
    }
    if field == "-" {
        return Ok(());
    }

    let mut count: u32 = 0;
    let mut count_started = false;
    for ch in field.chars() {
        if ch.is_ascii_digit() {
            count = count * 10 + (ch as u32 - '0' as u32);
            count_started = true;
        } else {
            let (kind, color) = parse_piece_char(ch)?;
            if !is_hand_kind(kind) {
                return Err(SfenError::InvalidHandPiece(ch));
            }
            let n = if count_started { count } else { 1 };
            for _ in 0..n {
                pos.hand_mut(color).increment(kind);
            }
            count = 0;
            count_started = false;
        }
    }

    if count_started {
        return Err(SfenError::MissingHandPieceAfterCount);
    }
    Ok(())
}

fn parse_ply(field: &str, pos: &mut Position) -> Result<(), SfenError> {
    if field.is_empty() {
        return Err(SfenError::InvalidPly);
    }
    let n: u16 = field.parse().map_err(|_| SfenError::InvalidPly)?;
    pos.set_ply(n);
    Ok(())
}

fn parse_piece_char(ch: char) -> Result<(PieceKind, Color), SfenError> {
    let color = if ch.is_ascii_uppercase() {
        Color::Black
    } else if ch.is_ascii_lowercase() {
        Color::White
    } else {
        return Err(SfenError::InvalidPiece(ch));
    };
    let kind = match ch.to_ascii_uppercase() {
        'P' => PieceKind::Pawn,
        'L' => PieceKind::Lance,
        'N' => PieceKind::Knight,
        'S' => PieceKind::Silver,
        'G' => PieceKind::Gold,
        'B' => PieceKind::Bishop,
        'R' => PieceKind::Rook,
        'K' => PieceKind::King,
        _ => return Err(SfenError::InvalidPiece(ch)),
    };
    Ok((kind, color))
}

const fn is_hand_kind(kind: PieceKind) -> bool {
    !matches!(kind, PieceKind::King)
}

const HAND_OUTPUT_ORDER: [PieceKind; 7] = [
    PieceKind::Rook,
    PieceKind::Bishop,
    PieceKind::Gold,
    PieceKind::Silver,
    PieceKind::Knight,
    PieceKind::Lance,
    PieceKind::Pawn,
];

pub fn format_sfen(pos: &Position) -> String {
    let mut out = String::new();
    format_board(pos, &mut out);
    out.push(' ');
    out.push(match pos.side_to_move() {
        Color::Black => 'b',
        Color::White => 'w',
    });
    out.push(' ');
    format_hands(pos, &mut out);
    out.push(' ');
    out.push_str(&pos.ply().to_string());
    out
}

fn format_board(pos: &Position, out: &mut String) {
    for rank in 0..Square::RANKS {
        if rank != 0 {
            out.push('/');
        }
        let mut empty: u8 = 0;
        for file in (0..Square::FILES).rev() {
            let sq = Square::new(file, rank).unwrap();
            match pos.board().get(sq) {
                None => empty += 1,
                Some(piece) => {
                    if empty > 0 {
                        out.push((b'0' + empty) as char);
                        empty = 0;
                    }
                    if piece.promoted {
                        out.push('+');
                    }
                    out.push(piece_char(piece.kind, piece.color));
                }
            }
        }
        if empty > 0 {
            out.push((b'0' + empty) as char);
        }
    }
}

fn format_hands(pos: &Position, out: &mut String) {
    let mut wrote = false;
    for color in [Color::Black, Color::White] {
        for &kind in &HAND_OUTPUT_ORDER {
            let n = pos.hand(color).count(kind);
            if n == 0 {
                continue;
            }
            wrote = true;
            if n != 1 {
                out.push_str(&n.to_string());
            }
            out.push(piece_char(kind, color));
        }
    }
    if !wrote {
        out.push('-');
    }
}

fn piece_char(kind: PieceKind, color: Color) -> char {
    let upper = match kind {
        PieceKind::Pawn => 'P',
        PieceKind::Lance => 'L',
        PieceKind::Knight => 'N',
        PieceKind::Silver => 'S',
        PieceKind::Gold => 'G',
        PieceKind::Bishop => 'B',
        PieceKind::Rook => 'R',
        PieceKind::King => 'K',
    };
    match color {
        Color::Black => upper,
        Color::White => upper.to_ascii_lowercase(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn startpos_round_trips_byte_for_byte() {
        let pos = parse_sfen(STARTPOS_SFEN).unwrap();
        assert_eq!(format_sfen(&pos), STARTPOS_SFEN);
    }

    #[test]
    fn startpos_places_kings_at_5a_and_5e() {
        let pos = parse_sfen(STARTPOS_SFEN).unwrap();
        let black_king_sq = Square::new(4, 8).unwrap();
        let white_king_sq = Square::new(4, 0).unwrap();
        assert_eq!(
            pos.board().get(black_king_sq),
            Some(Piece::new(PieceKind::King, Color::Black))
        );
        assert_eq!(
            pos.board().get(white_king_sq),
            Some(Piece::new(PieceKind::King, Color::White))
        );
    }

    #[test]
    fn side_to_move_b_and_w_parse() {
        let mut s = STARTPOS_SFEN.to_string();
        s = s.replace(" b ", " w ");
        let pos = parse_sfen(&s).unwrap();
        assert_eq!(pos.side_to_move(), Color::White);
        let pos2 = parse_sfen(STARTPOS_SFEN).unwrap();
        assert_eq!(pos2.side_to_move(), Color::Black);
    }

    #[test]
    fn empty_run_digits_place_piece_at_correct_file() {
        let sfen = "9/9/9/9/4P4/9/9/9/9 b - 1";
        let pos = parse_sfen(sfen).unwrap();
        let sq = Square::new(4, 4).unwrap();
        assert_eq!(
            pos.board().get(sq),
            Some(Piece::new(PieceKind::Pawn, Color::Black))
        );
        assert_eq!(format_sfen(&pos), sfen);
    }

    #[test]
    fn promoted_piece_round_trips() {
        let sfen = "9/9/9/9/4+P4/9/9/9/9 b - 1";
        let pos = parse_sfen(sfen).unwrap();
        let sq = Square::new(4, 4).unwrap();
        let piece = pos.board().get(sq).unwrap();
        assert!(piece.promoted);
        assert_eq!(piece.kind, PieceKind::Pawn);
        assert_eq!(piece.color, Color::Black);
        assert_eq!(format_sfen(&pos), sfen);
    }

    #[test]
    fn hand_with_mixed_counts_round_trips() {
        let sfen = "9/9/9/9/9/9/9/9/9 b P2p 1";
        let pos = parse_sfen(sfen).unwrap();
        assert_eq!(pos.hand(Color::Black).count(PieceKind::Pawn), 1);
        assert_eq!(pos.hand(Color::White).count(PieceKind::Pawn), 2);
        assert_eq!(format_sfen(&pos), sfen);
    }

    #[test]
    fn hand_dash_round_trips() {
        let sfen = "9/9/9/9/9/9/9/9/9 b - 1";
        let pos = parse_sfen(sfen).unwrap();
        assert_eq!(format_sfen(&pos), sfen);
    }

    #[test]
    fn hand_multi_digit_count_round_trips() {
        let sfen = "9/9/9/9/9/9/9/9/9 b 18P 1";
        let pos = parse_sfen(sfen).unwrap();
        assert_eq!(pos.hand(Color::Black).count(PieceKind::Pawn), 18);
        assert_eq!(format_sfen(&pos), sfen);
    }

    #[test]
    fn hand_full_canonical_order() {
        let sfen = "9/9/9/9/9/9/9/9/9 b RBGSNLPrbgsnlp 1";
        let pos = parse_sfen(sfen).unwrap();
        assert_eq!(format_sfen(&pos), sfen);
    }

    #[test]
    fn rejects_promoted_gold() {
        let sfen = "9/9/9/9/4+G4/9/9/9/9 b - 1";
        match parse_sfen(sfen) {
            Err(SfenError::NonPromotablePromoted('G')) => {}
            other => panic!("expected NonPromotablePromoted('G'), got {other:?}"),
        }
    }

    #[test]
    fn rejects_promote_marker_before_digit() {
        let sfen = "9/9/9/9/+1P6/9/9/9/9 b - 1";
        match parse_sfen(sfen) {
            Err(SfenError::UnexpectedPromoteMarker) => {}
            other => panic!("expected UnexpectedPromoteMarker, got {other:?}"),
        }
    }

    #[test]
    fn rejects_invalid_side_to_move() {
        let sfen = "9/9/9/9/9/9/9/9/9 x - 1";
        match parse_sfen(sfen) {
            Err(SfenError::InvalidSideToMove('x')) => {}
            other => panic!("expected InvalidSideToMove('x'), got {other:?}"),
        }
    }

    #[test]
    fn rejects_king_in_hand() {
        let sfen = "9/9/9/9/9/9/9/9/9 b K 1";
        match parse_sfen(sfen) {
            Err(SfenError::InvalidHandPiece('K')) => {}
            other => panic!("expected InvalidHandPiece('K'), got {other:?}"),
        }
    }

    #[test]
    fn rejects_count_without_piece() {
        let sfen = "9/9/9/9/9/9/9/9/9 b 5 1";
        match parse_sfen(sfen) {
            Err(SfenError::MissingHandPieceAfterCount) => {}
            other => panic!("expected MissingHandPieceAfterCount, got {other:?}"),
        }
    }

    #[test]
    fn rejects_overflowing_empty_run() {
        let sfen = "9/9/9/9/8P1/9/9/9/9 b - 1";
        match parse_sfen(sfen) {
            Err(SfenError::BoardCursorOverflow) => {}
            other => panic!("expected BoardCursorOverflow, got {other:?}"),
        }
    }

    #[test]
    fn rejects_short_rank() {
        let sfen = "8/9/9/9/9/9/9/9/9 b - 1";
        match parse_sfen(sfen) {
            Err(SfenError::BoardCursorIncomplete) => {}
            other => panic!("expected BoardCursorIncomplete, got {other:?}"),
        }
    }

    #[test]
    fn rejects_trailing_field() {
        let sfen = "9/9/9/9/9/9/9/9/9 b - 1 extra";
        match parse_sfen(sfen) {
            Err(SfenError::UnexpectedTrailing) => {}
            other => panic!("expected UnexpectedTrailing, got {other:?}"),
        }
    }

    #[test]
    fn startpos_helper_matches_parse() {
        let from_helper = Position::startpos();
        let from_parse = parse_sfen(STARTPOS_SFEN).unwrap();
        assert_eq!(from_helper, from_parse);
    }
}
