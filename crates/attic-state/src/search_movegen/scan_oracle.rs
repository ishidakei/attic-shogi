//! `#[cfg(test)]` scanning oracles for the piece-set move generators — the
//! emission-sequence gate.
//!
//! Each of the four search generators has a scan twin here that derives the
//! emission independently of the piece sets and the bitboard attack tables: the
//! generating pieces come from an 81-square scan (via
//! [`super::emit_group_scan`], which filters with `Group::contains` and computes
//! destinations with the movement-walk [`super::reachable_scan`]), and the
//! evasion king moves come from [`super::reachable_scan`] too. Only the
//! piece-move machinery is re-derived; the drop tail reuses the production
//! [`Position::emit_drops`], whose one bitboard-derived input (the file-mask
//! nifu mask) is pinned separately by
//! `nifu_blocked_files == nifu_blocked_files_scan`. So a twin's full output is
//! that of an entirely scan-based generator, and the sequence-equality gate
//! compares it move-for-move against production.

use super::{ExtMove, Group, Target, emit_group_scan, push_plain, reachable_scan};
use crate::movegen::try_find_king;
use crate::position::Position;

/// The `CAPTURES` piece groups (gold-group carries the king), shared by the
/// scan twins that emit every group.
const CAPTURE_GROUPS: [Group; 6] = [
    Group::Pawn,
    Group::Lance,
    Group::Knight,
    Group::Silver,
    Group::BishopRook,
    Group::GoldHdk { king: true },
];

impl Position {
    /// Scan twin of [`Position::generate_captures`].
    pub(super) fn generate_captures_scan(&self, all: bool, out: &mut Vec<ExtMove>) {
        let board = self.board();
        let stm = self.side_to_move();
        for group in CAPTURE_GROUPS {
            emit_group_scan(board, stm, group, Target::Captures, all, out);
        }
    }

    /// Scan twin of [`Position::generate_quiets`].
    pub(super) fn generate_quiets_scan(&self, all: bool, out: &mut Vec<ExtMove>) {
        let board = self.board();
        let stm = self.side_to_move();
        for group in CAPTURE_GROUPS {
            emit_group_scan(board, stm, group, Target::Quiets, all, out);
        }
        self.emit_drops(out);
    }

    /// Scan twin of [`Position::generate_evasions`].
    pub(super) fn generate_evasions_scan(&self, all: bool, out: &mut Vec<ExtMove>) {
        let board = self.board();
        let stm = self.side_to_move();

        if let Some(ksq) = try_find_king(board, stm) {
            let king = board.get(ksq).unwrap();
            for to in reachable_scan(board, ksq, king, Target::BlockOrCapture) {
                push_plain(ksq, to, king, out);
            }
        }

        // Same order as CAPTURES but the gold-group excludes the king (already
        // emitted), and destinations include empty (blocking) squares.
        const GROUPS: [Group; 6] = [
            Group::Pawn,
            Group::Lance,
            Group::Knight,
            Group::Silver,
            Group::BishopRook,
            Group::GoldHdk { king: false },
        ];
        for group in GROUPS {
            emit_group_scan(board, stm, group, Target::BlockOrCapture, all, out);
        }

        self.emit_drops(out);
    }

    /// Scan twin of [`Position::generate_non_evasions`].
    pub(super) fn generate_non_evasions_scan(&self, all: bool, out: &mut Vec<ExtMove>) {
        let board = self.board();
        let stm = self.side_to_move();
        for group in CAPTURE_GROUPS {
            emit_group_scan(board, stm, group, Target::BlockOrCapture, all, out);
        }
        self.emit_drops(out);
    }
}
