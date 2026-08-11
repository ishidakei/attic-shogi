//! `.ybb` opening-book reader — the YaneuraOu "YANE-BINBOOK-V1" binary book.
//!
//! The format is transcribed from the pinned reference
//! (`source/book/book.cpp`). A `.ybb`
//! file is a header, then a fixed-stride index region (one record per stored
//! position, sorted ascending by the 32-byte packed key), then a variable moves
//! region. A position is looked up by binary search over the index.
//!
//! Two read modes are provided and return identical results:
//! - [`Book::open_in_memory`] slurps the whole file (BookOnTheFly=false).
//! - [`Book::open_on_the_fly`] keeps file handles and seeks per lookup
//!   (BookOnTheFly=true).
//!
//! # Layering
//!
//! The Storage layer speaks only in primitives: this
//! reader takes the packed key as a raw `&[u8; 32]` and the game ply as a
//! `u16`, and returns raw move fragments ([`BookMove`]). Computing the packed
//! key from a `Position` (the PackedSfen encoder) and widening a `move16` into a
//! validated move both live above this layer.
//!
//! # Totality
//!
//! Malformed input never panics and never reads out of bounds. A corrupt or
//! truncated *header* is reported as an [`BookError`] at open time; a truncated
//! or out-of-range *index/moves* region degrades to a graceful miss
//! (`Ok(None)`) at probe time, mirroring the reference (a failed stream read
//! there yields "no book move").

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

/// 16-byte magic identifying a `.ybb` file (`YbbMagic`, `book.cpp`).
const MAGIC: &[u8; 16] = b"YANE-BINBOOK-V1\0";
/// Header size in bytes (`YbbHeaderSize`, `book.cpp`).
const HEADER_SIZE: u64 = 32;
/// Index record stride in bytes (`YbbIndexRecordSize`, `book.cpp`).
const INDEX_RECORD_SIZE: u64 = 44;
/// `flags` bit 0 — when set, each move record carries a trailing `depth` u16
/// (`YbbFlagMoveDepth`, `book.cpp`).
const FLAG_MOVE_DEPTH: u64 = 1;
/// The set of flag bits this reader understands (`YbbKnownFlags`,
/// `book.cpp`). Any other bit set means the file is not one we can read.
const KNOWN_FLAGS: u64 = FLAG_MOVE_DEPTH;

/// One decoded move from a book position record (`read_ybb_moves`,
/// `book.cpp`).
///
/// The reference's `BookMove` also carries a ponder move and an adoption count;
/// a `.ybb` stores neither, so ponder is always none (omitted here) and `count`
/// is always 0. `value` is the stored eval reinterpreted as a signed 16-bit
/// integer; `depth` is 0 when the file's move-depth flag is clear.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BookMove {
    /// 16-bit move fragment (YaneuraOu `Move16`).
    pub move16: u16,
    /// Stored eval, from the reference side-to-move's perspective.
    pub value: i16,
    /// Search depth at which the move was recorded (0 if the file omits depth).
    pub depth: u16,
    /// Adoption count — always 0 for a `.ybb` (the format stores no per-move count).
    pub count: u16,
}

/// Failure opening or validating a `.ybb` file.
#[derive(Debug)]
pub enum BookError {
    /// The file is shorter than the 32-byte header.
    TruncatedHeader,
    /// The leading 16 bytes are not the `.ybb` magic.
    BadMagic,
    /// The header's `flags` word has bits set that this reader does not know.
    UnknownFlags(u64),
    /// `header + record_count * 44` overflows `u64` — a corrupt record count.
    RecordCountOverflow,
    /// In-memory mode: the file is too short to hold the full index region the
    /// header declares.
    TruncatedIndex,
    /// An I/O error other than a clean end-of-file.
    Io(std::io::Error),
}

impl std::fmt::Display for BookError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BookError::TruncatedHeader => f.write_str("ybb: file shorter than the 32-byte header"),
            BookError::BadMagic => f.write_str("ybb: bad magic (not a YANE-BINBOOK-V1 file)"),
            BookError::UnknownFlags(flags) => {
                write!(f, "ybb: header flags {flags:#x} carry unknown bits")
            }
            BookError::RecordCountOverflow => {
                f.write_str("ybb: record count overflows the index region size")
            }
            BookError::TruncatedIndex => {
                f.write_str("ybb: file too short for the declared index region")
            }
            BookError::Io(e) => write!(f, "ybb: i/o error: {e}"),
        }
    }
}

impl std::error::Error for BookError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            BookError::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for BookError {
    fn from(e: std::io::Error) -> Self {
        BookError::Io(e)
    }
}

/// A decoded index record (`YbbIndexEntry`, `book.cpp`).
struct IndexEntry {
    packed: [u8; 32],
    moves_offset: u64,
    ply: u16,
    move_count: u16,
}

/// Backing store for a [`Book`] — the whole file in memory, or handles to seek.
enum Source {
    Memory(Vec<u8>),
    /// Two handles onto the same file: one for the index region, one for the
    /// moves region, so index seeks and move seeks never fight over one
    /// cursor (matches the reference's twin `fstream`s).
    File {
        index: File,
        moves: File,
    },
}

/// An opened `.ybb` opening book.
pub struct Book {
    record_count: u64,
    flags: u64,
    /// Absolute offset of the moves region (`= 32 + record_count * 44`).
    moves_base: u64,
    source: Source,
}

impl Book {
    /// Open a `.ybb` and slurp it entirely into memory (BookOnTheFly=false).
    ///
    /// Validates the header and that the file is long enough to hold the whole
    /// index region.
    pub fn open_in_memory(path: impl AsRef<Path>) -> Result<Self, BookError> {
        let data = std::fs::read(path)?;
        Self::from_memory(data)
    }

    /// Build an in-memory book from an already-loaded byte buffer. Same
    /// validation as [`open_in_memory`](Self::open_in_memory); useful for tests.
    pub fn from_memory(data: Vec<u8>) -> Result<Self, BookError> {
        let head = data
            .get(..HEADER_SIZE as usize)
            .ok_or(BookError::TruncatedHeader)?;
        let (record_count, flags) = parse_header(head)?;
        let moves_base = moves_base(record_count)?;
        if (data.len() as u64) < moves_base {
            return Err(BookError::TruncatedIndex);
        }
        Ok(Book {
            record_count,
            flags,
            moves_base,
            source: Source::Memory(data),
        })
    }

    /// Open a `.ybb` keeping two file handles and seeking per lookup
    /// (BookOnTheFly=true). Validates the header only; a short index/moves
    /// region degrades to a miss at probe time.
    pub fn open_on_the_fly(path: impl AsRef<Path>) -> Result<Self, BookError> {
        let path = path.as_ref();
        let mut index = File::open(path)?;
        let moves = File::open(path)?;

        let mut head = [0u8; HEADER_SIZE as usize];
        match index.read_exact(&mut head) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                return Err(BookError::TruncatedHeader);
            }
            Err(e) => return Err(BookError::Io(e)),
        }
        let (record_count, flags) = parse_header(&head)?;
        let moves_base = moves_base(record_count)?;
        Ok(Book {
            record_count,
            flags,
            moves_base,
            source: Source::File { index, moves },
        })
    }

    /// Number of position records the header declares.
    pub fn record_count(&self) -> u64 {
        self.record_count
    }

    /// Whether the file's move records carry a per-move depth (`flags` bit 0).
    pub fn has_move_depth(&self) -> bool {
        self.flags & FLAG_MOVE_DEPTH != 0
    }

    /// Probe the book for `packed` at `game_ply`.
    ///
    /// Returns `Ok(Some(moves))` on a hit, `Ok(None)` on a miss (key absent, or
    /// — when `ignore_book_ply` is false — the stored ply differs, or the
    /// index/moves region is truncated). `Err` is reserved for genuine I/O
    /// failures, never for malformed content.
    ///
    /// Moves are returned in the file's stored order; the reference applies a
    /// presentation sort (`sort_moves()`) that is move-selection policy and out
    /// of scope for the raw reader.
    pub fn probe(
        &self,
        packed: &[u8; 32],
        game_ply: u16,
        ignore_book_ply: bool,
    ) -> Result<Option<Vec<BookMove>>, BookError> {
        let mut left = 0u64;
        let mut right = self.record_count;
        while left < right {
            let middle = left + (right - left) / 2;
            let entry = match self.read_index_entry(middle)? {
                Some(e) => e,
                None => return Ok(None),
            };
            match packed[..].cmp(&entry.packed[..]) {
                std::cmp::Ordering::Less => right = middle,
                std::cmp::Ordering::Greater => left = middle + 1,
                std::cmp::Ordering::Equal => {
                    if !ignore_book_ply && entry.ply != game_ply {
                        return Ok(None);
                    }
                    return self.read_moves(&entry);
                }
            }
        }
        Ok(None)
    }

    /// Per-move record size in bytes: 6 with the depth flag, else 4
    /// (`ybb_move_record_size`, `book.cpp`).
    fn move_record_size(&self) -> u64 {
        if self.has_move_depth() { 6 } else { 4 }
    }

    /// Read index record `index` (`index < record_count`). `Ok(None)` on a
    /// truncated/out-of-range region (graceful miss); `Err` only on I/O error.
    fn read_index_entry(&self, index: u64) -> Result<Option<IndexEntry>, BookError> {
        // `index < record_count`, and `moves_base = 32 + record_count*44` was
        // proven not to overflow, so this offset cannot overflow either.
        let offset = HEADER_SIZE + index * INDEX_RECORD_SIZE;
        match &self.source {
            Source::Memory(data) => {
                let end = offset + INDEX_RECORD_SIZE;
                match data.get(offset as usize..end as usize) {
                    Some(buf) => Ok(Some(parse_index_entry(buf))),
                    None => Ok(None),
                }
            }
            Source::File { index, .. } => {
                let mut buf = [0u8; INDEX_RECORD_SIZE as usize];
                match read_at(index, offset, &mut buf)? {
                    true => Ok(Some(parse_index_entry(&buf))),
                    false => Ok(None),
                }
            }
        }
    }

    /// Decode the move list for `entry`. `Ok(None)` on a truncated/out-of-range
    /// moves region (graceful miss); `Err` only on I/O error.
    fn read_moves(&self, entry: &IndexEntry) -> Result<Option<Vec<BookMove>>, BookError> {
        let record_size = self.move_record_size();
        let absolute = match self.moves_base.checked_add(entry.moves_offset) {
            Some(a) => a,
            None => return Ok(None),
        };
        let total = u64::from(entry.move_count) * record_size;

        match &self.source {
            Source::Memory(data) => {
                let start = absolute as usize;
                let end = match start.checked_add(total as usize) {
                    Some(e) => e,
                    None => return Ok(None),
                };
                match data.get(start..end) {
                    Some(region) => Ok(Some(decode_moves(
                        region,
                        entry.move_count,
                        record_size,
                        self.has_move_depth(),
                    ))),
                    None => Ok(None),
                }
            }
            Source::File { moves, .. } => {
                let mut buf = vec![0u8; total as usize];
                match read_at(moves, absolute, &mut buf)? {
                    true => Ok(Some(decode_moves(
                        &buf,
                        entry.move_count,
                        record_size,
                        self.has_move_depth(),
                    ))),
                    false => Ok(None),
                }
            }
        }
    }
}

/// Validate the 32-byte header (`head.len() >= 32`), returning
/// `(record_count, flags)`.
fn parse_header(head: &[u8]) -> Result<(u64, u64), BookError> {
    if &head[..16] != MAGIC {
        return Err(BookError::BadMagic);
    }
    let record_count = read_u64_le(&head[16..24]);
    let flags = read_u64_le(&head[24..32]);
    if flags & !KNOWN_FLAGS != 0 {
        return Err(BookError::UnknownFlags(flags));
    }
    Ok((record_count, flags))
}

/// `32 + record_count * 44`, with overflow reported as a corrupt record count
/// (`ybb_index_size`, `book.cpp`).
fn moves_base(record_count: u64) -> Result<u64, BookError> {
    record_count
        .checked_mul(INDEX_RECORD_SIZE)
        .and_then(|body| body.checked_add(HEADER_SIZE))
        .ok_or(BookError::RecordCountOverflow)
}

/// Parse a 44-byte index record (`buf.len() >= 44`).
fn parse_index_entry(buf: &[u8]) -> IndexEntry {
    let mut packed = [0u8; 32];
    packed.copy_from_slice(&buf[..32]);
    IndexEntry {
        packed,
        moves_offset: read_u64_le(&buf[32..40]),
        ply: read_u16_le(&buf[40..42]),
        move_count: read_u16_le(&buf[42..44]),
    }
}

/// Decode `move_count` consecutive move records from `region`
/// (`region.len() >= move_count * record_size`).
fn decode_moves(
    region: &[u8],
    move_count: u16,
    record_size: u64,
    has_depth: bool,
) -> Vec<BookMove> {
    let record_size = record_size as usize;
    let mut moves = Vec::with_capacity(move_count as usize);
    for i in 0..move_count as usize {
        let base = i * record_size;
        let move16 = read_u16_le(&region[base..base + 2]);
        let value = read_u16_le(&region[base + 2..base + 4]) as i16;
        let depth = if has_depth {
            read_u16_le(&region[base + 4..base + 6])
        } else {
            0
        };
        moves.push(BookMove {
            move16,
            value,
            depth,
            count: 0,
        });
    }
    moves
}

/// Seek `file` to `offset` and fill `buf`. `Ok(true)` on a full read, `Ok(false)`
/// on a clean end-of-file (graceful miss), `Err` on any other I/O error.
///
/// Reads through `&File` (which implements `Read`/`Seek`) so the caller keeps a
/// shared borrow — the twin handles are never mutated through `&self`.
fn read_at(mut file: &File, offset: u64, buf: &mut [u8]) -> Result<bool, BookError> {
    file.seek(SeekFrom::Start(offset))?;
    match file.read_exact(buf) {
        Ok(()) => Ok(true),
        Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => Ok(false),
        Err(e) => Err(BookError::Io(e)),
    }
}

/// Read a little-endian `u16` from a 2-byte slice (`b.len() >= 2`).
fn read_u16_le(b: &[u8]) -> u16 {
    u16::from_le_bytes([b[0], b[1]])
}

/// Read a little-endian `u64` from an 8-byte slice (`b.len() >= 8`).
fn read_u64_le(b: &[u8]) -> u64 {
    let mut bytes = [0u8; 8];
    bytes.copy_from_slice(&b[..8]);
    u64::from_le_bytes(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `(move16, value, depth)`.
    type RawMove = (u16, i16, u16);
    /// `(packed key, ply, moves)`.
    type RawRecord = ([u8; 32], u16, Vec<RawMove>);

    /// Build a minimal well-formed `.ybb` from `(packed, ply, moves)` records.
    /// Records are sorted by packed key (as the format requires). Each move is
    /// `(move16, value, depth)`; `depth` is written iff `with_depth`.
    fn build_ybb(records: &[RawRecord], with_depth: bool) -> Vec<u8> {
        let mut sorted = records.to_vec();
        sorted.sort_by_key(|r| r.0);

        let flags: u64 = if with_depth { FLAG_MOVE_DEPTH } else { 0 };

        let mut header = Vec::new();
        header.extend_from_slice(MAGIC);
        header.extend_from_slice(&(sorted.len() as u64).to_le_bytes());
        header.extend_from_slice(&flags.to_le_bytes());

        let mut index = Vec::new();
        let mut moves = Vec::new();
        for (packed, ply, mvs) in &sorted {
            let moves_offset = moves.len() as u64;
            index.extend_from_slice(packed);
            index.extend_from_slice(&moves_offset.to_le_bytes());
            index.extend_from_slice(&ply.to_le_bytes());
            index.extend_from_slice(&(mvs.len() as u16).to_le_bytes());
            for (m, v, d) in mvs {
                moves.extend_from_slice(&m.to_le_bytes());
                moves.extend_from_slice(&(*v as u16).to_le_bytes());
                if with_depth {
                    moves.extend_from_slice(&d.to_le_bytes());
                }
            }
        }
        assert_eq!(index.len(), sorted.len() * INDEX_RECORD_SIZE as usize);
        let mut out = header;
        out.extend_from_slice(&index);
        out.extend_from_slice(&moves);
        out
    }

    fn key(first: u8) -> [u8; 32] {
        let mut k = [0u8; 32];
        k[0] = first;
        k
    }

    #[test]
    fn probes_hits_and_misses_with_depth() {
        let recs = vec![
            (
                key(3),
                1u16,
                vec![(0x1111u16, 42i16, 20u16), (0x2222, -13, 18)],
            ),
            (key(1), 1, vec![(0x3333, 0, 0)]),
            (key(7), 5, vec![(0x4444, 100, 7)]),
        ];
        let bytes = build_ybb(&recs, true);
        let book = Book::from_memory(bytes).unwrap();
        assert_eq!(book.record_count(), 3);
        assert!(book.has_move_depth());

        let hit = book.probe(&key(3), 1, false).unwrap().unwrap();
        assert_eq!(
            hit,
            vec![
                BookMove {
                    move16: 0x1111,
                    value: 42,
                    depth: 20,
                    count: 0
                },
                BookMove {
                    move16: 0x2222,
                    value: -13,
                    depth: 18,
                    count: 0
                },
            ]
        );

        // present key, wrong ply -> miss (ply enforced)
        assert!(book.probe(&key(3), 2, false).unwrap().is_none());
        // present key, wrong ply, but ply ignored -> hit
        assert!(book.probe(&key(3), 2, true).unwrap().is_some());
        // absent key -> miss
        assert!(book.probe(&key(9), 1, false).unwrap().is_none());
        assert!(book.probe(&key(0), 1, false).unwrap().is_none());
    }

    #[test]
    fn without_depth_flag_depth_is_zero() {
        let recs = vec![(key(5), 3, vec![(0xABCDu16, -1i16, 0u16)])];
        let bytes = build_ybb(&recs, false);
        let book = Book::from_memory(bytes).unwrap();
        assert!(!book.has_move_depth());
        let hit = book.probe(&key(5), 3, false).unwrap().unwrap();
        assert_eq!(
            hit,
            vec![BookMove {
                move16: 0xABCD,
                value: -1,
                depth: 0,
                count: 0
            }]
        );
    }

    #[test]
    fn bad_magic_and_flags_rejected() {
        let mut bytes = build_ybb(&[(key(1), 1, vec![(1, 0, 0)])], true);
        bytes[0] = b'X';
        assert!(matches!(Book::from_memory(bytes), Err(BookError::BadMagic)));

        let mut bytes = build_ybb(&[(key(1), 1, vec![(1, 0, 0)])], true);
        // set an unknown flag bit
        bytes[24] = 0x02;
        assert!(matches!(
            Book::from_memory(bytes),
            Err(BookError::UnknownFlags(_))
        ));
    }

    #[test]
    fn short_header_and_index_rejected() {
        assert!(matches!(
            Book::from_memory(vec![0u8; 10]),
            Err(BookError::TruncatedHeader)
        ));

        // Valid header claiming 2 records but no index bytes present.
        let mut bytes = Vec::new();
        bytes.extend_from_slice(MAGIC);
        bytes.extend_from_slice(&2u64.to_le_bytes());
        bytes.extend_from_slice(&1u64.to_le_bytes());
        assert!(matches!(
            Book::from_memory(bytes),
            Err(BookError::TruncatedIndex)
        ));
    }

    #[test]
    fn truncated_moves_region_is_a_miss_not_a_panic() {
        let recs = vec![(key(1), 1, vec![(0x1111u16, 5i16, 9u16)])];
        let mut bytes = build_ybb(&recs, true);
        // Drop the last two bytes of the move record: header+index are intact,
        // so open succeeds, but the moves read runs off the end -> miss.
        bytes.truncate(bytes.len() - 2);
        let book = Book::from_memory(bytes).unwrap();
        assert!(book.probe(&key(1), 1, false).unwrap().is_none());
    }
}
