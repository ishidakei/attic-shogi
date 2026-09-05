use std::collections::{BTreeMap, BTreeSet};

use crate::option_profile::BookOptionsVersion;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum OptionDecl {
    Spin {
        name: &'static str,
        default: i64,
        min: i64,
        max: i64,
    },
    String {
        name: &'static str,
        default: &'static str,
    },
    /// A boolean toggle (`type check`).
    Check { name: &'static str, default: bool },
    /// A fixed choice list (`type combo`). The default must be one of `choices`.
    Combo {
        name: &'static str,
        default: &'static str,
        choices: &'static [&'static str],
    },
}

impl OptionDecl {
    pub fn name(&self) -> &'static str {
        match self {
            OptionDecl::Spin { name, .. }
            | OptionDecl::String { name, .. }
            | OptionDecl::Check { name, .. }
            | OptionDecl::Combo { name, .. } => name,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum OptionValue {
    Spin(i64),
    String(String),
    Check(bool),
    Combo(String),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum OptionError {
    UnknownOption,
    NotAnInteger(String),
    OutOfRange { value: i64, min: i64, max: i64 },
    NotABool(String),
    InvalidComboChoice(String),
}

impl std::fmt::Display for OptionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OptionError::UnknownOption => write!(f, "unknown option"),
            OptionError::NotAnInteger(s) => write!(f, "value `{s}` is not an integer"),
            OptionError::OutOfRange { value, min, max } => {
                write!(f, "value {value} out of range [{min}, {max}]")
            }
            OptionError::NotABool(s) => write!(f, "value `{s}` is not a boolean"),
            OptionError::InvalidComboChoice(s) => write!(f, "value `{s}` is not a valid choice"),
        }
    }
}

/// The reference's `MaxThreads` (`engine.h`): `max(1024, 4 · cores)`, the
/// upper bound of the `Threads` spin option (`engine.cpp`).
fn max_threads() -> i64 {
    let cores = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1) as i64;
    std::cmp::max(1024, 4 * cores)
}

// The declarations below follow the reference's own (yaneuraou-engine, NNUE
// eval, 64-bit, non-Stockfish):
//
//   USI_Hash : yaneuraou-search.cpp  -> default 1024, min 1, max MaxHashMB
//   Threads  : engine.cpp        -> default 4,    min 1, max MaxThreads
//   MultiPV  : yaneuraou-search.cpp  -> default 1,    min 1, max MAX_MOVES
//   EvalDir  : eval/nnue/evaluate_nnue.cpp -> default "eval"
//
// 64-bit MaxHashMB is 33554432 (engine.h), but MaxThreads is dynamic, so the
// list is built per store rather than being a compile-time constant.

/// `MAX_PLY` (`types.h`), the upper bound of `BookPvMoves`.
const MAX_PLY: i64 = 246;

/// The `EnteringKingRule` combo choice list — `EKR_STRINGS` in the reference's
/// exact order (`types.cpp`), mirroring [`EnteringKingRule::STRINGS`].
const ENTERING_KING_RULE_CHOICES: &[&str] = &[
    "NoEnteringKing",
    "CSARule24",
    "CSARule24H",
    "CSARule27",
    "CSARule27H",
    "TryRule",
];

/// The `BookFile` combo choice list: the reference's stems
/// (`book.cpp`) respelled with the `.ybb` extension, since this engine
/// reads only `.ybb` books and a combo must not advertise values it cannot use.
///
/// A combo rejects any value outside this list, so the `.db → .ybb` sibling
/// fallback in [`crate::driver`] is unreachable from the option surface.
const BOOK_FILE_CHOICES: &[&str] = &[
    "no_book",
    "standard_book.ybb",
    "yaneura_book1.ybb",
    "yaneura_book2.ybb",
    "yaneura_book3.ybb",
    "yaneura_book4.ybb",
    "user_book1.ybb",
    "user_book2.ybb",
    "user_book3.ybb",
    "book.ybb",
];

/// Build the declaration list for `book_options` — the reference `add_options`
/// chain, whose book group branches on `OptionsMap::book_options_v2()`
/// (`book.cpp`). Everything outside that group is profile-independent.
fn declarations(book_options: BookOptionsVersion) -> Vec<OptionDecl> {
    let v2 = book_options.is_v2();
    let mut decls = vec![
        OptionDecl::Spin {
            name: "USI_Hash",
            default: 1024,
            min: 1,
            max: 33_554_432,
        },
        OptionDecl::Spin {
            name: "Threads",
            default: 4,
            min: 1,
            max: max_threads(),
        },
        OptionDecl::Spin {
            name: "MultiPV",
            default: 1,
            min: 1,
            max: 600,
        },
        OptionDecl::String {
            name: "EvalDir",
            default: "eval",
        },
        // NNUE fixed-point output scale (`evaluate_nnue.cpp`).
        // Suisho-family nets recommend 24 or more.
        OptionDecl::Spin {
            name: "FV_SCALE",
            default: 16,
            min: 1,
            max: 128,
        },
    ];

    // Opening-book options (book.cpp), profile-dependent.
    decls.extend(book_declarations(v2));

    decls.extend([
        // Entering-king (nyugyoku) declaration rule
        // (`yaneuraou-search.cpp`).
        OptionDecl::Combo {
            name: "EnteringKingRule",
            default: "CSARule27",
            choices: ENTERING_KING_RULE_CHOICES,
        },
        // Per-`go` search-depth / node ceilings, `0` meaning unlimited
        // (`engine.cpp`).
        OptionDecl::Spin {
            name: "DepthLimit",
            default: 0,
            min: 0,
            max: 2_147_483_647,
        },
        OptionDecl::Spin {
            name: "NodesLimit",
            default: 0,
            min: 0,
            max: 9_223_372_036_854_775_807,
        },
        // The game ply past which the search adjudicates an unconditional draw
        // (`yaneuraou-search.cpp`). A set value of `0` means unlimited.
        OptionDecl::Spin {
            name: "MaxMovesToDraw",
            default: 0,
            min: 0,
            max: 100_000,
        },
        // The PV-output throttle interval in ms (`0` never suppresses), the
        // consideration mode, and whether a fail-high/low emits a PV
        // (`yaneuraou-search.cpp`).
        OptionDecl::Spin {
            name: "PvInterval",
            default: 300,
            min: 0,
            max: 100_000_000,
        },
        OptionDecl::Check {
            name: "ConsiderationMode",
            default: false,
        },
        OptionDecl::Check {
            name: "OutputFailLHPV",
            default: true,
        },
        // The per-color draw score in centipawns, from each color's own
        // perspective (`yaneuraou-search.cpp`).
        OptionDecl::Spin {
            name: "DrawValueBlack",
            default: -2,
            min: -30000,
            max: 30000,
        },
        OptionDecl::Spin {
            name: "DrawValueWhite",
            default: -2,
            min: -30000,
            max: 30000,
        },
        // A centipawn-normalized best score at or below `-ResignValue` resigns
        // (`yaneuraou-search.cpp`); the default is effectively unreachable.
        OptionDecl::Spin {
            name: "ResignValue",
            default: 99999,
            min: 0,
            max: 99999,
        },
        // When true the search also considers the non-promoting moves the default
        // generator suppresses (`yaneuraou-search.cpp`).
        OptionDecl::Check {
            name: "GenerateAllLegalMoves",
            default: false,
        },
        // The average and worst-case GUI round-trip margins in ms, subtracted
        // from the clock so the search never overruns (`timeman.cpp`).
        OptionDecl::Spin {
            name: "NetworkDelay",
            default: 120,
            min: 0,
            max: 10000,
        },
        OptionDecl::Spin {
            name: "NetworkDelay2",
            default: 1120,
            min: 0,
            max: 10000,
        },
        // The floor in ms on a move's optimum time, before the network delay is
        // subtracted (`timeman.cpp`).
        OptionDecl::Spin {
            name: "MinimumThinkingTime",
            default: 2000,
            min: 1,
            max: 100000,
        },
        // A percentage multiplier on the optimum time (`timeman.cpp`).
        OptionDecl::Spin {
            name: "SlowMover",
            default: 100,
            min: 1,
            max: 1000,
        },
        // Use the clock right up to each whole second rather than leaving
        // sub-second slack (`timeman.cpp`).
        OptionDecl::Check {
            name: "RoundUpToFullSecond",
            default: true,
        },
        // How the machine maps to logical NUMA nodes, and whether worker threads
        // bind (`engine.cpp`). `auto` / `system` respect the process
        // affinity, `hardware` ignores it, `none` disables binding, and any other
        // value is a custom `':'`-separated node string.
        OptionDecl::String {
            name: "NumaPolicy",
            default: "auto",
        },
        // `engine.cpp`. At the option layer these only feed the
        // `optimumTime` bonus (`timeman.cpp`).
        OptionDecl::Check {
            name: "USI_Ponder",
            default: false,
        },
        OptionDecl::Check {
            name: "Stochastic_Ponder",
            default: false,
        },
    ]);

    decls
}

/// The opening-book option group, in the reference registration order
/// (`book.cpp`). `v2` drops `NarrowBook` / `ConsiderBookMoveCount`,
/// swaps `BookEvalDiff` and `BookDepthLimit` for their black/white-split
/// counterparts, and shifts the defaults towards large books.
fn book_declarations(v2: bool) -> Vec<OptionDecl> {
    let mut decls = vec![OptionDecl::Check {
        name: "USI_OwnBook",
        default: true,
    }];

    // V1 only (`book.cpp`); V2 behaves as if it were always false.
    if !v2 {
        decls.push(OptionDecl::Check {
            name: "NarrowBook",
            default: false,
        });
    }

    decls.extend([
        OptionDecl::Spin {
            name: "BookMoves",
            default: if v2 { 200 } else { 16 },
            min: 0,
            max: 10000,
        },
        OptionDecl::Spin {
            name: "BookIgnoreRate",
            default: 0,
            min: 0,
            max: 100,
        },
        OptionDecl::Combo {
            // The reference defaults to a real book file; this engine is
            // bookless in both profiles unless `BookFile` is set explicitly.
            name: "BookFile",
            default: "no_book",
            choices: BOOK_FILE_CHOICES,
        },
        OptionDecl::String {
            name: "BookDir",
            default: "book",
        },
    ]);

    // One option under V1, split per root side to move under V2
    // (`book.cpp`).
    if v2 {
        decls.extend([
            OptionDecl::Spin {
                name: "BookEvalBlackDiff",
                default: 0,
                min: 0,
                max: 99999,
            },
            OptionDecl::Spin {
                name: "BookEvalWhiteDiff",
                default: 0,
                min: 0,
                max: 99999,
            },
        ]);
    } else {
        decls.push(OptionDecl::Spin {
            name: "BookEvalDiff",
            default: 30,
            min: 0,
            max: 99999,
        });
    }

    decls.extend([
        OptionDecl::Spin {
            name: "BookEvalBlackLimit",
            default: 0,
            min: -99999,
            max: 99999,
        },
        OptionDecl::Spin {
            name: "BookEvalWhiteLimit",
            default: -140,
            min: -99999,
            max: 99999,
        },
    ]);

    // Likewise split per root side to move under V2 (`book.cpp`).
    if v2 {
        decls.extend([
            OptionDecl::Spin {
                name: "BookDepthBlackLimit",
                default: 0,
                min: 0,
                max: 99999,
            },
            OptionDecl::Spin {
                name: "BookDepthWhiteLimit",
                default: 5,
                min: 0,
                max: 99999,
            },
        ]);
    } else {
        decls.push(OptionDecl::Spin {
            name: "BookDepthLimit",
            default: 16,
            min: 0,
            max: 99999,
        });
    }

    // V2 targets huge books, so streaming reads default on (`book.cpp`).
    decls.push(OptionDecl::Check {
        name: "BookOnTheFly",
        default: v2,
    });

    // V1 only (`book.cpp`).
    if !v2 {
        decls.push(OptionDecl::Check {
            name: "ConsiderBookMoveCount",
            default: false,
        });
    }

    decls.extend([
        OptionDecl::Spin {
            name: "BookPvMoves",
            default: 8,
            min: 1,
            max: MAX_PLY,
        },
        // Also defaults on under V2 (`book.cpp`).
        OptionDecl::Check {
            name: "IgnoreBookPly",
            default: v2,
        },
        OptionDecl::Check {
            name: "FlippedBook",
            default: true,
        },
    ]);

    decls
}

#[derive(Clone, Debug)]
pub struct OptionStore {
    /// The declared options, in declaration order. Owned per store because the
    /// `Threads` upper bound is computed at runtime.
    decls: Vec<OptionDecl>,
    values: BTreeMap<&'static str, OptionValue>,
    /// Options locked by an override file (`usioption.cpp`). A fixed option
    /// ignores every later [`set_value`] silently, as the reference
    /// `Option::operator=` does (`usioption.cpp`).
    fixed: BTreeSet<&'static str>,
    /// The book-option profile this store was built with (`usioption.h`).
    /// The probe reads it back to pick the side-to-move-dependent option names.
    book_options: BookOptionsVersion,
}

impl OptionStore {
    /// A store with the default V1 book-option surface, as the reference uses
    /// when no `engine_option_profile.txt` is present.
    pub fn new() -> Self {
        Self::with_book_options(BookOptionsVersion::default())
    }

    /// A store whose book-option group follows `book_options`, selected by
    /// `engine_option_profile.txt` before the `usi` reply.
    pub fn with_book_options(book_options: BookOptionsVersion) -> Self {
        let decls = declarations(book_options);
        let mut values = BTreeMap::new();
        for decl in &decls {
            let default = match decl {
                OptionDecl::Spin { default, .. } => OptionValue::Spin(*default),
                OptionDecl::String { default, .. } => OptionValue::String((*default).to_string()),
                OptionDecl::Check { default, .. } => OptionValue::Check(*default),
                OptionDecl::Combo { default, .. } => OptionValue::Combo((*default).to_string()),
            };
            values.insert(decl.name(), default);
        }
        Self {
            decls,
            values,
            fixed: BTreeSet::new(),
            book_options,
        }
    }

    pub fn iter_declarations(&self) -> impl Iterator<Item = &OptionDecl> {
        self.decls.iter()
    }

    /// Whether the book options were registered under the V2 profile
    /// (`usioption.h`).
    pub fn book_options_v2(&self) -> bool {
        self.book_options.is_v2()
    }

    pub fn get(&self, name: &str) -> Option<&OptionValue> {
        self.decls
            .iter()
            .find(|d| d.name() == name)
            .and_then(|d| self.values.get(d.name()))
    }

    /// The current `Threads` value as a pool size, at least 1.
    pub fn threads(&self) -> usize {
        match self.get("Threads") {
            Some(OptionValue::Spin(n)) => (*n).max(1) as usize,
            // Unreachable: `Threads` is a declared spin.
            _ => 1,
        }
    }

    pub fn set_value(&mut self, name: &str, value: &str) -> Result<(), OptionError> {
        let decl = self
            .decls
            .iter()
            .find(|d| d.name() == name)
            .ok_or(OptionError::UnknownOption)?;
        // A fixed option silently ignores the assignment — no mutation, no
        // error, no output (`usioption.cpp`).
        if self.fixed.contains(decl.name()) {
            return Ok(());
        }
        match decl {
            OptionDecl::Spin { min, max, .. } => {
                let parsed: i64 = value
                    .parse()
                    .map_err(|_| OptionError::NotAnInteger(value.to_string()))?;
                if parsed < *min || parsed > *max {
                    return Err(OptionError::OutOfRange {
                        value: parsed,
                        min: *min,
                        max: *max,
                    });
                }
                self.values.insert(decl.name(), OptionValue::Spin(parsed));
            }
            OptionDecl::String { .. } => {
                self.values
                    .insert(decl.name(), OptionValue::String(value.to_string()));
            }
            OptionDecl::Check { .. } => {
                let parsed = parse_check(value)?;
                self.values.insert(decl.name(), OptionValue::Check(parsed));
            }
            OptionDecl::Combo { choices, .. } => {
                if !choices.contains(&value) {
                    return Err(OptionError::InvalidComboChoice(value.to_string()));
                }
                self.values
                    .insert(decl.name(), OptionValue::Combo(value.to_string()));
            }
        }
        Ok(())
    }

    /// A declared spin's current value, `0` if the name is not a spin.
    pub fn spin(&self, name: &str) -> i64 {
        match self.get(name) {
            Some(OptionValue::Spin(v)) => *v,
            _ => 0,
        }
    }

    /// A declared check's current value (`false` if the name is not a check).
    pub fn check(&self, name: &str) -> bool {
        match self.get(name) {
            Some(OptionValue::Check(v)) => *v,
            _ => false,
        }
    }

    /// Resolve `name` to its canonical declared spelling, comparing
    /// case-insensitively as the reference `OptionsMap` does
    /// (`usioption.h`).
    pub fn canonical_name(&self, name: &str) -> Option<&'static str> {
        self.decls
            .iter()
            .find(|d| d.name().eq_ignore_ascii_case(name))
            .map(|d| d.name())
    }

    /// Lock an option against further [`set_value`] mutation, the reference's
    /// `Option::fixed` (`usioption.cpp`). Idempotent.
    pub fn mark_fixed(&mut self, name: &'static str) {
        self.fixed.insert(name);
    }

    /// Whether an option is currently locked by an override.
    pub fn is_fixed(&self, name: &str) -> bool {
        match self.canonical_name(name) {
            Some(n) => self.fixed.contains(n),
            None => false,
        }
    }

    /// A declared string/combo's current value (`""` if the name is neither).
    pub fn text(&self, name: &str) -> &str {
        match self.get(name) {
            Some(OptionValue::String(s)) | Some(OptionValue::Combo(s)) => s.as_str(),
            _ => "",
        }
    }
}

/// Parse a USI `check` value (`true` / `false`, case-insensitive).
fn parse_check(value: &str) -> Result<bool, OptionError> {
    match value.trim().to_ascii_lowercase().as_str() {
        "true" => Ok(true),
        "false" => Ok(false),
        _ => Err(OptionError::NotABool(value.to_string())),
    }
}

impl Default for OptionStore {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_loaded_from_declarations() {
        let s = OptionStore::new();
        assert_eq!(s.get("USI_Hash"), Some(&OptionValue::Spin(1024)));
        assert_eq!(s.get("Threads"), Some(&OptionValue::Spin(4)));
        assert_eq!(s.get("MultiPV"), Some(&OptionValue::Spin(1)));
        assert_eq!(
            s.get("EvalDir"),
            Some(&OptionValue::String("eval".to_string()))
        );
        assert_eq!(s.get("FV_SCALE"), Some(&OptionValue::Spin(16)));
        assert_eq!(
            s.get("BookFile"),
            Some(&OptionValue::Combo("no_book".to_string()))
        );
        assert_eq!(s.get("USI_OwnBook"), Some(&OptionValue::Check(true)));
        assert_eq!(s.get("FlippedBook"), Some(&OptionValue::Check(true)));
        assert_eq!(s.get("NarrowBook"), Some(&OptionValue::Check(false)));
        assert_eq!(s.get("BookMoves"), Some(&OptionValue::Spin(16)));
        assert_eq!(s.get("BookEvalWhiteLimit"), Some(&OptionValue::Spin(-140)));
        assert_eq!(s.get("BookPvMoves"), Some(&OptionValue::Spin(8)));
        assert_eq!(
            s.get("EnteringKingRule"),
            Some(&OptionValue::Combo("CSARule27".to_string()))
        );
        assert_eq!(s.get("DepthLimit"), Some(&OptionValue::Spin(0)));
        assert_eq!(s.get("NodesLimit"), Some(&OptionValue::Spin(0)));
        assert_eq!(s.get("MaxMovesToDraw"), Some(&OptionValue::Spin(0)));
        assert_eq!(s.get("PvInterval"), Some(&OptionValue::Spin(300)));
        assert_eq!(s.get("ConsiderationMode"), Some(&OptionValue::Check(false)));
        assert_eq!(s.get("OutputFailLHPV"), Some(&OptionValue::Check(true)));
        assert_eq!(s.get("DrawValueBlack"), Some(&OptionValue::Spin(-2)));
        assert_eq!(s.get("DrawValueWhite"), Some(&OptionValue::Spin(-2)));
        assert_eq!(s.get("ResignValue"), Some(&OptionValue::Spin(99999)));
        assert_eq!(
            s.get("GenerateAllLegalMoves"),
            Some(&OptionValue::Check(false))
        );
    }

    #[test]
    fn v2_profile_swaps_the_book_group() {
        let s = OptionStore::with_book_options(BookOptionsVersion::V2);
        assert!(s.book_options_v2());

        assert_eq!(s.get("BookEvalBlackDiff"), Some(&OptionValue::Spin(0)));
        assert_eq!(s.get("BookEvalWhiteDiff"), Some(&OptionValue::Spin(0)));
        assert_eq!(s.get("BookDepthBlackLimit"), Some(&OptionValue::Spin(0)));
        assert_eq!(s.get("BookDepthWhiteLimit"), Some(&OptionValue::Spin(5)));
        for gone in [
            "NarrowBook",
            "ConsiderBookMoveCount",
            "BookEvalDiff",
            "BookDepthLimit",
        ] {
            assert_eq!(s.get(gone), None, "{gone} must be absent under V2");
        }

        assert_eq!(s.get("BookMoves"), Some(&OptionValue::Spin(200)));
        assert_eq!(s.get("BookOnTheFly"), Some(&OptionValue::Check(true)));
        assert_eq!(s.get("IgnoreBookPly"), Some(&OptionValue::Check(true)));

        // Unchanged in both profiles.
        assert_eq!(
            s.get("BookFile"),
            Some(&OptionValue::Combo("no_book".to_string()))
        );
        assert_eq!(s.get("USI_OwnBook"), Some(&OptionValue::Check(true)));
        assert_eq!(s.get("BookIgnoreRate"), Some(&OptionValue::Spin(0)));
        assert_eq!(
            s.get("BookDir"),
            Some(&OptionValue::String("book".to_string()))
        );
        assert_eq!(s.get("BookEvalBlackLimit"), Some(&OptionValue::Spin(0)));
        assert_eq!(s.get("BookEvalWhiteLimit"), Some(&OptionValue::Spin(-140)));
        assert_eq!(s.get("BookPvMoves"), Some(&OptionValue::Spin(8)));
        assert_eq!(s.get("FlippedBook"), Some(&OptionValue::Check(true)));

        assert_eq!(s.get("USI_Hash"), Some(&OptionValue::Spin(1024)));
        assert_eq!(s.get("Threads"), Some(&OptionValue::Spin(4)));
        assert_eq!(s.get("ResignValue"), Some(&OptionValue::Spin(99999)));
    }

    #[test]
    fn v1_is_the_default_profile() {
        let default_names: Vec<_> = OptionStore::new()
            .iter_declarations()
            .map(|d| d.name())
            .collect();
        let v1_names: Vec<_> = OptionStore::with_book_options(BookOptionsVersion::V1)
            .iter_declarations()
            .map(|d| d.name())
            .collect();
        assert_eq!(default_names, v1_names);
        assert!(!OptionStore::new().book_options_v2());
    }

    #[test]
    fn v2_declaration_order_follows_the_reference() {
        let s = OptionStore::with_book_options(BookOptionsVersion::V2);
        let book: Vec<_> = s
            .iter_declarations()
            .map(|d| d.name())
            .skip_while(|n| *n != "USI_OwnBook")
            .take_while(|n| *n != "EnteringKingRule")
            .collect();
        assert_eq!(
            book,
            vec![
                "USI_OwnBook",
                "BookMoves",
                "BookIgnoreRate",
                "BookFile",
                "BookDir",
                "BookEvalBlackDiff",
                "BookEvalWhiteDiff",
                "BookEvalBlackLimit",
                "BookEvalWhiteLimit",
                "BookDepthBlackLimit",
                "BookDepthWhiteLimit",
                "BookOnTheFly",
                "BookPvMoves",
                "IgnoreBookPly",
                "FlippedBook",
            ]
        );
    }

    #[test]
    fn set_check_and_combo() {
        let mut s = OptionStore::new();
        s.set_value("USI_OwnBook", "false").unwrap();
        assert!(!s.check("USI_OwnBook"));
        s.set_value("USI_OwnBook", "TRUE").unwrap();
        assert!(s.check("USI_OwnBook"));
        assert!(matches!(
            s.set_value("USI_OwnBook", "maybe"),
            Err(OptionError::NotABool(_))
        ));

        s.set_value("BookFile", "user_book1.ybb").unwrap();
        assert_eq!(s.text("BookFile"), "user_book1.ybb");
        assert!(matches!(
            s.set_value("BookFile", "not_listed.ybb"),
            Err(OptionError::InvalidComboChoice(_))
        ));
        // The reference's `.db` spellings are not offered, so they are rejected
        // like any other unlisted value and the prior value stands.
        assert!(matches!(
            s.set_value("BookFile", "user_book1.db"),
            Err(OptionError::InvalidComboChoice(_))
        ));
        assert_eq!(s.text("BookFile"), "user_book1.ybb");
    }

    #[test]
    fn every_book_file_choice_carries_a_loadable_spelling() {
        // Apart from the `no_book` sentinel, every advertised choice must name a
        // `.ybb` file, the only book format this engine reads.
        let mut s = OptionStore::new();
        for choice in BOOK_FILE_CHOICES {
            s.set_value("BookFile", choice)
                .unwrap_or_else(|e| panic!("advertised choice `{choice}` rejected: {e}"));
            assert_eq!(s.text("BookFile"), *choice);
            if *choice != "no_book" {
                assert!(
                    choice.ends_with(".ybb"),
                    "advertised choice `{choice}` is not a `.ybb` name"
                );
            }
        }
    }

    #[test]
    fn set_spin_happy_path() {
        let mut s = OptionStore::new();
        s.set_value("USI_Hash", "256").unwrap();
        assert_eq!(s.get("USI_Hash"), Some(&OptionValue::Spin(256)));
    }

    #[test]
    fn set_spin_out_of_range_low() {
        let mut s = OptionStore::new();
        let err = s.set_value("USI_Hash", "0").unwrap_err();
        assert!(matches!(err, OptionError::OutOfRange { .. }));
    }

    #[test]
    fn set_spin_out_of_range_high() {
        let mut s = OptionStore::new();
        let err = s.set_value("Threads", "9999").unwrap_err();
        assert!(matches!(err, OptionError::OutOfRange { .. }));
    }

    #[test]
    fn set_spin_type_mismatch_rejects() {
        let mut s = OptionStore::new();
        let err = s.set_value("USI_Hash", "not-a-number").unwrap_err();
        assert!(matches!(err, OptionError::NotAnInteger(_)));
    }

    #[test]
    fn set_string_happy_path() {
        let mut s = OptionStore::new();
        s.set_value("EvalDir", "/srv/eval").unwrap();
        assert_eq!(
            s.get("EvalDir"),
            Some(&OptionValue::String("/srv/eval".to_string()))
        );
    }

    #[test]
    fn fixed_option_ignores_further_set_value() {
        let mut s = OptionStore::new();
        s.set_value("FV_SCALE", "24").unwrap();
        assert_eq!(s.spin("FV_SCALE"), 24);
        s.mark_fixed("FV_SCALE");
        assert!(s.is_fixed("FV_SCALE"));
        s.set_value("FV_SCALE", "16").unwrap();
        assert_eq!(s.spin("FV_SCALE"), 24);
    }

    #[test]
    fn canonical_name_is_case_insensitive() {
        let s = OptionStore::new();
        assert_eq!(s.canonical_name("fv_scale"), Some("FV_SCALE"));
        assert_eq!(s.canonical_name("USI_HASH"), Some("USI_Hash"));
        assert_eq!(s.canonical_name("nope"), None);
    }

    #[test]
    fn set_unknown_option_rejects() {
        let mut s = OptionStore::new();
        let err = s.set_value("Nonexistent", "x").unwrap_err();
        assert_eq!(err, OptionError::UnknownOption);
    }

    #[test]
    fn iter_declarations_yields_in_declaration_order() {
        let s = OptionStore::new();
        let names: Vec<_> = s.iter_declarations().map(|d| d.name()).collect();
        assert_eq!(
            names,
            vec![
                "USI_Hash",
                "Threads",
                "MultiPV",
                "EvalDir",
                "FV_SCALE",
                "USI_OwnBook",
                "NarrowBook",
                "BookMoves",
                "BookIgnoreRate",
                "BookFile",
                "BookDir",
                "BookEvalDiff",
                "BookEvalBlackLimit",
                "BookEvalWhiteLimit",
                "BookDepthLimit",
                "BookOnTheFly",
                "ConsiderBookMoveCount",
                "BookPvMoves",
                "IgnoreBookPly",
                "FlippedBook",
                "EnteringKingRule",
                "DepthLimit",
                "NodesLimit",
                "MaxMovesToDraw",
                "PvInterval",
                "ConsiderationMode",
                "OutputFailLHPV",
                "DrawValueBlack",
                "DrawValueWhite",
                "ResignValue",
                "GenerateAllLegalMoves",
                "NetworkDelay",
                "NetworkDelay2",
                "MinimumThinkingTime",
                "SlowMover",
                "RoundUpToFullSecond",
                "NumaPolicy",
                "USI_Ponder",
                "Stochastic_Ponder",
            ]
        );
    }
}
