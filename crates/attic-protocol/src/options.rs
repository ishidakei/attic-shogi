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
    /// An empty value for a spin, check or combo option.
    EmptyValue,
    NotAnInteger(String),
    OutOfRange {
        value: i64,
        min: i64,
        max: i64,
    },
    NotABool(String),
    /// `values` is the combo's `default D var A var B …` list after `default`,
    /// which the reference quotes when it reports the miss.
    InvalidComboChoice {
        value: String,
        values: String,
    },
}

impl std::fmt::Display for OptionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OptionError::UnknownOption => write!(f, "unknown option"),
            OptionError::EmptyValue => write!(f, "empty value"),
            OptionError::NotAnInteger(s) => write!(f, "value `{s}` is not an integer"),
            OptionError::OutOfRange { value, min, max } => {
                write!(f, "value {value} out of range [{min}, {max}]")
            }
            OptionError::NotABool(s) => write!(f, "value `{s}` is not a boolean"),
            OptionError::InvalidComboChoice { value, values } => {
                write!(
                    f,
                    "Error! : combo value not found, value = `{value}`, values = {values}"
                )
            }
        }
    }
}

/// The reference's `MaxThreads`: `max(1024, 4 · cores)`, the upper bound of the
/// `Threads` spin option.
fn max_threads() -> i64 {
    let cores = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1) as i64;
    std::cmp::max(1024, 4 * cores)
}

// The declarations below follow the reference's own (yaneuraou-engine, NNUE
// eval, 64-bit, non-Stockfish):
//
//   USI_Hash : default 1024, min 1, max MaxHashMB
//   Threads  : default 4,    min 1, max MaxThreads
//   MultiPV  : default 1,    min 1, max MAX_MOVES
//   EvalDir  : default "eval"
//
// 64-bit MaxHashMB is 33554432, but MaxThreads is dynamic, so the list is built
// per store rather than being a compile-time constant.

/// `MAX_PLY`, the upper bound of `BookPvMoves`.
const MAX_PLY: i64 = 246;

/// The `EnteringKingRule` combo choice list — `EKR_STRINGS` in the reference's
/// exact order, mirroring [`EnteringKingRule::STRINGS`].
const ENTERING_KING_RULE_CHOICES: &[&str] = &[
    "NoEnteringKing",
    "CSARule24",
    "CSARule24H",
    "CSARule27",
    "CSARule27H",
    "TryRule",
];

/// The `BookFile` combo choice list: the reference's stems respelled with the
/// `.ybb` extension, since this engine reads only `.ybb` books and a combo must
/// not advertise values it cannot use.
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
/// chain, whose book group branches on `OptionsMap::book_options_v2()`.
/// Everything outside that group is profile-independent.
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
        // NNUE fixed-point output scale. Suisho-family nets recommend 24 or
        // more.
        OptionDecl::Spin {
            name: "FV_SCALE",
            default: 16,
            min: 1,
            max: 128,
        },
    ];

    // Opening-book options, profile-dependent.
    decls.extend(book_declarations(v2));

    decls.extend([
        // Entering-king (nyugyoku) declaration rule.
        OptionDecl::Combo {
            name: "EnteringKingRule",
            default: "CSARule27",
            choices: ENTERING_KING_RULE_CHOICES,
        },
        // Per-`go` search-depth / node ceilings, `0` meaning unlimited.
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
        // The game ply past which the search adjudicates an unconditional draw.
        // A set value of `0` means unlimited.
        OptionDecl::Spin {
            name: "MaxMovesToDraw",
            default: 0,
            min: 0,
            max: 100_000,
        },
        // The PV-output throttle interval in ms (`0` never suppresses), the
        // consideration mode, and whether a fail-high/low emits a PV.
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
        // perspective.
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
        // A centipawn-normalized best score at or below `-ResignValue` resigns;
        // the default is effectively unreachable.
        OptionDecl::Spin {
            name: "ResignValue",
            default: 99999,
            min: 0,
            max: 99999,
        },
        // When true the search also considers the non-promoting moves the default
        // generator suppresses.
        OptionDecl::Check {
            name: "GenerateAllLegalMoves",
            default: false,
        },
        // The average and worst-case GUI round-trip margins in ms, subtracted
        // from the clock so the search never overruns.
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
        // subtracted.
        OptionDecl::Spin {
            name: "MinimumThinkingTime",
            default: 2000,
            min: 1,
            max: 100000,
        },
        // A percentage multiplier on the optimum time.
        OptionDecl::Spin {
            name: "SlowMover",
            default: 100,
            min: 1,
            max: 1000,
        },
        // Use the clock right up to each whole second rather than leaving
        // sub-second slack.
        OptionDecl::Check {
            name: "RoundUpToFullSecond",
            default: true,
        },
        // How the machine maps to logical NUMA nodes, and whether worker threads
        // bind. `auto` / `system` respect the process affinity, `hardware`
        // ignores it, `none` disables binding, and any other value is a custom
        // `':'`-separated node string.
        OptionDecl::String {
            name: "NumaPolicy",
            default: "auto",
        },
        // At the option layer these only feed the `optimumTime` bonus.
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

/// The opening-book option group, in the reference registration order. `v2`
/// drops `NarrowBook` / `ConsiderBookMoveCount`, swaps `BookEvalDiff` and
/// `BookDepthLimit` for their black/white-split counterparts, and shifts the
/// defaults towards large books.
fn book_declarations(v2: bool) -> Vec<OptionDecl> {
    let mut decls = vec![OptionDecl::Check {
        name: "USI_OwnBook",
        default: true,
    }];

    // V1 only; V2 behaves as if it were always false.
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

    // One option under V1, split per root side to move under V2.
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

    // Likewise split per root side to move under V2.
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

    // V2 targets huge books, so streaming reads default on.
    decls.push(OptionDecl::Check {
        name: "BookOnTheFly",
        default: v2,
    });

    // V1 only.
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
        // Also defaults on under V2.
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
    /// Each option's value as the text that was accepted, the reference's
    /// `currentValue`: a spin set as `+24` keeps that spelling, a combo keeps
    /// the case it was given in.
    texts: BTreeMap<&'static str, String>,
    /// Options locked by an override file. A fixed option ignores every later
    /// [`set_value`] silently, as the reference `Option::operator=` does.
    fixed: BTreeSet<&'static str>,
    /// The book-option profile this store was built with. The probe reads it
    /// back to pick the side-to-move-dependent option names.
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
        let mut texts = BTreeMap::new();
        for decl in &decls {
            let default = match decl {
                OptionDecl::Spin { default, .. } => OptionValue::Spin(*default),
                OptionDecl::String { default, .. } => OptionValue::String((*default).to_string()),
                OptionDecl::Check { default, .. } => OptionValue::Check(*default),
                OptionDecl::Combo { default, .. } => OptionValue::Combo((*default).to_string()),
            };
            texts.insert(decl.name(), render(&default));
            values.insert(decl.name(), default);
        }
        Self {
            decls,
            values,
            texts,
            fixed: BTreeSet::new(),
            book_options,
        }
    }

    pub fn iter_declarations(&self) -> impl Iterator<Item = &OptionDecl> {
        self.decls.iter()
    }

    /// Whether the book options were registered under the V2 profile.
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

    /// Set a declared option, matching `name` case-insensitively (ASCII) as the
    /// reference `OptionsMap` does; the value is stored under the canonical name.
    pub fn set_value(&mut self, name: &str, value: &str) -> Result<(), OptionError> {
        let decl = self
            .decls
            .iter()
            .find(|d| d.name().eq_ignore_ascii_case(name))
            .ok_or(OptionError::UnknownOption)?;
        // A fixed option silently ignores the assignment — no mutation, no
        // error, no output.
        if self.fixed.contains(decl.name()) {
            return Ok(());
        }
        if value.is_empty() && !matches!(decl, OptionDecl::String { .. }) {
            return Err(OptionError::EmptyValue);
        }
        let (parsed, text) = match decl {
            OptionDecl::Spin { min, max, .. } => {
                let parsed = parse_spin(value)?;
                if parsed < *min || parsed > *max {
                    return Err(OptionError::OutOfRange {
                        value: parsed,
                        min: *min,
                        max: *max,
                    });
                }
                (OptionValue::Spin(parsed), value)
            }
            OptionDecl::String { .. } => {
                let text = if value == "<empty>" { "" } else { value };
                (OptionValue::String(text.to_string()), text)
            }
            OptionDecl::Check { .. } => (OptionValue::Check(parse_check(value)?), value),
            OptionDecl::Combo {
                default, choices, ..
            } => {
                // The reference matches case-insensitively but keeps the value
                // as given, so a reader comparing exact spellings (the
                // `EnteringKingRule` mapping, the `no_book` sentinel) does not
                // recognise `csarule27` or `NO_BOOK`.
                if !choices.iter().any(|c| c.eq_ignore_ascii_case(value)) {
                    let mut values = (*default).to_string();
                    for choice in *choices {
                        values.push_str(" var ");
                        values.push_str(choice);
                    }
                    return Err(OptionError::InvalidComboChoice {
                        value: value.to_string(),
                        values,
                    });
                }
                (OptionValue::Combo(value.to_string()), value)
            }
        };
        self.values.insert(decl.name(), parsed);
        self.texts.insert(decl.name(), text.to_string());
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
    /// case-insensitively as the reference `OptionsMap` does.
    pub fn canonical_name(&self, name: &str) -> Option<&'static str> {
        self.decls
            .iter()
            .find(|d| d.name().eq_ignore_ascii_case(name))
            .map(|d| d.name())
    }

    /// Lock an option against further [`set_value`] mutation, the reference's
    /// `Option::fixed`. Idempotent.
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

    /// Whether `name`, compared case-insensitively, is a declared string option.
    pub fn is_string(&self, name: &str) -> bool {
        self.decls
            .iter()
            .any(|d| matches!(d, OptionDecl::String { .. }) && d.name().eq_ignore_ascii_case(name))
    }

    /// The current value of a declared option as the text it was set with,
    /// `None` if `name` is not declared.
    pub fn value_text(&self, name: &str) -> Option<&str> {
        self.get(name)?;
        self.texts.get(name).map(String::as_str)
    }

    /// A declared string/combo's current value (`""` if the name is neither).
    pub fn text(&self, name: &str) -> &str {
        match self.get(name) {
            Some(OptionValue::String(s)) | Some(OptionValue::Combo(s)) => s.as_str(),
            _ => "",
        }
    }
}

fn render(value: &OptionValue) -> String {
    match value {
        OptionValue::Spin(v) => v.to_string(),
        OptionValue::Check(b) => b.to_string(),
        OptionValue::String(s) | OptionValue::Combo(s) => s.clone(),
    }
}

/// Parse a USI `check` value: exactly `true` or `false`, as the reference
/// accepts.
fn parse_check(value: &str) -> Result<bool, OptionError> {
    match value {
        "true" => Ok(true),
        "false" => Ok(false),
        _ => Err(OptionError::NotABool(value.to_string())),
    }
}

/// Parse a spin value as the reference's `strtoll` check does: leading C
/// whitespace and a sign are allowed, the whole remainder must be decimal
/// digits, and overflow is a rejection. `strtoll` reads a C string, so the
/// text ends at the first NUL; a value starting with one converts nothing and
/// leaves the end pointer on the terminator, which the check reads as `0`.
fn parse_spin(value: &str) -> Result<i64, OptionError> {
    let c_str = value.split('\0').next().unwrap_or("");
    if c_str.is_empty() {
        return Ok(0);
    }
    c_str
        .trim_start_matches([' ', '\t', '\n', '\x0B', '\x0C', '\r'])
        .parse()
        .map_err(|_| OptionError::NotAnInteger(value.to_string()))
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
    fn set_check_accepts_only_lowercase_true_and_false() {
        let mut s = OptionStore::new();
        s.set_value("USI_OwnBook", "false").unwrap();
        assert!(!s.check("USI_OwnBook"));
        s.set_value("USI_OwnBook", "true").unwrap();
        assert!(s.check("USI_OwnBook"));
        s.set_value("USI_OwnBook", "false").unwrap();
        for rejected in ["True", "TRUE", "maybe", " true", "true "] {
            assert_eq!(
                s.set_value("USI_OwnBook", rejected),
                Err(OptionError::NotABool(rejected.to_string()))
            );
            assert!(!s.check("USI_OwnBook"), "`{rejected}` changed the value");
        }
        assert_eq!(s.set_value("USI_OwnBook", ""), Err(OptionError::EmptyValue));
        assert_eq!(s.value_text("USI_OwnBook"), Some("false"));
    }

    #[test]
    fn set_combo_matches_case_insensitively_and_keeps_the_given_spelling() {
        let mut s = OptionStore::new();
        s.set_value("BookFile", "user_book1.ybb").unwrap();
        assert_eq!(s.text("BookFile"), "user_book1.ybb");

        s.set_value("BookFile", "USER_BOOK2.YBB").unwrap();
        assert_eq!(s.text("BookFile"), "USER_BOOK2.YBB");
        assert_eq!(s.value_text("BookFile"), Some("USER_BOOK2.YBB"));

        s.set_value("EnteringKingRule", "tryrule").unwrap();
        assert_eq!(s.text("EnteringKingRule"), "tryrule");
    }

    #[test]
    fn set_combo_rejects_an_unlisted_value_with_the_reference_message() {
        let mut s = OptionStore::new();
        s.set_value("BookFile", "user_book1.ybb").unwrap();
        let err = s.set_value("BookFile", "not_listed.ybb").unwrap_err();
        assert_eq!(
            err.to_string(),
            "Error! : combo value not found, value = `not_listed.ybb`, values = no_book \
             var no_book var standard_book.ybb var yaneura_book1.ybb var yaneura_book2.ybb \
             var yaneura_book3.ybb var yaneura_book4.ybb var user_book1.ybb var user_book2.ybb \
             var user_book3.ybb var book.ybb"
        );
        // The reference's `.db` spellings are not offered, so they are rejected
        // like any other unlisted value and the prior value stands.
        assert!(matches!(
            s.set_value("BookFile", "user_book1.db"),
            Err(OptionError::InvalidComboChoice { .. })
        ));
        // `var` separates the choices; it is not one.
        assert!(matches!(
            s.set_value("BookFile", "var"),
            Err(OptionError::InvalidComboChoice { .. })
        ));
        // An empty value is turned away before the choice list is consulted.
        assert_eq!(s.set_value("BookFile", ""), Err(OptionError::EmptyValue));
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
        for rejected in [
            "not-a-number",
            "1.5",
            "24x",
            "24 ",
            "0x10",
            "+",
            "  ",
            "+-1",
            "99999999999999999999",
        ] {
            let err = s.set_value("USI_Hash", rejected).unwrap_err();
            assert!(
                matches!(err, OptionError::NotAnInteger(_)),
                "`{rejected}`: {err:?}"
            );
        }
        assert_eq!(s.set_value("USI_Hash", ""), Err(OptionError::EmptyValue));
        assert_eq!(s.spin("USI_Hash"), 1024);
    }

    #[test]
    fn set_spin_reads_numbers_as_strtoll_and_keeps_the_given_text() {
        let mut s = OptionStore::new();
        for (text, value) in [("+24", 24), (" 25", 25), ("\t\x0B26", 26), ("-0", 0)] {
            s.set_value("BookMoves", text).unwrap();
            assert_eq!(s.spin("BookMoves"), value);
            assert_eq!(s.value_text("BookMoves"), Some(text));
        }
        // The value ends at the first NUL, as a C string does.
        s.set_value("BookMoves", "27\0junk").unwrap();
        assert_eq!(s.spin("BookMoves"), 27);
        s.set_value("BookMoves", "\0").unwrap();
        assert_eq!(s.spin("BookMoves"), 0);
    }

    #[test]
    fn set_string_reads_the_empty_marker_as_empty() {
        let mut s = OptionStore::new();
        s.set_value("BookDir", "<empty>").unwrap();
        assert_eq!(s.text("BookDir"), "");
        assert_eq!(s.value_text("BookDir"), Some(""));
        s.set_value("BookDir", "").unwrap();
        assert_eq!(s.text("BookDir"), "");
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
    fn value_text_renders_each_type() {
        let s = OptionStore::new();
        assert_eq!(s.value_text("FV_SCALE"), Some("16"));
        assert_eq!(s.value_text("USI_Ponder"), Some("false"));
        assert_eq!(s.value_text("EvalDir"), Some("eval"));
        assert_eq!(s.value_text("EnteringKingRule"), Some("CSARule27"));
        assert_eq!(s.value_text("nope"), None);
    }

    #[test]
    fn is_string_is_case_insensitive() {
        let s = OptionStore::new();
        assert!(s.is_string("evaldir"));
        assert!(!s.is_string("FV_SCALE"));
        assert!(!s.is_string("nope"));
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
    fn set_value_matches_the_name_case_insensitively() {
        let mut s = OptionStore::new();
        s.set_value("usi_hash", "16").unwrap();
        assert_eq!(s.spin("USI_Hash"), 16);
        s.set_value("THREADS", "1").unwrap();
        assert_eq!(s.spin("Threads"), 1);
        assert_eq!(s.value_text("Threads"), Some("1"));
        assert_eq!(
            s.set_value("USI_Hashh", "16").unwrap_err(),
            OptionError::UnknownOption
        );
        assert_eq!(
            s.set_value("USI Hash", "16").unwrap_err(),
            OptionError::UnknownOption
        );
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
