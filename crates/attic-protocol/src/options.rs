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
/// upper bound of the `Threads` spin option (`engine.cpp`). The `4 · cores`
/// term only exceeds the 1024 floor on machines with more than 256 hardware
/// threads; on everything else it is exactly 1024, matching the pinned floor.
fn max_threads() -> i64 {
    let cores = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1) as i64;
    std::cmp::max(1024, 4 * cores)
}

// Reference defaults pinned to upstream YaneuraOu/ as of the current submodule
// pin (yaneuraou-engine, NNUE eval, 64-bit, non-Stockfish, non-EMSCRIPTEN).
//
//   USI_Hash : yaneuraou-search.cpp  -> default 1024, min 1, max MaxHashMB
//   Threads  : engine.cpp        -> default 4,    min 1, max MaxThreads
//   MultiPV  : yaneuraou-search.cpp  -> default 1,    min 1, max MAX_MOVES (=600)
//   EvalDir  : eval/nnue/evaluate_nnue.cpp -> default "eval"
//
// The book option surface is ported verbatim from `BookMoveSelector::add_options`
// (`source/book/book.cpp`) — names, defaults, and
// ranges — with TWO deliberate divergences, both on `BookFile` and both flagged
// inline: its default is `no_book` in BOTH profiles (the pin defaults to
// `standard_book.db` under V1 and `user_book1.db` under V2), so this engine is
// bookless unless a book is explicitly selected; and its choice list offers the
// `.ybb` spellings of the pin's stems rather than the pin's `.db` / `.bin` ones
// (see [`BOOK_FILE_CHOICES`]). `BookPvMoves`' max is `MAX_PLY` (246,
// `config.h`/`types.h`).
//
// The book group is profile-dependent: `engine_option_profile.txt` selects V1
// (the historical surface, the default) or V2 (the black/white-split filters).
// See [`crate::option_profile`] and the per-option comments below.
//
// 64-bit MaxHashMB = 33554432 (engine.h); MaxThreads is dynamic in upstream
// (`max(1024, 4*cores)`, see [`max_threads`]). The declaration list is therefore
// built per store rather than being a compile-time constant.

/// `MAX_PLY` (`types.h`), the upper bound of `BookPvMoves`.
const MAX_PLY: i64 = 246;

/// The `EnteringKingRule` combo choice list — `EKR_STRINGS` in the reference's
/// exact order (`types.cpp`). Mirrors [`EnteringKingRule::STRINGS`]; kept
/// as a standalone `&'static` slice so the option table stays a plain literal.
const ENTERING_KING_RULE_CHOICES: &[&str] = &[
    "NoEnteringKing",
    "CSARule24",
    "CSARule24H",
    "CSARule27",
    "CSARule27H",
    "TryRule",
];

/// The `BookFile` combo choice list.
///
/// DELIBERATE DIVERGENCE (the second one on this option; the first is the
/// `no_book` default at the declaration site below). The pin's list
/// (`book.cpp`) is `no_book` plus eight `.db` names plus `book.bin`:
///
/// ```text
/// no_book standard_book.db yaneura_book{1,2,3,4}.db user_book{1,2,3}.db book.bin
/// ```
///
/// This engine reads only `.ybb` books (a deliberate scope divergence from the
/// pin, see [`crate::driver`]), so every one of those nine names is a choice
/// that cannot
/// be selected in practice: a combo must not advertise values the engine cannot
/// use. The list therefore keeps the pin's stems and count, respelled with the
/// `.ybb` extension (`book.bin`'s stem becomes `book.ybb`).
///
/// The name-resolution machinery is unaffected: the `.db → .ybb` sibling
/// fallback (`resolve_book_filename_with_ybb_fallback`) and the case-insensitive
/// extension test are simply unreachable from the combo, because a combo rejects
/// any value outside this list (`OptionError::InvalidComboChoice`).
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

/// Build the declaration list for `book_options` — the reference
/// `add_options` chain, whose book group branches on
/// `OptionsMap::book_options_v2()` (`book.cpp`). Everything outside the
/// book group is profile-independent.
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
        // NNUE fixed-point output scale (`evaluate_nnue.cpp`): registered
        // adjacent to `EvalDir` in the reference `add_options`. Default 16, the
        // fixture-capture condition; Suisho-family nets recommend 24+. Threaded
        // to the eval's live `FV_SCALE` global at each `go` (see the driver).
        OptionDecl::Spin {
            name: "FV_SCALE",
            default: 16,
            min: 1,
            max: 128,
        },
    ];

    // --- Opening-book options (book.cpp), profile-dependent. ---
    decls.extend(book_declarations(v2));

    decls.extend([
        // --- Entering-king (nyugyoku) declaration rule (yaneuraou-search.cpp).
        // Registered with `EKR_STRINGS[EKR_27_POINT]` as its default. ---
        OptionDecl::Combo {
            name: "EnteringKingRule",
            default: "CSARule27",
            choices: ENTERING_KING_RULE_CHOICES,
        },
        // --- Drive / limit group. ---
        // `DepthLimit` / `NodesLimit`, verbatim from `engine.cpp`: the
        // per-`go` search-depth / node ceilings, `0` meaning unlimited. When a
        // `go` carries no explicit `depth` / `nodes` token these seed
        // `limits.depth` / `limits.nodes` (`usi.cpp`); an explicit token
        // overwrites them for that `go`.
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
        // `MaxMovesToDraw`, verbatim from `yaneuraou-search.cpp`: the game
        // ply past which the search adjudicates an unconditional draw. A set value
        // of `0` is treated INTERNALLY as unlimited (remapped to 100000 in the
        // search); the option still reports `0`.
        OptionDecl::Spin {
            name: "MaxMovesToDraw",
            default: 0,
            min: 0,
            max: 100_000,
        },
        // --- MultiPV / PV-output group. ---
        // `PvInterval` / `ConsiderationMode` / `OutputFailLHPV`, verbatim from
        // `yaneuraou-search.cpp`: the PV-output throttle interval [ms]
        // (`0` = never suppress), the consideration mode (collect each PV from
        // the transposition table), and whether to emit a PV on a fail-high/low.
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
        // --- Behavior options group. ---
        // `DrawValueBlack` / `DrawValueWhite`, verbatim from
        // `yaneuraou-search.cpp`: the per-color draw score in centipawns
        // (from each color's own perspective). Consumed per-`go` from the root
        // side to move (`yaneuraou-search.cpp`).
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
        // `ResignValue`, verbatim from `yaneuraou-search.cpp`: after a real
        // search, if the centipawn-normalized best score `<= -ResignValue`, reply
        // `bestmove resign`. The default 99999 is effectively unreachable.
        OptionDecl::Spin {
            name: "ResignValue",
            default: 99999,
            min: 0,
            max: 99999,
        },
        // `GenerateAllLegalMoves`, verbatim from `yaneuraou-search.cpp`:
        // when true the search also considers the non-promoting moves the default
        // generator suppresses (pawn/lance/knight non-promotions etc.).
        OptionDecl::Check {
            name: "GenerateAllLegalMoves",
            default: false,
        },
        // --- Time-management group (timeman.cpp, engine.cpp). ---
        // `NetworkDelay` / `NetworkDelay2`, verbatim from `timeman.cpp`: the
        // average [ms] and worst-case (time-forfeit) [ms] GUI round-trip margins.
        // The search subtracts these from the clock so it never overruns. They
        // are kept as two separate margins, as the pin does, rather than
        // collapsed into one `MOVE_OVERHEAD` constant.
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
        // `MinimumThinkingTime` [ms] (`timeman.cpp`): the floor on a move's
        // optimum time before the network delay is subtracted.
        OptionDecl::Spin {
            name: "MinimumThinkingTime",
            default: 2000,
            min: 1,
            max: 100000,
        },
        // `SlowMover` (`timeman.cpp`): a percentage multiplier on the optimum
        // time (200 ⇒ think twice as long); an opening-emphasis / early-move dial.
        OptionDecl::Spin {
            name: "SlowMover",
            default: 100,
            min: 1,
            max: 1000,
        },
        // `RoundUpToFullSecond` (`timeman.cpp`): use the clock right up to each
        // whole second (byoyomi / per-second time controls) rather than leaving
        // sub-second slack.
        OptionDecl::Check {
            name: "RoundUpToFullSecond",
            default: true,
        },
        // `NumaPolicy` (`engine.cpp`): how the engine maps the machine to
        // logical NUMA nodes and whether it binds worker threads. `auto` /
        // `system` respect the process affinity; `hardware` ignores it; `none`
        // disables binding (single node); any other value is a custom
        // `':'`-separated node string. Registered immediately BEFORE `USI_Ponder`,
        // matching the pin's contiguous NumaPolicy → USI_Ponder → Stochastic_Ponder
        // order.
        OptionDecl::String {
            name: "NumaPolicy",
            default: "auto",
        },
        // `USI_Ponder` / `Stochastic_Ponder` (`engine.cpp`). At the
        // option layer these only feed the `optimumTime` bonus
        // (`timeman.cpp`) and are echoed in the USI option list.
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
/// (`book.cpp`). `v2` is `OptionsMap::book_options_v2()`: it drops
/// `NarrowBook` / `ConsiderBookMoveCount`, swaps `BookEvalDiff` and
/// `BookDepthLimit` for their black/white-split counterparts, and shifts the
/// large-book defaults (`BookMoves` 200, `BookOnTheFly` / `IgnoreBookPly` on).
fn book_declarations(v2: bool) -> Vec<OptionDecl> {
    let mut decls = vec![OptionDecl::Check {
        name: "USI_OwnBook",
        default: true,
    }];

    // `NarrowBook` exists only under V1 (`book.cpp`); V2 behaves as if
    // it were permanently false.
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
            // DELIBERATE DIVERGENCE #1: the pin defaults to `standard_book.db`
            // (V1) / `user_book1.db` (V2); this engine is bookless in BOTH
            // profiles unless `BookFile` is set explicitly.
            // DELIBERATE DIVERGENCE #2: the choice list offers the `.ybb`
            // spellings of the pin's stems, not the pin's `.db` / `.bin` names —
            // see [`BOOK_FILE_CHOICES`].
            name: "BookFile",
            default: "no_book",
            choices: BOOK_FILE_CHOICES,
        },
        OptionDecl::String {
            name: "BookDir",
            default: "book",
        },
    ]);

    // The eval-diff filter: one option under V1, split per root side to move
    // under V2 (`book.cpp`).
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

    // The depth floor: likewise split per root side to move under V2
    // (`book.cpp`).
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

    // V2 targets huge books, so streaming reads are on by default
    // (`book.cpp`).
    decls.push(OptionDecl::Check {
        name: "BookOnTheFly",
        default: v2,
    });

    // `ConsiderBookMoveCount` exists only under V1 (`book.cpp`).
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
        // Likewise on by default under V2 (`book.cpp`).
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
    /// `Threads` upper bound is computed at runtime (see [`max_threads`]).
    decls: Vec<OptionDecl>,
    values: BTreeMap<&'static str, OptionValue>,
    /// Options locked by an `engine_options.txt` / `eval_options.txt` override
    /// (`OptionsMap::build_option` sets `Option::fixed`, `usioption.cpp`).
    /// A fixed option ignores every later [`set_value`] — the reference
    /// `Option::operator=` cancels the assignment silently (`usioption.cpp`).
    fixed: BTreeSet<&'static str>,
    /// The book-option profile this store was built with (the reference
    /// `OptionsMap::book_options_version`, `usioption.h`). The probe reads it
    /// back to pick the side-to-move-dependent option names.
    book_options: BookOptionsVersion,
}

impl OptionStore {
    /// A store with the default (V1) book-option surface — the reference's
    /// behaviour when no `engine_option_profile.txt` is present.
    pub fn new() -> Self {
        Self::with_book_options(BookOptionsVersion::default())
    }

    /// A store whose book-option group follows `book_options`, as selected by
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

    /// Whether the book options were registered under the V2 profile — the
    /// reference `OptionsMap::book_options_v2()` (`usioption.h`).
    pub fn book_options_v2(&self) -> bool {
        self.book_options.is_v2()
    }

    pub fn get(&self, name: &str) -> Option<&OptionValue> {
        self.decls
            .iter()
            .find(|d| d.name() == name)
            .and_then(|d| self.values.get(d.name()))
    }

    /// The current `Threads` value as a pool size (always ≥ 1 by the option's
    /// declared minimum). Used by the driver to size its worker pool.
    pub fn threads(&self) -> usize {
        match self.get("Threads") {
            Some(OptionValue::Spin(n)) => (*n).max(1) as usize,
            // Threads is a declared spin, so this is unreachable; fall back to a
            // single worker for totality.
            _ => 1,
        }
    }

    pub fn set_value(&mut self, name: &str, value: &str) -> Result<(), OptionError> {
        let decl = self
            .decls
            .iter()
            .find(|d| d.name() == name)
            .ok_or(OptionError::UnknownOption)?;
        // A fixed (overridden) option silently ignores the assignment, mirroring
        // the reference `Option::operator=` cancel (`usioption.cpp`): no
        // mutation, no error, no output — the value stays what the override set.
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

    /// A declared spin's current value (`0` if the name is not a spin — never
    /// happens for a declared name; the fallback keeps callers total).
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
    /// case-insensitively like the reference `OptionsMap` (keyed by
    /// `CaseInsensitiveLess`, `usioption.h`). `None` if no such option.
    pub fn canonical_name(&self, name: &str) -> Option<&'static str> {
        self.decls
            .iter()
            .find(|d| d.name().eq_ignore_ascii_case(name))
            .map(|d| d.name())
    }

    /// Lock an option against further [`set_value`] mutation — the reference
    /// `Option::fixed` an override sets (`usioption.cpp`). Idempotent.
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
        // Book option surface: BookFile is a combo defaulting to `no_book`
        // (deliberate divergence from the pin's `standard_book.db`).
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
        // Entering-king rule: combo defaulting to the CSA 27-point rule.
        assert_eq!(
            s.get("EnteringKingRule"),
            Some(&OptionValue::Combo("CSARule27".to_string()))
        );
        // Drive / limit group: all default to 0 (unlimited).
        assert_eq!(s.get("DepthLimit"), Some(&OptionValue::Spin(0)));
        assert_eq!(s.get("NodesLimit"), Some(&OptionValue::Spin(0)));
        assert_eq!(s.get("MaxMovesToDraw"), Some(&OptionValue::Spin(0)));
        // MultiPV / PV-output group.
        assert_eq!(s.get("PvInterval"), Some(&OptionValue::Spin(300)));
        assert_eq!(s.get("ConsiderationMode"), Some(&OptionValue::Check(false)));
        assert_eq!(s.get("OutputFailLHPV"), Some(&OptionValue::Check(true)));
        // Behavior options group.
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

        // The split filters replace their single-option V1 counterparts.
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

        // Large-book defaults.
        assert_eq!(s.get("BookMoves"), Some(&OptionValue::Spin(200)));
        assert_eq!(s.get("BookOnTheFly"), Some(&OptionValue::Check(true)));
        assert_eq!(s.get("IgnoreBookPly"), Some(&OptionValue::Check(true)));

        // Unchanged in both profiles — including the `no_book` divergence.
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

        // Everything outside the book group is profile-independent.
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
        // The pin's `.db` spellings are not offered, so they are rejected like
        // any other unlisted value — and the store keeps its prior value.
        assert!(matches!(
            s.set_value("BookFile", "user_book1.db"),
            Err(OptionError::InvalidComboChoice(_))
        ));
        assert_eq!(s.text("BookFile"), "user_book1.ybb");
    }

    #[test]
    fn every_book_file_choice_carries_a_loadable_spelling() {
        // The divergence's whole point: apart from the `no_book` sentinel, every
        // advertised choice names a `.ybb` file, which is the only book format
        // this engine reads. (That each one actually loads end-to-end is proved
        // by tests/book_file_choices.rs.)
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
        // A subsequent set is silently ignored — no error, value unchanged
        // (mirrors the reference `Option::operator=` fixed cancel).
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
