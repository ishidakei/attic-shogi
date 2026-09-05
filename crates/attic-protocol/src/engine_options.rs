//! Parser for `engine_options.txt` / `eval_options.txt` override lines, ported
//! from the reference `OptionsMap::build_option` (`usioption.cpp`).
//!
//! Only the pure line → [`OverrideLine`] parse lives here; applying the result
//! against the live option store is the driver's, which owns the output sink.

/// The parse of one override line. The reference first replaces every `'='`
/// with `' '`, then splits on whitespace; a leading `option` token selects the
/// full form.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OverrideLine {
    Empty,
    /// A `<Name> <Value>` override. `value` is `""` when the line names an
    /// option but no value.
    Plain {
        name: String,
        value: String,
    },
    /// The full `option name <N> type <T> default <V> [min .. max .. var ..]`
    /// form; `value` comes from the `default` token. `invalid_tokens` holds the
    /// tokens the reference scan loop did not recognise, which the driver
    /// reports without aborting the override.
    Full {
        name: String,
        value: String,
        invalid_tokens: Vec<String>,
    },
}

/// Parse one raw line into an [`OverrideLine`], mirroring the reference
/// `build_option` scan (`usioption.cpp`).
pub fn parse_override_line(line: &str) -> OverrideLine {
    let replaced: String = line.replace('=', " ");
    let mut tokens = replaced.split_whitespace();

    let first = match tokens.next() {
        None => return OverrideLine::Empty,
        Some(t) => t,
    };

    if first != "option" {
        let name = first.to_string();
        let value = tokens.next().unwrap_or("").to_string();
        return OverrideLine::Plain { name, value };
    }

    // The reference reads a keyword then its argument; an unrecognised keyword
    // is reported as an invalid command but does not abort the scan.
    let mut name = String::new();
    let mut value = String::new();
    let mut invalid_tokens = Vec::new();
    while let Some(tok) = tokens.next() {
        match tok {
            "name" => name = tokens.next().unwrap_or("").to_string(),
            "type" => {
                let _ = tokens.next();
            }
            "default" => value = tokens.next().unwrap_or("").to_string(),
            "min" | "max" => {
                let _ = tokens.next();
            }
            "var" => {
                let _ = tokens.next();
            }
            other => invalid_tokens.push(other.to_string()),
        }
    }
    OverrideLine::Full {
        name,
        value,
        invalid_tokens,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_line_is_empty() {
        assert_eq!(parse_override_line(""), OverrideLine::Empty);
        assert_eq!(parse_override_line("   "), OverrideLine::Empty);
        assert_eq!(parse_override_line("="), OverrideLine::Empty);
    }

    #[test]
    fn plain_name_value() {
        assert_eq!(
            parse_override_line("FV_SCALE 24"),
            OverrideLine::Plain {
                name: "FV_SCALE".to_string(),
                value: "24".to_string(),
            }
        );
    }

    #[test]
    fn name_equals_value_collapses_to_plain() {
        assert_eq!(
            parse_override_line("FV_SCALE=24"),
            OverrideLine::Plain {
                name: "FV_SCALE".to_string(),
                value: "24".to_string(),
            }
        );
        assert_eq!(
            parse_override_line("FV_SCALE = 24"),
            OverrideLine::Plain {
                name: "FV_SCALE".to_string(),
                value: "24".to_string(),
            }
        );
    }

    #[test]
    fn plain_name_without_value() {
        assert_eq!(
            parse_override_line("FV_SCALE"),
            OverrideLine::Plain {
                name: "FV_SCALE".to_string(),
                value: String::new(),
            }
        );
    }

    #[test]
    fn full_option_form_takes_default_as_value() {
        assert_eq!(
            parse_override_line("option name USI_Hash type spin default 256 min 1 max 1024"),
            OverrideLine::Full {
                name: "USI_Hash".to_string(),
                value: "256".to_string(),
                invalid_tokens: Vec::new(),
            }
        );
    }

    #[test]
    fn full_option_form_flags_unknown_trailing_tokens() {
        assert_eq!(
            parse_override_line("option name FV_SCALE type spin default 24 bogus"),
            OverrideLine::Full {
                name: "FV_SCALE".to_string(),
                value: "24".to_string(),
                invalid_tokens: vec!["bogus".to_string()],
            }
        );
    }
}
