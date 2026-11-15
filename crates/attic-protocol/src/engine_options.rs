//! Parser for `engine_options.txt` / `eval_options.txt` override lines, ported
//! from the reference `OptionsMap::build_option`.
//!
//! Only the pure line → [`OverrideLine`] parse lives here; applying the result
//! against the live option store is the driver's, which owns the output sink.

/// The parse of one override line.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OverrideLine {
    /// A blank line, or a comment starting with `#` or `//`.
    Skip,
    /// An override of `name` with `value`, from either the `<Name> <Value>` /
    /// `<Name>=<Value>` form or the `default` of the full
    /// `option name <N> type <T> default <V> [min .. max .. var ..]` form.
    /// `value` is `""` when the line names an option but no value.
    Override { name: String, value: String },
    /// A full-form line rejected as a whole: no `name` after `option`, no
    /// `type`, or a keyword other than `default`, `min`, `max` and `var`.
    InvalidDeclaration,
}

/// The characters a line and a value are trimmed of.
const TRIM: &[char] = &[' ', '\t', '\r', '\n'];

/// Parse one raw line into an [`OverrideLine`], mirroring the reference
/// `build_option`.
///
/// In the short form the name ends at the first `=`, space or tab, and the
/// value is the rest of the line, trimmed and with one leading `=` dropped, so
/// a value keeps its inner spaces and `=`s. In the full form the name runs
/// from `name` to `type`, and `is_string` tells whether the named option is a
/// string option, whose `default` takes the rest of the line rather than one
/// token.
pub fn parse_override_line(line: &str, is_string: impl Fn(&str) -> bool) -> OverrideLine {
    let text = line.trim_matches(TRIM);
    if text.is_empty() || text.starts_with('#') || text.starts_with("//") {
        return OverrideLine::Skip;
    }

    let name_end = text.find(['=', ' ', '\t']).unwrap_or(text.len());
    if &text[..name_end] != "option" {
        let name = text[..name_end].to_string();
        let mut value = text[name_end..].trim_matches(TRIM);
        if let Some(rest) = value.strip_prefix('=') {
            value = rest.trim_matches(TRIM);
        }
        return OverrideLine::Override {
            name,
            value: value.to_string(),
        };
    }

    let mut scanner = Scanner { rest: text };
    let _ = scanner.token(); // "option"
    if scanner.token() != Some("name") {
        return OverrideLine::InvalidDeclaration;
    }
    let mut name = String::new();
    let mut last = None;
    while let Some(t) = scanner.token() {
        last = Some(t);
        if t == "type" {
            break;
        }
        if !name.is_empty() {
            name.push(' ');
        }
        name.push_str(t);
    }
    // The type itself is not used: an override never changes an option's type.
    if last != Some("type") || scanner.token().is_none() {
        return OverrideLine::InvalidDeclaration;
    }

    let mut value = String::new();
    while let Some(t) = scanner.token() {
        match t {
            "default" => {
                if is_string(&name) {
                    value = scanner.rest.trim_matches(TRIM).to_string();
                    break;
                }
                // With no token left the value is kept, as the reference's
                // failed stream extraction leaves it.
                if let Some(v) = scanner.token() {
                    value = v.to_string();
                }
            }
            // An existing option's range and choices are not changed.
            "min" | "max" | "var" => {
                let _ = scanner.token();
            }
            _ => return OverrideLine::InvalidDeclaration,
        }
    }
    OverrideLine::Override { name, value }
}

/// A whitespace tokenizer that keeps the unread rest of the line, which a
/// string option's `default` takes whole.
struct Scanner<'a> {
    rest: &'a str,
}

impl<'a> Scanner<'a> {
    fn token(&mut self) -> Option<&'a str> {
        let start = self
            .rest
            .trim_start_matches(|c: char| c.is_ascii_whitespace());
        let end = start
            .find(|c: char| c.is_ascii_whitespace())
            .unwrap_or(start.len());
        self.rest = &start[end..];
        (end > 0).then(|| &start[..end])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(line: &str) -> OverrideLine {
        parse_override_line(line, |name| name == "EvalDir")
    }

    fn over(name: &str, value: &str) -> OverrideLine {
        OverrideLine::Override {
            name: name.to_string(),
            value: value.to_string(),
        }
    }

    #[test]
    fn blank_and_comment_lines_are_skipped() {
        assert_eq!(parse(""), OverrideLine::Skip);
        assert_eq!(parse("   \t\r"), OverrideLine::Skip);
        assert_eq!(parse("# FV_SCALE 24"), OverrideLine::Skip);
        assert_eq!(parse("  // FV_SCALE 24"), OverrideLine::Skip);
    }

    #[test]
    fn name_space_value() {
        assert_eq!(parse("FV_SCALE 24"), over("FV_SCALE", "24"));
        assert_eq!(parse("  FV_SCALE \t 24  \r"), over("FV_SCALE", "24"));
    }

    #[test]
    fn name_equals_value() {
        assert_eq!(parse("FV_SCALE=24"), over("FV_SCALE", "24"));
        assert_eq!(parse("FV_SCALE = 24"), over("FV_SCALE", "24"));
        assert_eq!(parse("FV_SCALE =24"), over("FV_SCALE", "24"));
    }

    #[test]
    fn value_keeps_inner_spaces() {
        assert_eq!(parse("EvalDir my eval dir"), over("EvalDir", "my eval dir"));
        assert_eq!(
            parse("EvalDir = my  eval\tdir "),
            over("EvalDir", "my  eval\tdir")
        );
    }

    #[test]
    fn value_keeps_equals_after_the_first() {
        assert_eq!(parse("EvalDir=a=b"), over("EvalDir", "a=b"));
        assert_eq!(parse("EvalDir==b"), over("EvalDir", "=b"));
        assert_eq!(parse("EvalDir a = b"), over("EvalDir", "a = b"));
    }

    #[test]
    fn name_without_value() {
        assert_eq!(parse("FV_SCALE"), over("FV_SCALE", ""));
        assert_eq!(parse("FV_SCALE ="), over("FV_SCALE", ""));
        assert_eq!(parse("="), over("", ""));
    }

    #[test]
    fn full_form_takes_default_as_value() {
        assert_eq!(
            parse("option name USI_Hash type spin default 256 min 1 max 1024"),
            over("USI_Hash", "256")
        );
        assert_eq!(
            parse("option name BookFile type combo default no_book var no_book var book.ybb"),
            over("BookFile", "no_book")
        );
    }

    #[test]
    fn full_form_string_default_takes_the_rest_of_the_line() {
        assert_eq!(
            parse("option name EvalDir type string default my eval = dir  "),
            over("EvalDir", "my eval = dir")
        );
    }

    #[test]
    fn full_form_name_may_have_spaces() {
        assert_eq!(
            parse("option name Two Words type spin default 3"),
            over("Two Words", "3")
        );
    }

    #[test]
    fn full_form_without_default_has_empty_value() {
        assert_eq!(
            parse("option name FV_SCALE type spin"),
            over("FV_SCALE", "")
        );
    }

    #[test]
    fn full_form_malformed_is_invalid() {
        for line in [
            "option",
            "option FV_SCALE type spin default 24",
            "option name FV_SCALE",
            "option name FV_SCALE type",
            "option name FV_SCALE type spin default 24 bogus",
        ] {
            assert_eq!(parse(line), OverrideLine::InvalidDeclaration, "{line}");
        }
    }
}
