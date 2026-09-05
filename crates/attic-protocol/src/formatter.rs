use std::io::{self, Write};

use crate::options::OptionDecl;

pub struct Formatter<'w, W: Write + ?Sized> {
    writer: &'w mut W,
}

impl<'w, W: Write + ?Sized> Formatter<'w, W> {
    pub fn new(writer: &'w mut W) -> Self {
        Self { writer }
    }

    pub fn id_name(&mut self, name: &str) -> io::Result<()> {
        self.line(format_args!("id name {name}"))
    }

    pub fn id_author(&mut self, author: &str) -> io::Result<()> {
        self.line(format_args!("id author {author}"))
    }

    pub fn option_decl(&mut self, decl: &OptionDecl) -> io::Result<()> {
        match decl {
            OptionDecl::Spin {
                name,
                default,
                min,
                max,
            } => self.line(format_args!(
                "option name {name} type spin default {default} min {min} max {max}"
            )),
            OptionDecl::String { name, default } => self.line(format_args!(
                "option name {name} type string default {default}"
            )),
            OptionDecl::Check { name, default } => self.line(format_args!(
                "option name {name} type check default {default}"
            )),
            OptionDecl::Combo {
                name,
                default,
                choices,
            } => {
                // `option name X type combo default D var A var B ...`
                let mut body = format!("option name {name} type combo default {default}");
                for choice in *choices {
                    body.push_str(" var ");
                    body.push_str(choice);
                }
                self.line(format_args!("{body}"))
            }
        }
    }

    pub fn usiok(&mut self) -> io::Result<()> {
        self.line(format_args!("usiok"))
    }

    pub fn readyok(&mut self) -> io::Result<()> {
        self.line(format_args!("readyok"))
    }

    pub fn info_string(&mut self, msg: &str) -> io::Result<()> {
        self.line(format_args!("info string {msg}"))
    }

    /// Emit a generic `info <body>` line; the caller composes everything after
    /// the `info ` keyword.
    pub fn info(&mut self, body: &str) -> io::Result<()> {
        self.line(format_args!("info {body}"))
    }

    pub fn bestmove(&mut self, move_str: &str) -> io::Result<()> {
        self.line(format_args!("bestmove {move_str}"))
    }

    /// Emit a verbatim line with no USI keyword prefix — only the
    /// option-override diagnostics the reference prints to raw `std::cout`
    /// (`usioption.cpp`).
    pub fn raw_line(&mut self, text: &str) -> io::Result<()> {
        self.line(format_args!("{text}"))
    }

    fn line(&mut self, args: std::fmt::Arguments<'_>) -> io::Result<()> {
        self.writer.write_fmt(args)?;
        self.writer.write_all(b"\n")?;
        self.writer.flush()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn captured<F>(f: F) -> String
    where
        F: FnOnce(&mut Formatter<'_, Vec<u8>>),
    {
        let mut buf: Vec<u8> = Vec::new();
        {
            let mut fmtr = Formatter::new(&mut buf);
            f(&mut fmtr);
        }
        String::from_utf8(buf).expect("utf-8")
    }

    #[test]
    fn id_name_emits_one_line() {
        let s = captured(|f| f.id_name("Attic 9.70git").unwrap());
        assert_eq!(s, "id name Attic 9.70git\n");
    }

    #[test]
    fn id_author_emits_one_line() {
        let s = captured(|f| f.id_author("Kei Ishida <ishida.kei@gmail.com>").unwrap());
        assert_eq!(s, "id author Kei Ishida <ishida.kei@gmail.com>\n");
    }

    #[test]
    fn option_spin_decl_format() {
        let decl = OptionDecl::Spin {
            name: "USI_Hash",
            default: 1024,
            min: 1,
            max: 33_554_432,
        };
        let s = captured(|f| f.option_decl(&decl).unwrap());
        assert_eq!(
            s,
            "option name USI_Hash type spin default 1024 min 1 max 33554432\n"
        );
    }

    #[test]
    fn option_string_decl_format() {
        let decl = OptionDecl::String {
            name: "EvalDir",
            default: "eval",
        };
        let s = captured(|f| f.option_decl(&decl).unwrap());
        assert_eq!(s, "option name EvalDir type string default eval\n");
    }

    #[test]
    fn option_check_decl_format() {
        let decl = OptionDecl::Check {
            name: "USI_OwnBook",
            default: true,
        };
        let s = captured(|f| f.option_decl(&decl).unwrap());
        assert_eq!(s, "option name USI_OwnBook type check default true\n");
    }

    #[test]
    fn option_combo_decl_format() {
        let decl = OptionDecl::Combo {
            name: "BookFile",
            default: "no_book",
            choices: &["no_book", "standard_book.db", "book.bin"],
        };
        let s = captured(|f| f.option_decl(&decl).unwrap());
        assert_eq!(
            s,
            "option name BookFile type combo default no_book var no_book var standard_book.db var book.bin\n"
        );
    }

    #[test]
    fn usiok_and_readyok() {
        assert_eq!(captured(|f| f.usiok().unwrap()), "usiok\n");
        assert_eq!(captured(|f| f.readyok().unwrap()), "readyok\n");
    }

    #[test]
    fn info_string_format() {
        let s = captured(|f| f.info_string("unknown command: foo").unwrap());
        assert_eq!(s, "info string unknown command: foo\n");
    }

    #[test]
    fn info_body_format() {
        let s = captured(|f| f.info("depth 1 score cp 12 nodes 30 pv 7g7f").unwrap());
        assert_eq!(s, "info depth 1 score cp 12 nodes 30 pv 7g7f\n");
    }

    #[test]
    fn bestmove_format() {
        assert_eq!(captured(|f| f.bestmove("7g7f").unwrap()), "bestmove 7g7f\n");
        assert_eq!(
            captured(|f| f.bestmove("resign").unwrap()),
            "bestmove resign\n"
        );
    }
}
