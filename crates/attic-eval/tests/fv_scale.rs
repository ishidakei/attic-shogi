//! Runtime `FV_SCALE` test: the final score is the raw network output divided
//! by the live scale, so changing it at runtime rescales the eval exactly.
//!
//! This is the only test in its binary, so the process-global scale it toggles
//! never races another test; it is restored on the way out regardless.
//!
//! The network file is staged locally and never committed, so when it is absent
//! the test prints a notice and passes.

use std::path::PathBuf;

use attic_eval::{FV_SCALE_DEFAULT, NnueNetwork, evaluate, fv_scale, load_network, set_fv_scale};
use attic_state::{Position, parse_sfen, parse_usi_move};

fn workspace_relative(rel: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join(rel)
}

fn position_for(sfen: &str, moves: &[String]) -> Position {
    let mut pos = parse_sfen(sfen).unwrap_or_else(|e| panic!("bad sfen `{sfen}`: {e:?}"));
    for mv in moves {
        let parsed = parse_usi_move(mv, &pos).unwrap_or_else(|e| panic!("bad move `{mv}`: {e:?}"));
        pos.do_move(parsed);
    }
    pos
}

/// The highest-magnitude fixture's `(sfen, moves)`, so that the division under
/// test runs on a value well above its divisor.
fn richest_fixture() -> (String, Vec<String>) {
    let dir = workspace_relative("tests/fixtures/eval");
    let mut best: Option<(i64, String, Vec<String>)> = None;
    for entry in std::fs::read_dir(&dir).expect("read fixtures dir") {
        let path = entry.expect("dir entry").path();
        if path.extension().is_none_or(|e| e != "json") {
            continue;
        }
        let text = std::fs::read_to_string(&path).expect("read fixture");
        let json: serde_json::Value = serde_json::from_str(&text).expect("parse fixture");
        let Some(sfen) = json["sfen"].as_str() else {
            continue;
        };
        let eval = json["eval"].as_i64().unwrap_or(0);
        let moves = match json.get("moves") {
            Some(serde_json::Value::Array(a)) => a
                .iter()
                .filter_map(|m| m.as_str().map(str::to_string))
                .collect(),
            _ => Vec::new(),
        };
        if best.as_ref().is_none_or(|(b, ..)| eval.abs() > b.abs()) {
            best = Some((eval, sfen.to_string(), moves));
        }
    }
    let (_, sfen, moves) = best.expect("at least one eval fixture");
    (sfen, moves)
}

#[test]
#[cfg_attr(miri, ignore)]
fn fv_scale_24_divides_raw_output_by_24() {
    let nn_bin = workspace_relative("eval/nn.bin");
    if !nn_bin.exists() {
        eprintln!(
            "skipping fv_scale_24_divides_raw_output_by_24: {} is not present (obtained out-of-band)",
            nn_bin.display()
        );
        return;
    }
    let net: NnueNetwork = load_network(&nn_bin).expect("real nn.bin should load and validate");

    let (sfen, moves) = richest_fixture();
    let pos = position_for(&sfen, &moves);

    // Evaluating at scale 1 recovers the raw network output.
    set_fv_scale(1);
    let raw = evaluate(&net, &pos);
    assert!(
        raw.abs() > 24,
        "fixture output {raw} too small to exercise a /24 division"
    );

    // The default scale reconfirms `raw` is the true numerator.
    set_fv_scale(FV_SCALE_DEFAULT);
    assert_eq!(evaluate(&net, &pos), raw / FV_SCALE_DEFAULT);

    set_fv_scale(24);
    assert_eq!(fv_scale(), 24);
    assert_eq!(
        evaluate(&net, &pos),
        raw / 24,
        "FV_SCALE 24 must divide the raw network output by 24"
    );

    set_fv_scale(FV_SCALE_DEFAULT);
}
