//! Parity test: static NNUE evaluation must match the reference engine
//! **exactly** on every `tests/fixtures/eval/*.json` fixture.
//!
//! The network file is staged locally and never committed, so when it is absent
//! the test prints a notice and passes and the default `cargo test` run stays
//! green everywhere.

use std::path::{Path, PathBuf};

use attic_eval::{Backend, NnueNetwork, active_backend, evaluate, load_network};
use attic_state::{Position, parse_sfen, parse_usi_move};

const TOLERANCE: i32 = 0;

fn workspace_relative(rel: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join(rel)
}

fn nn_bin_path() -> PathBuf {
    workspace_relative("eval/nn.bin")
}

fn fixtures_dir() -> PathBuf {
    workspace_relative("tests/fixtures/eval")
}

/// An SFEN, an optional `moves` prefix, and the expected evaluation.
struct EvalFixture {
    name: String,
    sfen: String,
    moves: Vec<String>,
    eval: i32,
}

fn load_fixtures(dir: &Path) -> Vec<EvalFixture> {
    let mut entries: Vec<PathBuf> = std::fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("cannot read fixtures dir {}: {e}", dir.display()))
        .map(|e| e.expect("dir entry").path())
        .filter(|p| p.extension().is_some_and(|ext| ext == "json"))
        .collect();
    entries.sort();

    entries
        .iter()
        .map(|path| {
            let text = std::fs::read_to_string(path)
                .unwrap_or_else(|e| panic!("cannot read fixture {}: {e}", path.display()));
            let json: serde_json::Value = serde_json::from_str(&text)
                .unwrap_or_else(|e| panic!("fixture {} is not valid JSON: {e}", path.display()));

            let name = path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("<unknown>")
                .to_string();
            let sfen = json["sfen"]
                .as_str()
                .unwrap_or_else(|| panic!("fixture {name} missing string `sfen`"))
                .to_string();
            let eval = json["eval"]
                .as_i64()
                .unwrap_or_else(|| panic!("fixture {name} missing integer `eval`"))
                as i32;
            let moves = match json.get("moves") {
                None | Some(serde_json::Value::Null) => Vec::new(),
                Some(serde_json::Value::Array(arr)) => arr
                    .iter()
                    .map(|m| {
                        m.as_str()
                            .unwrap_or_else(|| panic!("fixture {name} has a non-string move"))
                            .to_string()
                    })
                    .collect(),
                Some(other) => panic!("fixture {name} `moves` is not an array: {other}"),
            };

            EvalFixture {
                name,
                sfen,
                moves,
                eval,
            }
        })
        .collect()
}

/// Parse the SFEN and play the `moves` prefix.
fn position_for(fixture: &EvalFixture) -> Position {
    let mut pos = parse_sfen(&fixture.sfen).unwrap_or_else(|e| {
        panic!(
            "fixture {}: bad sfen `{}`: {e:?}",
            fixture.name, fixture.sfen
        )
    });
    for mv in &fixture.moves {
        let parsed = parse_usi_move(mv, &pos)
            .unwrap_or_else(|e| panic!("fixture {}: bad move `{mv}`: {e:?}", fixture.name));
        pos.do_move(parsed);
    }
    pos
}

/// Backend selection is a compile-time decision, so a VNNI-capable CPU running
/// a scalar build means the build was not made for its host. On a CPU without
/// VNNI this only logs the scalar path.
fn assert_simd_path_selected() {
    #[cfg(target_arch = "x86_64")]
    {
        let has_vnni = std::arch::is_x86_feature_detected!("avx512f")
            && std::arch::is_x86_feature_detected!("avx512bw")
            && std::arch::is_x86_feature_detected!("avx512vnni");
        if has_vnni {
            assert_eq!(
                active_backend(),
                Backend::Avx512Vnni,
                "CPU reports AVX-512 VNNI but this build compiled the scalar \
                 backend — check that `-C target-cpu=native` is in effect",
            );
        } else {
            eprintln!(
                "eval parity running on the scalar backend ({:?}): CPU lacks AVX-512 VNNI",
                active_backend()
            );
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    eprintln!(
        "eval parity running on the scalar backend ({:?}): non-x86_64 target",
        active_backend()
    );
}

#[test]
#[cfg_attr(miri, ignore)]
fn eval_fixtures_match_reference_exactly() {
    let nn_bin = nn_bin_path();
    if !nn_bin.exists() {
        eprintln!(
            "skipping eval_fixtures_match_reference_exactly: {} is not present (obtained out-of-band)",
            nn_bin.display()
        );
        return;
    }

    let net: NnueNetwork = load_network(&nn_bin).expect("real nn.bin should load and validate");

    // Without this, the test could silently be proving only the scalar path
    // matches the reference.
    assert_simd_path_selected();

    let fixtures = load_fixtures(&fixtures_dir());
    assert!(!fixtures.is_empty(), "no eval fixtures found");

    let mut mismatches = Vec::new();
    for fixture in &fixtures {
        let pos = position_for(fixture);
        let actual = evaluate(&net, &pos);
        let diff = (actual - fixture.eval).abs();
        if diff > TOLERANCE {
            mismatches.push(format!(
                "  {}: expected {}, actual {} (diff {})",
                fixture.name, fixture.eval, actual, diff
            ));
        }
    }

    assert!(
        mismatches.is_empty(),
        "eval parity mismatch (TOLERANCE = {TOLERANCE}):\n{}",
        mismatches.join("\n"),
    );
}
