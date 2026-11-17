//! Emits `cfg(target_cpu_native)` when this crate is compiled with
//! `-C target-cpu=native`.
//!
//! The SIMD backend is chosen from compile-time target features, so a test can
//! only expect a host with AVX-512 VNNI to get the SIMD build when the build
//! targets that host. Under a fixed baseline such as `x86-64-v3` the scalar
//! backend is the correct result on any CPU, and the host check has to stay
//! out of the build.

fn main() {
    println!("cargo::rustc-check-cfg=cfg(target_cpu_native)");
    println!("cargo::rerun-if-changed=build.rs");
    println!("cargo::rerun-if-env-changed=CARGO_ENCODED_RUSTFLAGS");

    let flags = std::env::var("CARGO_ENCODED_RUSTFLAGS").unwrap_or_default();
    if target_cpu(&flags) == Some("native") {
        println!("cargo::rustc-cfg=target_cpu_native");
    }
}

/// The `target-cpu` value rustc ends up with: the last one given, in either the
/// `-C target-cpu=…` or the `-Ctarget-cpu=…` spelling.
fn target_cpu(encoded: &str) -> Option<&str> {
    let mut args = encoded.split('\x1f').filter(|a| !a.is_empty());
    let mut cpu = None;
    while let Some(arg) = args.next() {
        let codegen = match arg {
            "-C" | "--codegen" => args.next(),
            _ => arg
                .strip_prefix("-C")
                .or_else(|| arg.strip_prefix("--codegen=")),
        };
        if let Some(value) = codegen.and_then(|c| c.strip_prefix("target-cpu=")) {
            cpu = Some(value);
        }
    }
    cpu
}
