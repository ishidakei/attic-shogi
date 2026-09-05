//! Load the real SFNN-1536 `nn.bin` when it is present.
//!
//! The network file is staged locally and never committed, so when it is absent
//! the test prints a notice and passes and the default `cargo test` run stays
//! green everywhere.

use std::path::PathBuf;

use attic_eval::{HIDDEN_SIZE, LAYER_STACKS, NUM_FEATURES, load_network};

/// Resolves the staged network path relative to the workspace root.
fn nn_bin_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../eval/nn.bin")
}

#[test]
#[cfg_attr(miri, ignore)]
fn loads_real_network_if_present() {
    let path = nn_bin_path();
    if !path.exists() {
        eprintln!(
            "skipping loads_real_network_if_present: {} is not present (obtained out-of-band)",
            path.display()
        );
        return;
    }

    let net = load_network(&path).expect("real nn.bin should load and validate");

    assert_eq!(net.stacks.len(), LAYER_STACKS, "layer-stack count");
    assert_eq!(net.ft_biases.len(), HIDDEN_SIZE, "ft bias count");
    assert_eq!(
        net.ft_weights.len(),
        HIDDEN_SIZE * NUM_FEATURES,
        "ft weight count"
    );

    // Every buffer must land on a 64-byte boundary for the SIMD kernels.
    let is_aligned = |ptr: *const u8| (ptr as usize).is_multiple_of(64);
    assert!(is_aligned(net.ft_biases.as_ptr() as *const u8));
    assert!(is_aligned(net.ft_weights.as_ptr() as *const u8));
    for (i, stack) in net.stacks.iter().enumerate() {
        assert!(
            is_aligned(stack.fc_0_biases.as_ptr() as *const u8),
            "stack[{i}].fc_0_biases not 64-byte aligned"
        );
        assert!(
            is_aligned(stack.fc_0_weights.as_ptr() as *const u8),
            "stack[{i}].fc_0_weights not 64-byte aligned"
        );
        assert!(
            is_aligned(stack.fc_1_biases.as_ptr() as *const u8),
            "stack[{i}].fc_1_biases not 64-byte aligned"
        );
        assert!(
            is_aligned(stack.fc_1_weights.as_ptr() as *const u8),
            "stack[{i}].fc_1_weights not 64-byte aligned"
        );
        assert!(
            is_aligned(stack.fc_2_biases.as_ptr() as *const u8),
            "stack[{i}].fc_2_biases not 64-byte aligned"
        );
        assert!(
            is_aligned(stack.fc_2_weights.as_ptr() as *const u8),
            "stack[{i}].fc_2_weights not 64-byte aligned"
        );
    }
}
