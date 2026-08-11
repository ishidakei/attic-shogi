//! Evaluation-layer NNUE support for the SFNN-1536 architecture.
//!
//! This crate is the Evaluation layer: it may import State and Storage, never
//! Protocol or Search. It covers loading and validating the `nn.bin` network
//! file, feature extraction, the accumulator, the forward pass, and the SIMD
//! kernels.
//!
//! The parsing logic is ported from a read-only Rust NNUE reference
//! implementation (verified bit-identical against the reference C++
//! engine), adapted to this workspace's std-only conventions.

mod aligned;
mod features;
mod loader;
mod network;
mod simd;
mod transformer;
mod types;

pub use aligned::Aligned64;
pub use features::{
    FEATURE_DIMENSION, FeatureIndex, MAX_ACTIVE_FEATURES, MoveDelta, PerspectiveDelta,
    active_features, active_features_both, requires_full_refresh,
};
pub use loader::{load_network, load_network_with_warnings};
pub use network::{
    FV_SCALE_DEFAULT, evaluate, evaluate_with, fv_scale, layer_stack_index, set_fv_scale,
};
pub use simd::{Backend, active_backend};
pub use transformer::{Accumulator, FT_OUTPUT_DIMS};
pub use types::{
    FC_0_INPUT_DIMS, FC_0_OUTPUT_DIMS, FC_0_PADDED_INPUT_DIMS, FC_1_INPUT_DIMS, FC_1_OUTPUT_DIMS,
    FC_1_PADDED_INPUT_DIMS, FC_2_INPUT_DIMS, FC_2_OUTPUT_DIMS, FC_2_PADDED_INPUT_DIMS, HIDDEN_SIZE,
    HIDDEN1_DIMS, HIDDEN2_DIMS, LAYER_STACKS, NUM_FEATURES, NetDims, NetHeader, NetworkStack,
    NnueError, NnueNetwork, NnueNetworkBuilder,
};
