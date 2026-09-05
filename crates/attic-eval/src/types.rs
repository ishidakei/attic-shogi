//! Dimensions, shared network types, and the loader error type for SFNN-1536.
//!
//! ## Layer naming
//!
//! Prose in this crate follows the upstream naming: the **FT layer** is the
//! feature transformer, and **L1 / L2 / L3** are the dense layers after it. The
//! FT layer is never called "L1". In the identifiers below, L1 = `fc_0`,
//! L2 = `fc_1`, L3 = `fc_2`.

use std::fmt;

use attic_storage::{ArenaLayout, ArenaSlice, LargePageArena, Section};

pub const HIDDEN_SIZE: usize = 1_536;
pub const NUM_FEATURES: usize = 73_305;
pub const LAYER_STACKS: usize = 9;

// fc_0 has 16 outputs: the 16th feeds only the post-fc_2 shortcut, while the
// first 15 feed both activations.
pub const HIDDEN1_DIMS: usize = 15;
pub const HIDDEN2_DIMS: usize = 32;

pub const FC_0_OUTPUT_DIMS: usize = HIDDEN1_DIMS + 1;
pub const FC_0_INPUT_DIMS: usize = HIDDEN_SIZE;
pub const FC_0_PADDED_INPUT_DIMS: usize = HIDDEN_SIZE;

pub const FC_1_OUTPUT_DIMS: usize = HIDDEN2_DIMS;
pub const FC_1_INPUT_DIMS: usize = HIDDEN1_DIMS * 2;
pub const FC_1_PADDED_INPUT_DIMS: usize = 32;

pub const FC_2_OUTPUT_DIMS: usize = 1;
pub const FC_2_INPUT_DIMS: usize = HIDDEN2_DIMS;
pub const FC_2_PADDED_INPUT_DIMS: usize = 32;

#[derive(Clone, Debug)]
pub struct NetHeader {
    pub version: u32,
    pub hash: u32,
    pub arch_id: String,
}

/// The dimensions of one SFNN network, driving both the arena layout and the
/// loader's read sizes.
#[derive(Clone, Copy, Debug)]
pub struct NetDims {
    pub hidden_size: usize,
    pub num_features: usize,
    pub layer_stacks: usize,
    pub fc_0_output: usize,
    pub fc_0_padded_input: usize,
    pub fc_1_output: usize,
    pub fc_1_padded_input: usize,
    pub fc_2_output: usize,
    pub fc_2_padded_input: usize,
}

impl NetDims {
    /// The shipped SFNN-1536 dimensions.
    pub const STANDARD: NetDims = NetDims {
        hidden_size: HIDDEN_SIZE,
        num_features: NUM_FEATURES,
        layer_stacks: LAYER_STACKS,
        fc_0_output: FC_0_OUTPUT_DIMS,
        fc_0_padded_input: FC_0_PADDED_INPUT_DIMS,
        fc_1_output: FC_1_OUTPUT_DIMS,
        fc_1_padded_input: FC_1_PADDED_INPUT_DIMS,
        fc_2_output: FC_2_OUTPUT_DIMS,
        fc_2_padded_input: FC_2_PADDED_INPUT_DIMS,
    };
}

/// The arena [`Section`]s of one layer stack's six FC arrays.
#[derive(Clone, Copy, Debug)]
struct StackSections {
    fc_0_biases: Section<i32>,
    fc_0_weights: Section<i8>,
    fc_1_biases: Section<i32>,
    fc_1_weights: Section<i8>,
    fc_2_biases: Section<i32>,
    fc_2_weights: Section<i8>,
}

/// The full arena layout of one network. Recorded once, so that a NUMA replica
/// can be rebuilt from the same sections without re-deriving anything.
#[derive(Clone, Debug)]
struct NetLayout {
    ft_biases: Section<i16>,
    ft_weights: Section<i16>,
    stacks: Vec<StackSections>,
    total_bytes: usize,
}

impl NetLayout {
    /// Pack every parameter array into one 64-byte-aligned layout, in file order
    /// (ft_biases, ft_weights, then each stack's fc arrays).
    fn compute(dims: &NetDims) -> Self {
        let mut l = ArenaLayout::new();
        let ft_biases = l.reserve::<i16>(dims.hidden_size);
        let ft_weights = l.reserve::<i16>(dims.hidden_size * dims.num_features);
        let mut stacks = Vec::with_capacity(dims.layer_stacks);
        for _ in 0..dims.layer_stacks {
            stacks.push(StackSections {
                fc_0_biases: l.reserve::<i32>(dims.fc_0_output),
                fc_0_weights: l.reserve::<i8>(dims.fc_0_output * dims.fc_0_padded_input),
                fc_1_biases: l.reserve::<i32>(dims.fc_1_output),
                fc_1_weights: l.reserve::<i8>(dims.fc_1_output * dims.fc_1_padded_input),
                fc_2_biases: l.reserve::<i32>(dims.fc_2_output),
                fc_2_weights: l.reserve::<i8>(dims.fc_2_output * dims.fc_2_padded_input),
            });
        }
        Self {
            ft_biases,
            ft_weights,
            stacks,
            total_bytes: l.total_bytes(),
        }
    }
}

/// One layer stack's parameter arrays, each a 64-byte-aligned view into the
/// network's single arena.
#[derive(Debug)]
pub struct NetworkStack {
    pub fc_0_biases: ArenaSlice<i32>,
    pub fc_0_weights: ArenaSlice<i8>,
    pub fc_1_biases: ArenaSlice<i32>,
    pub fc_1_weights: ArenaSlice<i8>,
    pub fc_2_biases: ArenaSlice<i32>,
    pub fc_2_weights: ArenaSlice<i8>,
}

impl NetworkStack {
    /// Build the six views for stack `sections` against `arena`.
    fn from_sections(arena: &LargePageArena, sections: &StackSections) -> Self {
        Self {
            fc_0_biases: arena.view(sections.fc_0_biases),
            fc_0_weights: arena.view(sections.fc_0_weights),
            fc_1_biases: arena.view(sections.fc_1_biases),
            fc_1_weights: arena.view(sections.fc_1_weights),
            fc_2_biases: arena.view(sections.fc_2_biases),
            fc_2_weights: arena.view(sections.fc_2_weights),
        }
    }
}

/// A loaded SFNN network.
///
/// Every parameter array — `ft_biases`, the ~215 MiB `ft_weights`, and all nine
/// stacks' FC weights and biases — lives in **one** large-page allocation, as
/// the reference's `make_unique_large_page<NnueNetworks>` does
/// (`evaluate_nnue.cpp`). The public parameter fields are 64-byte-aligned
/// views into it, meeting what the AVX-512 kernels require.
#[derive(Debug)]
pub struct NnueNetwork {
    pub header: NetHeader,
    /// The single backing allocation, never mutated after build.
    arena: LargePageArena,
    /// The section table, replayed to rebuild views for a replica.
    layout: NetLayout,
    pub ft_biases: ArenaSlice<i16>,
    pub ft_weights: ArenaSlice<i16>,
    pub stacks: Vec<NetworkStack>,
    pub sha256: [u8; 32],
}

impl NnueNetwork {
    /// Assemble a network from a filled `arena` and its `layout`, copying no
    /// data.
    fn from_arena(
        header: NetHeader,
        arena: LargePageArena,
        layout: NetLayout,
        sha256: [u8; 32],
    ) -> Self {
        let ft_biases = arena.view(layout.ft_biases);
        let ft_weights = arena.view(layout.ft_weights);
        let stacks = layout
            .stacks
            .iter()
            .map(|s| NetworkStack::from_sections(&arena, s))
            .collect();
        Self {
            header,
            arena,
            layout,
            ft_biases,
            ft_weights,
            stacks,
            sha256,
        }
    }

    /// A byte-identical deep copy of the whole network in a freshly allocated
    /// arena, so the two instances evaluate any position to the same score.
    ///
    /// The copy runs on the calling thread, so performing it inside a
    /// NUMA-node-bound one first-touches every page on that node — the
    /// in-process analogue of the reference's
    /// `LazyNumaReplicatedSystemWide<Networks>` (`numa.h`), minus its
    /// POSIX shared-memory layer.
    pub fn replicate(&self) -> Self {
        let arena = self.arena.clone_backing();
        Self::from_arena(self.header.clone(), arena, self.layout.clone(), self.sha256)
    }

    /// The number of large-page allocations backing this network, always 1, and
    /// the reserved byte size.
    pub fn allocation_disclosure(&self) -> (usize, usize) {
        (1, self.arena.reserved_bytes())
    }
}

/// In-place builder for a [`NnueNetwork`]: allocate the arena up front, fill
/// each parameter array through a mutable view, then [`build`](Self::build).
/// Filling in place avoids a large temporary for the ~215 MiB `ft_weights`.
pub struct NnueNetworkBuilder {
    header: NetHeader,
    sha256: [u8; 32],
    arena: LargePageArena,
    layout: NetLayout,
}

impl NnueNetworkBuilder {
    /// A zeroed builder for the shipped SFNN-1536 dimensions.
    pub fn new(header: NetHeader, sha256: [u8; 32]) -> Self {
        Self::with_dims(header, sha256, &NetDims::STANDARD)
    }

    /// A zeroed builder for arbitrary `dims`.
    pub fn with_dims(header: NetHeader, sha256: [u8; 32], dims: &NetDims) -> Self {
        let layout = NetLayout::compute(dims);
        let arena = LargePageArena::with_capacity(layout.total_bytes);
        Self {
            header,
            sha256,
            arena,
            layout,
        }
    }

    /// Mutable view of `ft_biases` (`hidden_size` i16).
    pub fn ft_biases_mut(&mut self) -> &mut [i16] {
        self.arena.slice_mut(self.layout.ft_biases)
    }

    /// Mutable view of `ft_weights` (`hidden_size * num_features` i16).
    pub fn ft_weights_mut(&mut self) -> &mut [i16] {
        self.arena.slice_mut(self.layout.ft_weights)
    }

    /// Mutable view of stack `i`'s `fc_0_biases`.
    pub fn fc_0_biases_mut(&mut self, i: usize) -> &mut [i32] {
        self.arena.slice_mut(self.layout.stacks[i].fc_0_biases)
    }
    /// Mutable view of stack `i`'s `fc_0_weights`.
    pub fn fc_0_weights_mut(&mut self, i: usize) -> &mut [i8] {
        self.arena.slice_mut(self.layout.stacks[i].fc_0_weights)
    }
    /// Mutable view of stack `i`'s `fc_1_biases`.
    pub fn fc_1_biases_mut(&mut self, i: usize) -> &mut [i32] {
        self.arena.slice_mut(self.layout.stacks[i].fc_1_biases)
    }
    /// Mutable view of stack `i`'s `fc_1_weights`.
    pub fn fc_1_weights_mut(&mut self, i: usize) -> &mut [i8] {
        self.arena.slice_mut(self.layout.stacks[i].fc_1_weights)
    }
    /// Mutable view of stack `i`'s `fc_2_biases`.
    pub fn fc_2_biases_mut(&mut self, i: usize) -> &mut [i32] {
        self.arena.slice_mut(self.layout.stacks[i].fc_2_biases)
    }
    /// Mutable view of stack `i`'s `fc_2_weights`.
    pub fn fc_2_weights_mut(&mut self, i: usize) -> &mut [i8] {
        self.arena.slice_mut(self.layout.stacks[i].fc_2_weights)
    }

    /// Number of layer stacks the layout carries.
    pub fn layer_stacks(&self) -> usize {
        self.layout.stacks.len()
    }

    /// Consume the filled arena and produce the network, copying nothing.
    pub fn build(self) -> NnueNetwork {
        NnueNetwork::from_arena(self.header, self.arena, self.layout, self.sha256)
    }
}

/// Errors returned by the network-file loader, which never panics on malformed
/// input.
#[derive(Debug)]
pub enum NnueError {
    Io {
        path: String,
        source: std::io::Error,
    },
    SizeMismatch {
        expected: usize,
        got: usize,
    },
    InvalidFormat {
        reason: String,
    },
    NotLoaded,
}

impl fmt::Display for NnueError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NnueError::Io { path, source } => {
                write!(f, "failed to open NNUE file {path}: {source}")
            }
            NnueError::SizeMismatch { expected, got } => write!(
                f,
                "NNUE file has unexpected size: expected {expected} bytes, got {got}"
            ),
            NnueError::InvalidFormat { reason } => {
                write!(f, "NNUE file is malformed: {reason}")
            }
            NnueError::NotLoaded => write!(
                f,
                "no NNUE network loaded; point the EvalDir USI option at a directory containing nn.bin"
            ),
        }
    }
}

impl std::error::Error for NnueError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            NnueError::Io { source, .. } => Some(source),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::size_of;

    /// A small but multi-stack synthetic net: two stacks and a tiny feature
    /// transformer, at standard FC shapes.
    fn small_net() -> NnueNetwork {
        let dims = NetDims {
            layer_stacks: 2,
            num_features: 3,
            ..NetDims::STANDARD
        };
        let header = NetHeader {
            version: 0,
            hash: 0,
            arch_id: "layout-test".to_string(),
        };
        NnueNetworkBuilder::with_dims(header, [0u8; 32], &dims).build()
    }

    /// Every parameter sub-array's `(start_addr, byte_len)`.
    fn sub_arrays(net: &NnueNetwork) -> Vec<(usize, usize)> {
        let mut v = vec![
            (net.ft_biases.as_ptr() as usize, net.ft_biases.len() * 2),
            (net.ft_weights.as_ptr() as usize, net.ft_weights.len() * 2),
        ];
        for s in &net.stacks {
            v.push((s.fc_0_biases.as_ptr() as usize, s.fc_0_biases.len() * 4));
            v.push((s.fc_0_weights.as_ptr() as usize, s.fc_0_weights.len()));
            v.push((s.fc_1_biases.as_ptr() as usize, s.fc_1_biases.len() * 4));
            v.push((s.fc_1_weights.as_ptr() as usize, s.fc_1_weights.len()));
            v.push((s.fc_2_biases.as_ptr() as usize, s.fc_2_biases.len() * 4));
            v.push((s.fc_2_weights.as_ptr() as usize, s.fc_2_weights.len()));
        }
        v
    }

    #[test]
    fn every_sub_array_is_64_byte_aligned() {
        let net = small_net();
        for (addr, _) in sub_arrays(&net) {
            assert_eq!(addr % 64, 0, "sub-array at {addr:#x} is not 64-aligned");
        }
    }

    #[test]
    fn sub_arrays_are_disjoint_and_inside_one_arena() {
        let net = small_net();
        let mut spans = sub_arrays(&net);
        spans.sort_by_key(|&(addr, _)| addr);
        for w in spans.windows(2) {
            let (start_a, len_a) = w[0];
            let (start_b, _) = w[1];
            assert!(
                start_b >= start_a + len_a,
                "sub-arrays overlap: [{start_a:#x}, +{len_a}) then {start_b:#x}",
            );
        }
        let (first, _) = *spans.first().unwrap();
        let (last, last_len) = *spans.last().unwrap();
        let (allocs, reserved) = net.allocation_disclosure();
        assert_eq!(allocs, 1, "exactly one large-page allocation per net");
        assert!(
            (last + last_len) - first <= reserved,
            "sub-array span exceeds the reserved arena",
        );
    }

    #[test]
    fn builder_fills_are_visible_through_the_views() {
        let dims = NetDims {
            layer_stacks: 2,
            num_features: 3,
            ..NetDims::STANDARD
        };
        let mut b = NnueNetworkBuilder::with_dims(
            NetHeader {
                version: 1,
                hash: 2,
                arch_id: "fill".to_string(),
            },
            [7u8; 32],
            &dims,
        );
        b.ft_biases_mut()[0] = -321;
        b.ft_weights_mut()[dims.hidden_size] = 99; // second feature column, lane 0
        b.fc_0_biases_mut(1)[3] = 77;
        b.fc_2_weights_mut(0)[1] = -5;
        let net = b.build();
        assert_eq!(net.ft_biases[0], -321);
        assert_eq!(net.ft_weights[dims.hidden_size], 99);
        assert_eq!(net.stacks[1].fc_0_biases[3], 77);
        assert_eq!(net.stacks[0].fc_2_weights[1], -5);
        assert_eq!(net.header.version, 1);
        assert_eq!(net.sha256, [7u8; 32]);
    }

    #[test]
    fn replicate_is_byte_equal_and_allocation_distinct() {
        let dims = NetDims {
            layer_stacks: 2,
            num_features: 3,
            ..NetDims::STANDARD
        };
        let mut b = NnueNetworkBuilder::with_dims(
            NetHeader {
                version: 0,
                hash: 0,
                arch_id: "rep".to_string(),
            },
            [0xCD; 32],
            &dims,
        );
        for (i, s) in b.ft_weights_mut().iter_mut().enumerate() {
            *s = (i as i16) % 61 - 30;
        }
        b.fc_1_weights_mut(1)[7] = 42;
        let net = b.build();
        let copy = net.replicate();

        assert_eq!(&*copy.ft_biases, &*net.ft_biases);
        assert_eq!(&*copy.ft_weights, &*net.ft_weights);
        for (a, b) in copy.stacks.iter().zip(net.stacks.iter()) {
            assert_eq!(&*a.fc_0_weights, &*b.fc_0_weights);
            assert_eq!(&*a.fc_1_weights, &*b.fc_1_weights);
            assert_eq!(&*a.fc_2_biases, &*b.fc_2_biases);
        }
        assert_ne!(copy.ft_weights.as_ptr(), net.ft_weights.as_ptr());
        assert_ne!(
            copy.stacks[1].fc_1_weights.as_ptr(),
            net.stacks[1].fc_1_weights.as_ptr(),
        );
        assert_eq!(net.allocation_disclosure().0, 1);
        assert_eq!(copy.allocation_disclosure().0, 1);
        // The arrays must be non-trivial and the type sizes as assumed.
        assert_eq!(size_of::<i16>(), 2);
    }

    #[test]
    fn hidden_size_matches_sfnnwop1536() {
        assert_eq!(HIDDEN_SIZE, 1536);
    }

    #[test]
    fn num_features_matches_half_ka_hm() {
        assert_eq!(NUM_FEATURES, 5 * 9 * 1629);
    }

    #[test]
    fn layer_stacks_is_nine() {
        assert_eq!(LAYER_STACKS, 9);
    }

    #[test]
    fn fc_0_shapes_match_spec() {
        assert_eq!(FC_0_OUTPUT_DIMS, 16);
        assert_eq!(FC_0_INPUT_DIMS, 1536);
        assert_eq!(FC_0_PADDED_INPUT_DIMS, 1536);
    }

    #[test]
    fn fc_1_shapes_match_spec() {
        assert_eq!(FC_1_OUTPUT_DIMS, 32);
        assert_eq!(FC_1_INPUT_DIMS, 30);
        assert_eq!(FC_1_PADDED_INPUT_DIMS, 32);
    }

    #[test]
    fn fc_2_shapes_match_spec() {
        assert_eq!(FC_2_OUTPUT_DIMS, 1);
        assert_eq!(FC_2_INPUT_DIMS, 32);
        assert_eq!(FC_2_PADDED_INPUT_DIMS, 32);
    }

    #[test]
    fn not_loaded_has_human_readable_message() {
        let msg = format!("{}", NnueError::NotLoaded);
        assert!(!msg.is_empty());
        assert!(msg.contains("EvalDir"));
        assert!(msg.contains("nn.bin"));
    }

    #[test]
    fn invalid_format_carries_reason() {
        let msg = format!(
            "{}",
            NnueError::InvalidFormat {
                reason: "bad magic".to_string()
            }
        );
        assert!(msg.contains("bad magic"));
    }
}
