//! Applying RoPE to embeddings.
//!
//! ## Implemented (Stage 4)
//!
//! - [`RotaryEmbedding`]: holds a [`RopeConfig`] and its inverse frequencies
//!   `θ₀, ..., θ_(d/2−1)`. It calculates the frequencies **one time**, in
//!   [`RotaryEmbedding::new`].
//! - [`RotaryEmbedding::rotate`]: returns a **new** rotated vector. The
//!   input is borrowed and does not change.
//! - [`RotaryEmbedding::rotate_in_place`]: rotates a vector **in place**.
//!   No heap allocation.
//!
//! ## Implemented (Stage 5)
//!
//! - Query and key tensors `[B, S, H, D]`: [`RotaryEmbedding::rotate_tensor_in_place`],
//!   [`RotaryEmbedding::rotate_query_key_in_place`], [`RotaryEmbedding::rotate_query_key`].
//!
//! ## Implemented (Stage 7)
//!
//! - Both layouts ([`RopeLayout`]) and partial RoPE (head dimension `D` ≥
//!   rotary dimension `d`), selected in the [`RopeConfig`]. All methods
//!   (vectors and tensors) support them through one loop.
//!
//! ## Implemented (Stage 9)
//!
//! - [`RotaryEmbedding::sin_cos_table`]: builds a [`SinCosTable`] (a
//!   `cos`/`sin` cache) for a range of positions.
//! - [`RotaryEmbedding::rotate_tensor_with_table`] and
//!   [`RotaryEmbedding::rotate_query_key_with_table`]: the tensor rotation
//!   without trigonometry. Bit-identical to the Stage 5 functions.
//! - [`RotaryEmbedding::rotate_tensor_parallel`]: the same rotation on
//!   several threads ([`std::thread::scope`]).
//!
//! ## Not implemented
//!
//! - ~~The split-half dimension layout.~~ Implemented in Stage 7.
//! - ~~Caches of `cos`/`sin`.~~ Implemented in Stage 9 ([`SinCosTable`]).
//! - Attention and softmax: outside the scope of the project.
//! - `f32` tensors and explicit SIMD: not implemented (explain.md, Entry 17).
//!
//! ## The transformation
//!
//! For an embedding `x = [x₀, x₁, ..., x_(d−1)]` at the token position `m`:
//!
//! ```text
//! for each pair j = 0, 1, ..., d/2 − 1:
//!     φ = m · θⱼ                                   (angle, Stage 3)
//!     x'₂ⱼ   = x₂ⱼ · cos φ − x₂ⱼ₊₁ · sin φ         (rotation, rope::rotations)
//!     x'₂ⱼ₊₁ = x₂ⱼ · sin φ + x₂ⱼ₊₁ · cos φ
//! ```
//!
//! ## Dimension layout: interleaved (adjacent pairs)
//!
//! **Stage 4 used only the interleaved layout; Stage 7 adds split-half** (see
//! [`RopeLayout`]). In the interleaved layout, pair `j` is the two
//! **adjacent** components `(x₂ⱼ, x₂ⱼ₊₁)`:
//!
//! ```text
//! d = 8:  [x0, x1 | x2, x3 | x4, x5 | x6, x7]
//!          pair 0   pair 1   pair 2   pair 3
//! ```
//!
//! This is the layout of the RoFormer paper and of the original LLaMA code
//! (which views adjacent values as one complex number).
//!
//! The **split-half** layout (used, for example, by the Hugging Face
//! `rotate_half` implementation and GPT-NeoX) pairs component `j` with
//! component `j + d/2`:
//!
//! ```text
//! d = 8:  [x0, x1, x2, x3 | x4, x5, x6, x7]
//!          pairs: (x0, x4), (x1, x5), (x2, x6), (x3, x7)
//! ```
//!
//! Both layouts give a valid RoPE with the same mathematical properties, but
//! they give **different numbers** for the same input vector. Thus a model
//! trained with one layout needs the same layout at inference time. Compare
//! results with an external implementation only after you confirm its
//! layout.

use crate::math::tensor::{Tensor4, TensorShape};
use crate::rope::config::{RopeConfig, RopeLayout};
use crate::rope::frequencies::{SinCosTable, angle_unchecked, inverse_frequencies};
use crate::rope::position::{Position, PositionRange};
use crate::rope::rotations::{rotate_pair, rotate_pair_sin_cos};
use crate::rope::scaling::{RopeScaling, scaled_inverse_frequencies};
use crate::utils::validation::{RopeError, validate_embedding_dimension};
use std::num::NonZeroUsize;

/// The RoPE transformation for one configuration (`d`, `b`), with the
/// interleaved (adjacent-pair) layout.
///
/// # Why a struct?
///
/// The inverse frequencies depend only on the configuration. A model applies
/// RoPE to many vectors with the same configuration. The struct calculates
/// the `d/2` frequencies one time and keeps them. Each rotation then only
/// reads them. Without the struct, every call would repeat `d/2` `powf`
/// calculations.
///
/// # Example
///
/// ```
/// use rust_rope::rope::config::RopeConfig;
/// use rust_rope::rope::embeddings::RotaryEmbedding;
/// use rust_rope::rope::position::Position;
///
/// let rope = RotaryEmbedding::new(RopeConfig::new(4, 10_000.0)?);
/// let x = [1.0, 0.0, 1.0, 0.0];
///
/// // Position 0: no rotation.
/// assert_eq!(rope.rotate(&x, Position::ZERO)?, x);
///
/// // Position 1: pair 0 rotates by θ₀ = 1 rad, pair 1 by θ₁ = 0.01 rad.
/// let y = rope.rotate(&x, Position::new(1)?)?;
/// assert_eq!(y, vec![1f64.cos(), 1f64.sin(), 0.01f64.cos(), 0.01f64.sin()]);
/// # Ok::<(), rust_rope::utils::validation::RopeError>(())
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct RotaryEmbedding {
    config: RopeConfig,
    /// `θⱼ = b^(−2j/d)` for `j = 0..d/2` (or the scaled `θ'ⱼ`, Stage 8).
    /// Length `d/2`. Never empty.
    inverse_frequencies: Vec<f64>,
    /// The scaling method (Stage 8). `RopeScaling::None` for `new`.
    scaling: RopeScaling,
    /// The factor for the rotated components: 1.0 except for YaRN (Stage 8).
    attention_factor: f64,
}

impl RotaryEmbedding {
    /// Creates the transformation and calculates the `d/2` inverse
    /// frequencies.
    ///
    /// This function cannot fail: a [`RopeConfig`] is always valid.
    ///
    /// `config` is passed by value. `RopeConfig` is `Copy` (16 bytes), so the
    /// caller keeps its own copy.
    pub fn new(config: RopeConfig) -> Self {
        RotaryEmbedding {
            inverse_frequencies: inverse_frequencies(&config),
            config,
            scaling: RopeScaling::None,
            attention_factor: 1.0,
        }
    }

    /// Creates the transformation with a scaling method (Stage 8).
    ///
    /// The scaled frequencies `θ'ⱼ` and the attention factor are calculated
    /// **one time** here. The rotation code then uses them exactly as it
    /// uses the standard frequencies. `RopeScaling::None` gives the same
    /// result as [`RotaryEmbedding::new`].
    ///
    /// # Errors
    ///
    /// The errors of [`RopeScaling::validate`].
    ///
    /// # Example
    ///
    /// ```
    /// use rust_rope::rope::config::RopeConfig;
    /// use rust_rope::rope::embeddings::RotaryEmbedding;
    /// use rust_rope::rope::position::Position;
    /// use rust_rope::rope::scaling::RopeScaling;
    ///
    /// let config = RopeConfig::new(8, 10_000.0)?;
    /// let pi = RotaryEmbedding::with_scaling(
    ///     config,
    ///     RopeScaling::PositionInterpolation { original_context: 2048, target_context: 8192 },
    /// )?;
    /// let standard = RotaryEmbedding::new(config);
    /// // Position 4096 with interpolation = position 1024 without it.
    /// let x = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
    /// let a = pi.rotate(&x, Position::new(4096)?)?;
    /// let b = standard.rotate(&x, Position::new(1024)?)?;
    /// assert!(a.iter().zip(&b).all(|(p, q)| (p - q).abs() < 1e-12));
    /// # Ok::<(), rust_rope::utils::validation::RopeError>(())
    /// ```
    pub fn with_scaling(config: RopeConfig, scaling: RopeScaling) -> Result<Self, RopeError> {
        Ok(RotaryEmbedding {
            inverse_frequencies: scaled_inverse_frequencies(&config, &scaling)?,
            config,
            scaling,
            attention_factor: scaling.attention_factor(),
        })
    }

    /// Returns the scaling method (Stage 8).
    pub fn scaling(&self) -> RopeScaling {
        self.scaling
    }

    /// Returns the factor that multiplies the rotated components: 1.0
    /// except for YaRN (Stage 8).
    pub fn attention_factor(&self) -> f64 {
        self.attention_factor
    }

    /// Returns the configuration.
    pub fn config(&self) -> RopeConfig {
        self.config
    }

    /// Returns the required length of every input vector: the head
    /// dimension `D`.
    ///
    /// Before Stage 7, this was always the rotary dimension `d`. For a
    /// configuration without [`RopeConfig::with_head_dimension`], `D = d`, so
    /// the value is unchanged.
    pub fn dimension(&self) -> usize {
        self.config.head_dimension()
    }

    /// Returns the rotary dimension `d`: the number of components that
    /// rotate (Stage 7).
    pub fn rotary_dimension(&self) -> usize {
        self.config.rotary_dimension()
    }

    /// Returns the inverse frequencies `[θ₀, ..., θ_(d/2−1)]` as a borrowed
    /// slice.
    pub fn inverse_frequencies(&self) -> &[f64] {
        &self.inverse_frequencies
    }

    /// Applies RoPE to `embedding` at the token position `m` and returns the
    /// rotated vector as a **new** `Vec<f64>`.
    ///
    /// The function **borrows** the input (`&[f64]`). The input does not
    /// change. The function allocates exactly one `Vec` of `d` values.
    ///
    /// # Errors
    ///
    /// [`RopeError::EmbeddingDimensionMismatch`] if `embedding.len() != d`.
    /// Then the function does not allocate.
    ///
    /// # Numerical values
    ///
    /// The function does not check the values of the components. `NaN` or
    /// infinity in a pair gives `NaN` or infinity in the same pair of the
    /// result. Other pairs are not affected.
    pub fn rotate(&self, embedding: &[f64], position: Position) -> Result<Vec<f64>, RopeError> {
        // Check first, so that an invalid input causes no allocation.
        validate_embedding_dimension(embedding.len(), self.dimension())?;
        // The one necessary copy: the caller keeps `embedding`, and the result
        // needs its own memory. `to_vec` allocates exactly `d` values and
        // copies them.
        let mut output = embedding.to_vec();
        self.rotate_pairs(&mut output, position);
        Ok(output)
    }

    /// Applies RoPE to `embedding` at the token position `m`, **in place**:
    /// the function overwrites the input values with the rotated values.
    ///
    /// The function borrows the input mutably (`&mut [f64]`). It does not
    /// allocate memory.
    ///
    /// # Errors
    ///
    /// [`RopeError::EmbeddingDimensionMismatch`] if `embedding.len() != d`.
    /// Then the function does not change the input.
    pub fn rotate_in_place(
        &self,
        embedding: &mut [f64],
        position: Position,
    ) -> Result<(), RopeError> {
        validate_embedding_dimension(embedding.len(), self.dimension())?;
        self.rotate_pairs(embedding, position);
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Stage 5: query and key tensors with the layout [B, S, H, D]
    // -----------------------------------------------------------------------

    /// Checks a `[B, S, H, D]` tensor for RoPE and returns the positions of
    /// its `S` tokens: `start, start + 1, ..., start + S − 1`.
    fn check_tensor(&self, tensor: &Tensor4, start: Position) -> Result<PositionRange, RopeError> {
        let shape = tensor.shape();
        // Each head vector has D features. It must match the RoPE dimension.
        validate_embedding_dimension(shape.features, self.dimension())?;
        PositionRange::new(start.value(), shape.sequence)
    }

    /// Checks that a query tensor and a key tensor can be rotated together.
    fn check_query_key(
        &self,
        query: &Tensor4,
        key: &Tensor4,
        start: Position,
    ) -> Result<(), RopeError> {
        let (q, k) = (query.shape(), key.shape());
        if (q.batch, q.sequence) != (k.batch, k.sequence) {
            return Err(RopeError::QueryKeyShapeMismatch {
                query: (q.batch, q.sequence),
                key: (k.batch, k.sequence),
            });
        }
        self.check_tensor(query, start)?;
        self.check_tensor(key, start)?;
        Ok(())
    }

    /// Applies RoPE to every head vector of a `[B, S, H, D]` tensor, in
    /// place.
    ///
    /// The token at sequence index `s` has the position `m = start + s`. The
    /// position is the same for every batch element `b` and every head `h`.
    ///
    /// ```text
    /// for b in 0..B:                       (batch elements are independent)
    ///     for s in 0..S:  m = start + s    (one position per token)
    ///         for h in 0..H:               (heads are independent)
    ///             tensor[b, s, h, :] = R_m · tensor[b, s, h, :]
    /// ```
    ///
    /// No allocation. Every head uses the same `θⱼ` (stored in `self`).
    ///
    /// # Errors
    ///
    /// - [`RopeError::EmbeddingDimensionMismatch`] if `D ≠ d`.
    /// - [`RopeError::SequenceTooLong`] if `start + S − 1 > 2^53`.
    ///
    /// On an error, the tensor does not change.
    pub fn rotate_tensor_in_place(
        &self,
        tensor: &mut Tensor4,
        start: Position,
    ) -> Result<(), RopeError> {
        let positions = self.check_tensor(tensor, start)?;
        self.rotate_tensor_unchecked(tensor, positions);
        Ok(())
    }

    /// Applies RoPE to a query tensor and a key tensor, in place.
    ///
    /// Both tensors use the layout `[B, S, H, D]` and the same positions
    /// `start + s`. `B` and `S` must be equal. The numbers of heads may be
    /// different. (Some models use fewer key heads than query heads.)
    ///
    /// The function checks **both** tensors before it changes **either**.
    /// Thus an error leaves both tensors unchanged.
    ///
    /// There is no value tensor parameter. Standard RoPE does not rotate the
    /// values (explain.md, Entry 12).
    ///
    /// # Errors
    ///
    /// - [`RopeError::QueryKeyShapeMismatch`] if `B` or `S` differ.
    /// - The errors of [`RotaryEmbedding::rotate_tensor_in_place`].
    pub fn rotate_query_key_in_place(
        &self,
        query: &mut Tensor4,
        key: &mut Tensor4,
        start: Position,
    ) -> Result<(), RopeError> {
        self.check_query_key(query, key, start)?;
        let positions = PositionRange::new(start.value(), query.shape().sequence)?;
        self.rotate_tensor_unchecked(query, positions);
        self.rotate_tensor_unchecked(key, positions);
        Ok(())
    }

    /// Applies RoPE to a query tensor and a key tensor and returns the
    /// rotated copies `(query', key')`. The inputs are borrowed and do not
    /// change.
    ///
    /// This allocates two new tensors (one `clone` each). Use
    /// [`RotaryEmbedding::rotate_query_key_in_place`] to avoid the
    /// allocations.
    ///
    /// # Errors
    ///
    /// The errors of [`RotaryEmbedding::rotate_query_key_in_place`]. On an
    /// error, nothing is allocated.
    ///
    /// # Example
    ///
    /// ```
    /// use rust_rope::math::tensor::{Tensor4, TensorShape};
    /// use rust_rope::rope::config::RopeConfig;
    /// use rust_rope::rope::embeddings::RotaryEmbedding;
    /// use rust_rope::rope::position::Position;
    ///
    /// let rope = RotaryEmbedding::new(RopeConfig::new(4, 10_000.0)?);
    /// let shape = TensorShape::new(2, 3, 2, 4); // B = 2, S = 3, H = 2, D = 4
    /// let q = Tensor4::from_fn(shape, |b, s, h, d| (b + s + h + d) as f64)?;
    /// let k = Tensor4::from_fn(shape, |_, _, _, d| d as f64)?;
    ///
    /// let (q2, k2) = rope.rotate_query_key(&q, &k, Position::ZERO)?;
    /// // Token s = 0 has position 0: no rotation.
    /// assert_eq!(q2.vector(1, 0, 1), q.vector(1, 0, 1));
    /// assert_ne!(k2.vector(0, 2, 0), k.vector(0, 2, 0));
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn rotate_query_key(
        &self,
        query: &Tensor4,
        key: &Tensor4,
        start: Position,
    ) -> Result<(Tensor4, Tensor4), RopeError> {
        self.check_query_key(query, key, start)?;
        let positions = PositionRange::new(start.value(), query.shape().sequence)?;
        // The two necessary copies: the caller keeps the originals.
        let (mut q, mut k) = (query.clone(), key.clone());
        self.rotate_tensor_unchecked(&mut q, positions);
        self.rotate_tensor_unchecked(&mut k, positions);
        Ok((q, k))
    }

    // -----------------------------------------------------------------------
    // Stage 9: rotation with a sin/cos table, and multithreaded rotation
    // -----------------------------------------------------------------------

    /// Calculates the `sin`/`cos` table for `positions` with the
    /// frequencies of this transformation (Stage 9).
    ///
    /// Build the table **one time** for the longest sequence that you
    /// expect (for example, the context length of the model). Then use it
    /// for every tensor with [`RotaryEmbedding::rotate_tensor_with_table`].
    ///
    /// # Errors
    ///
    /// [`RopeError::TableTooLarge`] if `L · d/2` does not fit in a `usize`.
    ///
    /// # Example
    ///
    /// ```
    /// use rust_rope::math::tensor::{Tensor4, TensorShape};
    /// use rust_rope::rope::config::RopeConfig;
    /// use rust_rope::rope::embeddings::RotaryEmbedding;
    /// use rust_rope::rope::position::{Position, PositionRange};
    ///
    /// let rope = RotaryEmbedding::new(RopeConfig::new(8, 10_000.0)?);
    /// let table = rope.sin_cos_table(PositionRange::from_zero(1024)?)?;
    ///
    /// let shape = TensorShape::new(2, 16, 4, 8);
    /// let input = Tensor4::from_fn(shape, |b, s, h, d| (b + s + h + d) as f64)?;
    /// let (mut cached, mut direct) = (input.clone(), input);
    ///
    /// // Tokens 100..116, for example a second chunk of a long text.
    /// let start = Position::new(100)?;
    /// rope.rotate_tensor_with_table(&mut cached, &table, start)?;
    /// rope.rotate_tensor_in_place(&mut direct, start)?;
    /// assert_eq!(cached, direct); // bit-identical
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn sin_cos_table(&self, positions: PositionRange) -> Result<SinCosTable, RopeError> {
        SinCosTable::new_unchecked(positions, &self.inverse_frequencies)
    }

    /// Checks that `table` fits this transformation and contains the `S`
    /// positions `start, ..., start + S − 1`. Returns the table row of
    /// `start`.
    fn check_table(
        &self,
        table: &SinCosTable,
        start: Position,
        sequence: usize,
    ) -> Result<usize, RopeError> {
        let expected = self.inverse_frequencies.len();
        if table.num_pairs() != expected {
            return Err(RopeError::TablePairCountMismatch {
                expected,
                actual: table.num_pairs(),
            });
        }
        let needed = PositionRange::new(start.value(), sequence)?;
        let table_start = table.positions().start().value();
        let table_end = table_start + table.num_positions(); // no overflow: ≤ 2^53 + 1
        let needed_end = start.value() + needed.len();
        if start.value() < table_start || needed_end > table_end {
            return Err(RopeError::TableDoesNotCoverPositions {
                table_start,
                table_length: table.num_positions(),
                start: start.value(),
                length: sequence,
            });
        }
        Ok(start.value() - table_start)
    }

    /// Applies RoPE to every head vector of a `[B, S, H, D]` tensor, in
    /// place, with precomputed `sin`/`cos` values (Stage 9).
    ///
    /// The result is **bit-identical** to
    /// [`RotaryEmbedding::rotate_tensor_in_place`]. The difference is the
    /// speed: this function calls no trigonometric function. Each pair costs
    /// 4 multiplications and 2 additions (plus the attention factor).
    ///
    /// The table must come from [`RotaryEmbedding::sin_cos_table`] of the
    /// **same** transformation. The function checks the number of pairs and
    /// the positions. It cannot check the values of the frequencies. A table
    /// from a different base or scaling method gives wrong rotations.
    ///
    /// No heap allocation.
    ///
    /// # Errors
    ///
    /// - [`RopeError::EmbeddingDimensionMismatch`] if `D` is not the head
    ///   dimension.
    /// - [`RopeError::TablePairCountMismatch`] if the table has a different
    ///   number of pairs.
    /// - [`RopeError::TableDoesNotCoverPositions`] if the table does not
    ///   contain the positions `start, ..., start + S − 1`.
    /// - [`RopeError::SequenceTooLong`] if `start + S − 1 > 2^53`.
    ///
    /// On an error, the tensor does not change.
    pub fn rotate_tensor_with_table(
        &self,
        tensor: &mut Tensor4,
        table: &SinCosTable,
        start: Position,
    ) -> Result<(), RopeError> {
        let shape = tensor.shape();
        validate_embedding_dimension(shape.features, self.dimension())?;
        let first_row = self.check_table(table, start, shape.sequence)?;
        self.rotate_tokens_with_table(tensor.as_mut_slice(), 0, shape, table, first_row);
        Ok(())
    }

    /// Applies RoPE to a query tensor and a key tensor, in place, with a
    /// `sin`/`cos` table (Stage 9). The rules are the same as for
    /// [`RotaryEmbedding::rotate_query_key_in_place`]. Both tensors use the
    /// same table rows, so one table serves both.
    ///
    /// # Errors
    ///
    /// The errors of [`RotaryEmbedding::rotate_query_key_in_place`] and of
    /// [`RotaryEmbedding::rotate_tensor_with_table`]. On an error, neither
    /// tensor changes.
    pub fn rotate_query_key_with_table(
        &self,
        query: &mut Tensor4,
        key: &mut Tensor4,
        table: &SinCosTable,
        start: Position,
    ) -> Result<(), RopeError> {
        self.check_query_key(query, key, start)?;
        let first_row = self.check_table(table, start, query.shape().sequence)?;
        let (q_shape, k_shape) = (query.shape(), key.shape());
        self.rotate_tokens_with_table(query.as_mut_slice(), 0, q_shape, table, first_row);
        self.rotate_tokens_with_table(key.as_mut_slice(), 0, k_shape, table, first_row);
        Ok(())
    }

    /// Applies RoPE to a `[B, S, H, D]` tensor with `threads` operating
    /// system threads and a `sin`/`cos` table (Stage 9).
    ///
    /// The function divides the `B · S` tokens into `threads` contiguous
    /// blocks. Each thread rotates one block. The blocks do not overlap, so
    /// the threads never write to the same value. The result is
    /// **bit-identical** to [`RotaryEmbedding::rotate_tensor_with_table`].
    ///
    /// ```text
    /// data: [ token 0 | token 1 | ... | token B·S−1 ]
    ///         └─ thread 0 ─┘└─ thread 1 ─┘ ... └─ thread n−1 ─┘
    /// ```
    ///
    /// Starting a thread costs time (tens of microseconds on many
    /// systems). For small tensors, the single-threaded function is faster.
    /// See the Stage 9 benchmark results in `explain.md` (Entry 17).
    ///
    /// The function uses [`std::thread::scope`]. All threads finish before
    /// the function returns. Thus the threads can borrow `self`, `table` and
    /// the tensor data without `Arc` or copies. `threads` is a
    /// [`NonZeroUsize`]: the type system excludes 0 threads.
    ///
    /// # Errors
    ///
    /// The errors of [`RotaryEmbedding::rotate_tensor_with_table`].
    ///
    /// # Panics
    ///
    /// If the operating system cannot start a thread, or if a thread panics
    /// (an internal bug).
    ///
    /// # Example
    ///
    /// ```
    /// use std::num::NonZeroUsize;
    /// use rust_rope::math::tensor::{Tensor4, TensorShape};
    /// use rust_rope::rope::config::RopeConfig;
    /// use rust_rope::rope::embeddings::RotaryEmbedding;
    /// use rust_rope::rope::position::{Position, PositionRange};
    ///
    /// let rope = RotaryEmbedding::new(RopeConfig::new(8, 10_000.0)?);
    /// let table = rope.sin_cos_table(PositionRange::from_zero(64)?)?;
    /// let shape = TensorShape::new(3, 64, 2, 8);
    /// let input = Tensor4::from_fn(shape, |b, s, h, d| (b * s + h + d) as f64)?;
    ///
    /// let (mut serial, mut parallel) = (input.clone(), input);
    /// rope.rotate_tensor_with_table(&mut serial, &table, Position::ZERO)?;
    /// let threads = NonZeroUsize::new(4).unwrap();
    /// rope.rotate_tensor_parallel(&mut parallel, &table, Position::ZERO, threads)?;
    /// assert_eq!(serial, parallel);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn rotate_tensor_parallel(
        &self,
        tensor: &mut Tensor4,
        table: &SinCosTable,
        start: Position,
        threads: NonZeroUsize,
    ) -> Result<(), RopeError> {
        let shape = tensor.shape();
        validate_embedding_dimension(shape.features, self.dimension())?;
        let first_row = self.check_table(table, start, shape.sequence)?;

        let num_tokens = shape.batch * shape.sequence; // ≥ 1: no zero extents
        let token_len = shape.heads * shape.features; // H·D values per token
        // More threads than tokens would give empty blocks.
        let threads = threads.get().min(num_tokens);
        // Round up, so that `threads` blocks contain all tokens.
        let tokens_per_block = num_tokens.div_ceil(threads);

        let data = tensor.as_mut_slice();
        if threads == 1 {
            // No thread start for one block: the current thread does the work.
            self.rotate_tokens_with_table(data, 0, shape, table, first_row);
            return Ok(());
        }

        // `chunks_mut` gives non-overlapping `&mut [f64]` blocks. Each block
        // moves into exactly one thread. The borrow checker proves that no
        // two threads can write to the same memory.
        std::thread::scope(|scope| {
            for (block_index, block) in data.chunks_mut(tokens_per_block * token_len).enumerate() {
                // `move` copies the references `self` and `table` and moves
                // `block` into the thread. `RotaryEmbedding` and
                // `SinCosTable` are `Sync`, so shared references can cross
                // threads.
                scope.spawn(move || {
                    let first_token = block_index * tokens_per_block;
                    self.rotate_tokens_with_table(block, first_token, shape, table, first_row);
                });
            }
            // The end of `scope` waits for (joins) every thread. A panic in
            // a thread continues as a panic here.
        });
        Ok(())
    }

    /// The table loop for a block of whole tokens. `tokens` starts at the
    /// global token index `first_token` (`= b · S + s`). The caller must
    /// check the shape and the table.
    ///
    /// ```text
    /// for each token t in the block:     s = t mod S, row = first_row + s
    ///     (sin, cos) = table rows of row
    ///     for each head h: rotate the head vector with (sin, cos)
    /// ```
    fn rotate_tokens_with_table(
        &self,
        tokens: &mut [f64],
        first_token: usize,
        shape: TensorShape,
        table: &SinCosTable,
        first_row: usize,
    ) {
        let token_len = shape.heads * shape.features;
        debug_assert_eq!(tokens.len() % token_len, 0);
        for (offset, token) in tokens.chunks_exact_mut(token_len).enumerate() {
            // The sequence index of this token. All batch elements use the
            // same positions, so `s` restarts at 0 in each batch element.
            let s = (first_token + offset) % shape.sequence;
            let (sin, cos) = table.rows_at(first_row + s);
            for head in token.chunks_exact_mut(shape.features) {
                self.rotate_pairs_with_table(head, sin, cos);
            }
        }
    }

    /// The RoPE loop for one head vector with precomputed `sin`/`cos`
    /// rows. It has the same structure as `rotate_pairs`. The only
    /// difference: it reads `sin φ` and `cos φ` and does not calculate them.
    #[inline]
    fn rotate_pairs_with_table(&self, embedding: &mut [f64], sin: &[f64], cos: &[f64]) {
        let rotary_dimension = self.config.rotary_dimension();
        debug_assert_eq!(embedding.len(), self.config.head_dimension());
        debug_assert_eq!(sin.len(), rotary_dimension / 2);
        debug_assert_eq!(cos.len(), rotary_dimension / 2);

        let (rotary, _pass_through) = embedding.split_at_mut(rotary_dimension);
        let factor = self.attention_factor;

        match self.config.layout() {
            RopeLayout::Interleaved => {
                // `as_chunks_mut::<2>()` views the slice as `&mut [[f64; 2]]`.
                // The length 2 is part of the type, so the compiler needs no
                // bounds check for `pair[0]` and `pair[1]` (Stage 9, step 1).
                let (pairs, rest) = rotary.as_chunks_mut::<2>();
                debug_assert!(rest.is_empty()); // d is even
                for ((pair, &s), &c) in pairs.iter_mut().zip(sin).zip(cos) {
                    let [ra, rb] = rotate_pair_sin_cos(*pair, s, c);
                    *pair = [ra * factor, rb * factor];
                }
            }
            RopeLayout::SplitHalf => {
                let (first, second) = rotary.split_at_mut(rotary_dimension / 2);
                for (((a, b), &s), &c) in first.iter_mut().zip(second.iter_mut()).zip(sin).zip(cos)
                {
                    let [ra, rb] = rotate_pair_sin_cos([*a, *b], s, c);
                    *a = ra * factor;
                    *b = rb * factor;
                }
            }
        }
    }

    /// The tensor loop. The caller must check `D = d` and the positions.
    ///
    /// The nested `chunks_exact_mut` calls follow the memory layout
    /// `[B, S, H, D]` from the outside to the inside:
    ///
    /// ```text
    /// data.chunks_exact_mut(S·H·D)   → one batch element per item
    ///   .chunks_exact_mut(H·D)       → one token per item (zip with its position)
    ///     .chunks_exact_mut(D)       → one head vector per item → rotate_pairs
    /// ```
    ///
    /// The traversal visits every value exactly once, in increasing memory
    /// order.
    fn rotate_tensor_unchecked(&self, tensor: &mut Tensor4, positions: PositionRange) {
        let shape = tensor.shape();
        let token_len = shape.heads * shape.features; // H·D values per token
        let batch_len = shape.sequence * token_len; // S·H·D values per batch element
        debug_assert_eq!(positions.len(), shape.sequence);

        for batch_element in tensor.as_mut_slice().chunks_exact_mut(batch_len) {
            for (token, position) in batch_element
                .chunks_exact_mut(token_len)
                .zip(positions.iter())
            {
                for head in token.chunks_exact_mut(shape.features) {
                    self.rotate_pairs(head, position);
                }
            }
        }
    }

    /// The RoPE loop for one head vector. The caller must check that
    /// `embedding.len() == D` (the head dimension).
    ///
    /// Stage 7: only the first `d` (rotary) components rotate; the layout
    /// selects the pairs. The components `d..D` are not touched.
    ///
    /// ```text
    /// embedding:  [ x0 x1 x2 x3 x4 x5 | x6 x7 ]      d = 6, D = 8
    ///               └── rotary (d) ──┘  └ pass ┘
    /// Interleaved:  rotary.as_chunks_mut::<2>() → (x0,x1) (x2,x3) (x4,x5)
    /// SplitHalf:    rotary.split_at_mut(3)     → (x0,x3) (x1,x4) (x2,x5)
    /// ```
    fn rotate_pairs(&self, embedding: &mut [f64], position: Position) {
        let rotary_dimension = self.config.rotary_dimension();
        // Internal consistency checks: a bug if they fail, not a caller error.
        debug_assert_eq!(embedding.len(), self.config.head_dimension());
        debug_assert_eq!(rotary_dimension, 2 * self.inverse_frequencies.len());

        // `split_at_mut(d)` gives two non-overlapping mutable slices: the
        // rotary part `[0, d)` and the pass-through part `[d, D)`. We use only
        // the first one, so the second one stays exactly unchanged.
        let (rotary, _pass_through) = embedding.split_at_mut(rotary_dimension);
        let factor = self.attention_factor;

        match self.config.layout() {
            RopeLayout::Interleaved => {
                // `as_chunks_mut::<2>()`: the adjacent pairs (x₂ⱼ, x₂ⱼ₊₁) as
                // `[f64; 2]` arrays. (Stage 4 used `chunks_exact_mut(2)`,
                // which gives slices of unknown length; Stage 9 changed it.)
                // `zip` walks the pairs and the frequencies together.
                let (pairs, rest) = rotary.as_chunks_mut::<2>();
                debug_assert!(rest.is_empty()); // d is even
                for (pair, &theta) in pairs.iter_mut().zip(&self.inverse_frequencies) {
                    let phi = angle_unchecked(position, theta); // φ = m·θⱼ
                    let [ra, rb] = rotate_pair(*pair, phi); // R(φ)·(x₂ⱼ, x₂ⱼ₊₁)
                    // Write back into the same two slots. `factor` is 1.0
                    // except for YaRN; `x · 1.0 = x` exactly.
                    *pair = [ra * factor, rb * factor];
                }
            }
            RopeLayout::SplitHalf => {
                // Two halves: first = [x0 .. x(d/2−1)], second = [x(d/2) .. x(d−1)].
                // Pair j = (first[j], second[j]) = (xⱼ, x_(j + d/2)).
                let (first, second) = rotary.split_at_mut(rotary_dimension / 2);
                for ((a, b), &theta) in first
                    .iter_mut()
                    .zip(second.iter_mut())
                    .zip(&self.inverse_frequencies)
                {
                    let phi = angle_unchecked(position, theta); // φ = m·θⱼ
                    let [ra, rb] = rotate_pair([*a, *b], phi); // R(φ)·(xⱼ, x_(j+d/2))
                    *a = ra * factor;
                    *b = rb * factor;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::{DEFAULT_TOLERANCE, approx_eq};

    fn rope(d: usize) -> RotaryEmbedding {
        RotaryEmbedding::new(RopeConfig::with_default_base(d).unwrap())
    }

    fn pos(m: usize) -> Position {
        Position::new(m).unwrap()
    }

    #[test]
    fn frequencies_are_calculated_once_and_stored() {
        let config = RopeConfig::new(8, 10_000.0).unwrap();
        let r = RotaryEmbedding::new(config);
        assert_eq!(r.inverse_frequencies(), inverse_frequencies(&config));
        assert_eq!(r.config(), config);
        assert_eq!(r.dimension(), 8);
    }

    #[test]
    fn each_pair_uses_its_own_angle() {
        let r = rope(8);
        let x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let m = pos(3);
        let y = r.rotate(&x, m).unwrap();
        for (j, &theta) in r.inverse_frequencies().iter().enumerate() {
            let expected = rotate_pair([x[2 * j], x[2 * j + 1]], 3.0 * theta);
            assert_eq!([y[2 * j], y[2 * j + 1]], expected, "pair {j}");
        }
    }

    #[test]
    fn new_vector_and_in_place_give_identical_results() {
        let r = rope(8);
        let x = vec![0.5, -1.0, 2.0, 0.0, -3.0, 4.0, 1e-3, -1e3];
        let rotated = r.rotate(&x, pos(17)).unwrap();
        let mut y = x.clone(); // clone only because the test needs both versions
        r.rotate_in_place(&mut y, pos(17)).unwrap();
        assert_eq!(rotated, y);
        // `rotate` did not change its input.
        assert_eq!(x, vec![0.5, -1.0, 2.0, 0.0, -3.0, 4.0, 1e-3, -1e3]);
    }

    #[test]
    fn wrong_length_is_an_error_and_changes_nothing() {
        let r = rope(8);
        let mut short = vec![1.0; 6];
        assert_eq!(
            r.rotate_in_place(&mut short, pos(1)),
            Err(RopeError::EmbeddingDimensionMismatch {
                expected: 8,
                actual: 6
            })
        );
        assert_eq!(short, vec![1.0; 6]);
        assert!(r.rotate(&[1.0; 7], pos(1)).is_err());
        assert!(r.rotate(&[], pos(1)).is_err());
    }

    #[test]
    fn position_zero_returns_the_input() {
        let r = rope(4);
        let x = [1.0, -2.0, 3.0, -4.0];
        assert_eq!(r.rotate(&x, Position::ZERO).unwrap(), x);
    }

    #[test]
    fn rotation_keeps_the_length_of_each_pair() {
        let r = rope(4);
        let x = [3.0, 4.0, -1.0, 2.0];
        let y = r.rotate(&x, pos(1000)).unwrap();
        assert!(approx_eq(
            y[0] * y[0] + y[1] * y[1],
            25.0,
            DEFAULT_TOLERANCE
        ));
        assert!(approx_eq(y[2] * y[2] + y[3] * y[3], 5.0, DEFAULT_TOLERANCE));
    }

    #[test]
    fn non_finite_values_stay_in_their_pair() {
        let r = rope(4);
        let y = r.rotate(&[f64::NAN, 1.0, 1.0, 0.0], pos(1)).unwrap();
        assert!(y[0].is_nan() && y[1].is_nan());
        assert!(y[2].is_finite() && y[3].is_finite());
    }
}
