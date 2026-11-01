//! Four-dimensional tensors stored in one flat, contiguous `Vec<f64>`.
//!
//! ## Implemented (Stage 5)
//!
//! - [`TensorShape`]: the four extents `[B, S, H, D]`.
//! - [`Tensor4`]: a four-dimensional array of `f64` values with row-major
//!   (C-order) layout.
//! - [`TensorError`]: errors for invalid shapes and data lengths.
//!
//! This module is general. It does not know about RoPE or attention. The
//! names of the axes (batch, sequence, heads, features) describe the
//! standard Transformer layout `[B, S, H, D]`, but any four-dimensional data
//! can use the type.
//!
//! ## Layout and the index formula
//!
//! The element `(b, s, h, d)` is at the flat index:
//!
//! ```text
//! index = ((b·S + s)·H + h)·D + d
//! ```
//!
//! The last axis (`d`) changes fastest. Thus the `D` features of one
//! `(b, s, h)` are contiguous: one head vector is one contiguous slice.
//!
//! ```text
//! B = 1, S = 2, H = 2, D = 2:
//! index:   0      1      2      3      4      5      6      7
//! (b,s,h,d) 0000   0001   0010   0011   0100   0101   0110   0111
//!          └ head 0 ┘   └ head 1 ┘   └ head 0 ┘   └ head 1 ┘
//!          └──── token s = 0 ─────┘   └──── token s = 1 ─────┘
//! ```

use std::fmt;

/// The extents of a four-dimensional tensor `[B, S, H, D]`.
///
/// | Field | Symbol | Meaning in a Transformer |
/// |---|---|---|
/// | `batch` | `B` | number of independent sequences |
/// | `sequence` | `S` | number of tokens per sequence |
/// | `heads` | `H` | number of attention heads |
/// | `features` | `D` | number of values per head vector (the head dimension) |
///
/// The fields are public: a shape has no invariant by itself. A
/// [`Tensor4`] checks the shape when you create it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TensorShape {
    /// `B`, the batch size.
    pub batch: usize,
    /// `S`, the sequence length.
    pub sequence: usize,
    /// `H`, the number of heads.
    pub heads: usize,
    /// `D`, the number of features per head.
    pub features: usize,
}

impl TensorShape {
    /// Creates the shape `[batch, sequence, heads, features]`.
    pub const fn new(batch: usize, sequence: usize, heads: usize, features: usize) -> Self {
        TensorShape {
            batch,
            sequence,
            heads,
            features,
        }
    }

    /// Returns the number of elements `B·S·H·D`, or `None` if the product
    /// does not fit in a `usize`.
    pub fn num_elements(&self) -> Option<usize> {
        self.batch
            .checked_mul(self.sequence)?
            .checked_mul(self.heads)?
            .checked_mul(self.features)
    }
}

/// The errors of [`Tensor4`] construction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorError {
    /// One extent is 0. Every axis must have at least one element.
    ZeroExtent {
        /// The name of the axis: `"batch"`, `"sequence"`, `"heads"` or
        /// `"features"`.
        axis: &'static str,
    },
    /// `B·S·H·D` does not fit in a `usize`.
    TooManyElements,
    /// The length of the data is not `B·S·H·D`.
    LengthMismatch {
        /// `B·S·H·D`.
        expected: usize,
        /// The length of the data.
        actual: usize,
    },
}

impl fmt::Display for TensorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            TensorError::ZeroExtent { axis } => {
                write!(f, "the {axis} extent is 0; every extent must be at least 1")
            }
            TensorError::TooManyElements => {
                write!(f, "the number of elements B·S·H·D does not fit in a usize")
            }
            TensorError::LengthMismatch { expected, actual } => write!(
                f,
                "the data has {actual} values, but the shape needs B·S·H·D = {expected}"
            ),
        }
    }
}

impl std::error::Error for TensorError {}

/// A four-dimensional tensor of `f64` values with the logical shape
/// `[B, S, H, D]`, stored in one flat `Vec<f64>` in row-major order.
///
/// # Example
///
/// ```
/// use rust_rope::math::tensor::{Tensor4, TensorShape};
///
/// // B = 1, S = 2, H = 1, D = 3: two tokens, one head, three features.
/// let shape = TensorShape::new(1, 2, 1, 3);
/// let t = Tensor4::new(shape, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])?;
///
/// assert_eq!(t.get(0, 1, 0, 2), Some(6.0));            // last value
/// assert_eq!(t.vector(0, 1, 0), Some(&[4.0, 5.0, 6.0][..]));
/// assert_eq!(t.get(0, 2, 0, 0), None);                  // s = 2 is out of range
/// # Ok::<(), rust_rope::math::tensor::TensorError>(())
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct Tensor4 {
    shape: TensorShape,
    /// Row-major: element `(b, s, h, d)` is at `((b·S + s)·H + h)·D + d`.
    data: Vec<f64>,
}

impl Tensor4 {
    /// Checks a shape: every extent ≥ 1 and `B·S·H·D` fits in a `usize`.
    /// Returns `B·S·H·D`.
    fn checked_len(shape: TensorShape) -> Result<usize, TensorError> {
        for (axis, extent) in [
            ("batch", shape.batch),
            ("sequence", shape.sequence),
            ("heads", shape.heads),
            ("features", shape.features),
        ] {
            if extent == 0 {
                return Err(TensorError::ZeroExtent { axis });
            }
        }
        shape.num_elements().ok_or(TensorError::TooManyElements)
    }

    /// Creates a tensor from flat row-major data.
    ///
    /// The function **takes ownership** of `data`: the `Vec` moves into the
    /// tensor. Nothing is copied. The caller cannot use `data` after the
    /// call.
    ///
    /// # Errors
    ///
    /// - [`TensorError::ZeroExtent`] if an extent is 0.
    /// - [`TensorError::TooManyElements`] if `B·S·H·D` overflows.
    /// - [`TensorError::LengthMismatch`] if `data.len() != B·S·H·D`.
    pub fn new(shape: TensorShape, data: Vec<f64>) -> Result<Self, TensorError> {
        let expected = Self::checked_len(shape)?;
        if data.len() != expected {
            return Err(TensorError::LengthMismatch {
                expected,
                actual: data.len(),
            });
        }
        Ok(Tensor4 { shape, data })
    }

    /// Creates a tensor with all values 0.
    ///
    /// # Errors
    ///
    /// [`TensorError::ZeroExtent`] or [`TensorError::TooManyElements`].
    pub fn zeros(shape: TensorShape) -> Result<Self, TensorError> {
        let len = Self::checked_len(shape)?;
        Ok(Tensor4 {
            shape,
            data: vec![0.0; len],
        })
    }

    /// Creates a tensor and sets each element `(b, s, h, d)` to
    /// `f(b, s, h, d)`.
    ///
    /// `f` is any closure (or function) that takes four `usize` values and
    /// returns an `f64`. `impl FnMut(...)` means: the compiler generates a
    /// separate version of `from_fn` for each closure type (no dynamic
    /// dispatch). `FnMut` allows the closure to change captured variables,
    /// for example a counter.
    ///
    /// The function calls `f` in memory order, so it fills the `Vec` with
    /// sequential `push` operations into exactly one allocation.
    ///
    /// # Errors
    ///
    /// [`TensorError::ZeroExtent`] or [`TensorError::TooManyElements`].
    pub fn from_fn(
        shape: TensorShape,
        mut f: impl FnMut(usize, usize, usize, usize) -> f64,
    ) -> Result<Self, TensorError> {
        let len = Self::checked_len(shape)?;
        let mut data = Vec::with_capacity(len);
        for b in 0..shape.batch {
            for s in 0..shape.sequence {
                for h in 0..shape.heads {
                    for d in 0..shape.features {
                        data.push(f(b, s, h, d));
                    }
                }
            }
        }
        Ok(Tensor4 { shape, data })
    }

    /// Returns the shape.
    pub fn shape(&self) -> TensorShape {
        self.shape
    }

    /// Returns the flat index `((b·S + s)·H + h)·D + d` of the element
    /// `(b, s, h, d)`, or `None` if an index is out of range.
    ///
    /// The checks prevent a wrong but in-range result. Example: with
    /// `D = 4`, the element `(0, 0, 0, 5)` does not exist, but the formula
    /// would give 5, which is the valid index of `(0, 0, 1, 1)`.
    pub fn index(&self, b: usize, s: usize, h: usize, d: usize) -> Option<usize> {
        let TensorShape {
            batch,
            sequence,
            heads,
            features,
        } = self.shape;
        if b < batch && s < sequence && h < heads && d < features {
            Some(((b * sequence + s) * heads + h) * features + d)
        } else {
            None
        }
    }

    /// Returns a copy of the element `(b, s, h, d)`, or `None` if out of
    /// range.
    pub fn get(&self, b: usize, s: usize, h: usize, d: usize) -> Option<f64> {
        self.index(b, s, h, d).map(|i| self.data[i])
    }

    /// Returns a **mutable reference** to the element `(b, s, h, d)`, or
    /// `None` if out of range.
    ///
    /// While the reference exists, the tensor is mutably borrowed: no other
    /// code can read or change it.
    ///
    /// ```
    /// use rust_rope::math::tensor::{Tensor4, TensorShape};
    ///
    /// let mut t = Tensor4::zeros(TensorShape::new(1, 1, 1, 2))?;
    /// if let Some(x) = t.get_mut(0, 0, 0, 1) {
    ///     *x = 7.5;               // write through the reference
    /// }
    /// assert_eq!(t.as_slice(), &[0.0, 7.5]);
    /// # Ok::<(), rust_rope::math::tensor::TensorError>(())
    /// ```
    pub fn get_mut(&mut self, b: usize, s: usize, h: usize, d: usize) -> Option<&mut f64> {
        let i = self.index(b, s, h, d)?;
        self.data.get_mut(i)
    }

    /// Returns the `D` features of `(b, s, h)` as one contiguous slice, or
    /// `None` if out of range.
    pub fn vector(&self, b: usize, s: usize, h: usize) -> Option<&[f64]> {
        let start = self.index(b, s, h, 0)?;
        Some(&self.data[start..start + self.shape.features])
    }

    /// Returns the `D` features of `(b, s, h)` as a mutable slice, or `None`
    /// if out of range.
    pub fn vector_mut(&mut self, b: usize, s: usize, h: usize) -> Option<&mut [f64]> {
        let start = self.index(b, s, h, 0)?;
        let features = self.shape.features;
        Some(&mut self.data[start..start + features])
    }

    /// Returns all values as one flat row-major slice.
    pub fn as_slice(&self) -> &[f64] {
        &self.data
    }

    /// Returns all values as one flat, mutable row-major slice.
    ///
    /// The slice has a fixed length. A caller can change values, but cannot
    /// add or remove values. Thus the shape stays correct.
    pub fn as_mut_slice(&mut self) -> &mut [f64] {
        &mut self.data
    }

    /// Consumes the tensor and returns its flat data. Nothing is copied.
    pub fn into_vec(self) -> Vec<f64> {
        self.data
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shape() -> TensorShape {
        TensorShape::new(2, 3, 4, 5)
    }

    #[test]
    fn index_formula_matches_memory_order() {
        // `from_fn` fills in memory order, so the k-th value written is k.
        let mut counter = 0.0;
        let t = Tensor4::from_fn(shape(), |_, _, _, _| {
            counter += 1.0;
            counter - 1.0
        })
        .unwrap();
        for b in 0..2 {
            for s in 0..3 {
                for h in 0..4 {
                    for d in 0..5 {
                        let expected = ((b * 3 + s) * 4 + h) * 5 + d;
                        assert_eq!(t.index(b, s, h, d), Some(expected));
                        assert_eq!(t.get(b, s, h, d), Some(expected as f64));
                    }
                }
            }
        }
    }

    #[test]
    fn last_index_is_length_minus_one() {
        let t = Tensor4::zeros(shape()).unwrap();
        assert_eq!(t.index(1, 2, 3, 4), Some(2 * 3 * 4 * 5 - 1));
        assert_eq!(t.as_slice().len(), 120);
    }

    #[test]
    fn out_of_range_indices_return_none() {
        let t = Tensor4::zeros(shape()).unwrap();
        assert_eq!(t.index(2, 0, 0, 0), None);
        assert_eq!(t.index(0, 3, 0, 0), None);
        assert_eq!(t.index(0, 0, 4, 0), None);
        // In range for the formula (result 5), but d = 5 is not a valid feature.
        assert_eq!(t.index(0, 0, 0, 5), None);
        assert_eq!(t.vector(0, 0, 4), None);
    }

    #[test]
    fn construction_errors() {
        assert_eq!(
            Tensor4::new(TensorShape::new(1, 2, 1, 2), vec![0.0; 3]),
            Err(TensorError::LengthMismatch {
                expected: 4,
                actual: 3
            })
        );
        assert_eq!(
            Tensor4::zeros(TensorShape::new(1, 0, 1, 2)),
            Err(TensorError::ZeroExtent { axis: "sequence" })
        );
        assert_eq!(
            Tensor4::zeros(TensorShape::new(usize::MAX, 2, 1, 1)),
            Err(TensorError::TooManyElements)
        );
    }

    #[test]
    fn vector_is_a_contiguous_head_slice() {
        let t = Tensor4::from_fn(shape(), |b, s, h, d| {
            (1000 * b + 100 * s + 10 * h + d) as f64
        })
        .unwrap();
        assert_eq!(
            t.vector(1, 2, 3).unwrap(),
            &[1230.0, 1231.0, 1232.0, 1233.0, 1234.0]
        );
    }

    #[test]
    fn mutable_access_changes_only_the_selected_values() {
        let mut t = Tensor4::zeros(TensorShape::new(1, 2, 2, 2)).unwrap();
        *t.get_mut(0, 1, 0, 1).unwrap() = 3.0;
        t.vector_mut(0, 0, 1).unwrap().copy_from_slice(&[1.0, 2.0]);
        assert_eq!(t.as_slice(), &[0.0, 0.0, 1.0, 2.0, 0.0, 3.0, 0.0, 0.0]);
        t.as_mut_slice()[0] = -1.0;
        assert_eq!(t.into_vec()[0], -1.0);
    }
}
