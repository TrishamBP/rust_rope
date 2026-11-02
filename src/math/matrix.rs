//! Matrix and vector operations.
//!
//! ## Implemented
//!
//! - [`Vector`]: an `n`-dimensional vector of `f64` values, stored on the
//!   heap. Operations: dimension, addition, subtraction, dot product and
//!   magnitude (Euclidean norm).
//! - [`DimensionMismatch`]: the error for operations on two vectors with
//!   different dimensions.
//! - [`Vector2`]: a two-dimensional vector `[x, y]`, stored inline as
//!   `[f64; 2]`.
//! - [`Matrix2`]: a 2 × 2 matrix. Operations: construction, element access,
//!   transpose, determinant, matrix-vector multiplication, matrix-matrix
//!   multiplication and the rotation matrix `R(θ)`.
//! - [`rotate_vector`]: rotates a [`Vector2`] around the origin by `θ`.
//!
//! ## Notation
//!
//! A vector with `n` components is `v = [v₁, v₂, ..., vₙ]`.
//! `n` is the **dimension** of the vector.
//!
//! A 2 × 2 matrix has 2 rows and 2 columns. `mᵣ꜀` is the element in row `r`
//! and column `c`:
//!
//! ```text
//! M = [ m₁₁  m₁₂ ]
//!     [ m₂₁  m₂₂ ]
//! ```
//!
//! In Rust, rows and columns count from 0: `m₁₁` is `get(0, 0)`.

use crate::math::trigonometry::sin_cos;
use std::fmt;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// The error for an operation on two vectors with different dimensions.
///
/// Addition, subtraction and the dot product are defined only for two
/// vectors with the same dimension `n`. If the dimensions are different,
/// the operation returns this error. It does not panic, because the caller
/// supplies the vectors and the caller can recover from the error.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DimensionMismatch {
    /// The dimension of the left operand (`self`).
    pub left: usize,
    /// The dimension of the right operand (`other`).
    pub right: usize,
}

impl fmt::Display for DimensionMismatch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "dimension mismatch: the left vector has {} components, the right vector has {}",
            self.left, self.right
        )
    }
}

// An empty `impl` is sufficient: all methods of `std::error::Error` have
// default implementations. This makes `DimensionMismatch` usable with `?`
// in functions that return `Box<dyn std::error::Error>`.
impl std::error::Error for DimensionMismatch {}

// ---------------------------------------------------------------------------
// Vector
// ---------------------------------------------------------------------------

/// An `n`-dimensional vector `v = [v₁, v₂, ..., vₙ]` of `f64` values.
///
/// The components are stored in a `Vec<f64>`. The `Vec` stores its data on
/// the heap, so the dimension can be selected at run time. RoPE needs this,
/// because the embedding dimension `d` is a configuration value (for
/// example 64 or 128).
///
/// # Example
///
/// ```
/// use rust_rope::math::matrix::Vector;
///
/// let a = Vector::new(vec![1.0, 2.0, 3.0]);
/// let b = Vector::from_slice(&[4.0, 5.0, 6.0]);
///
/// assert_eq!(a.dimension(), 3);
/// assert_eq!(a.dot(&b), Ok(32.0)); // 1·4 + 2·5 + 3·6
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct Vector {
    components: Vec<f64>,
}

impl Vector {
    /// Creates a vector that takes ownership of `components`.
    ///
    /// This function **moves** the `Vec` into the new `Vector`. It does not
    /// copy the data and it does not allocate.
    pub fn new(components: Vec<f64>) -> Self {
        Self { components }
    }

    /// Creates a vector from a borrowed slice of values.
    ///
    /// This function **copies** the values into a new heap allocation. The
    /// caller keeps ownership of the original data.
    pub fn from_slice(values: &[f64]) -> Self {
        Self {
            components: values.to_vec(),
        }
    }

    /// Returns the dimension `n` (the number of components).
    pub fn dimension(&self) -> usize {
        self.components.len()
    }

    /// Returns `true` if the vector has no components (`n = 0`).
    pub fn is_empty(&self) -> bool {
        self.components.is_empty()
    }

    /// Returns a read-only view of the components.
    ///
    /// The return type `&[f64]` is a **borrowed slice**. It points to the
    /// heap data of the vector. No data is copied.
    pub fn components(&self) -> &[f64] {
        &self.components
    }

    /// Consumes the vector and returns the internal `Vec<f64>`.
    ///
    /// The `self` parameter (not `&self`) **moves** the vector into this
    /// method. After the call, the caller cannot use the vector again. The
    /// heap buffer moves to the caller without a copy.
    pub fn into_components(self) -> Vec<f64> {
        self.components
    }

    /// Returns an error if `self` and `other` have different dimensions.
    fn check_same_dimension(&self, other: &Vector) -> Result<(), DimensionMismatch> {
        if self.dimension() == other.dimension() {
            Ok(())
        } else {
            Err(DimensionMismatch {
                left: self.dimension(),
                right: other.dimension(),
            })
        }
    }

    /// Calculates the sum `a + b = [a₁ + b₁, a₂ + b₂, ..., aₙ + bₙ]`.
    ///
    /// Both operands are **borrowed**. The result is a new vector with a new
    /// heap allocation.
    ///
    /// # Errors
    ///
    /// Returns [`DimensionMismatch`] if the dimensions are different.
    pub fn checked_add(&self, other: &Vector) -> Result<Vector, DimensionMismatch> {
        self.check_same_dimension(other)?;
        let components = self
            .components
            .iter()
            .zip(&other.components)
            .map(|(a, b)| a + b)
            .collect();
        Ok(Vector { components })
    }

    /// Calculates the difference `a − b = [a₁ − b₁, a₂ − b₂, ..., aₙ − bₙ]`.
    ///
    /// # Errors
    ///
    /// Returns [`DimensionMismatch`] if the dimensions are different.
    pub fn checked_sub(&self, other: &Vector) -> Result<Vector, DimensionMismatch> {
        self.check_same_dimension(other)?;
        let components = self
            .components
            .iter()
            .zip(&other.components)
            .map(|(a, b)| a - b)
            .collect();
        Ok(Vector { components })
    }

    /// Calculates the dot product `a · b = Σᵢ aᵢ·bᵢ`.
    ///
    /// The dot product of two empty vectors is `0.0` (an empty sum).
    /// This function does not allocate.
    ///
    /// # Errors
    ///
    /// Returns [`DimensionMismatch`] if the dimensions are different.
    pub fn dot(&self, other: &Vector) -> Result<f64, DimensionMismatch> {
        self.check_same_dimension(other)?;
        Ok(self
            .components
            .iter()
            .zip(&other.components)
            .map(|(a, b)| a * b)
            .sum())
    }

    /// Calculates the magnitude (Euclidean norm) `‖v‖ = √(Σᵢ vᵢ²)`.
    ///
    /// The magnitude of an empty vector is `0.0`.
    /// This function does not allocate.
    pub fn magnitude(&self) -> f64 {
        self.components.iter().map(|x| x * x).sum::<f64>().sqrt()
    }
}

// ---------------------------------------------------------------------------
// Two-dimensional vectors and 2 × 2 matrices
// ---------------------------------------------------------------------------

/// A two-dimensional vector `[x, y]`.
///
/// This is a **type alias**: `Vector2` is another name for `[f64; 2]`. The
/// dimension 2 is part of the type, so the compiler checks it. A
/// `Vector2` lives inline (on the stack or in registers). It needs no heap
/// allocation and no runtime dimension check.
///
/// For general `n`-dimensional vectors, use [`Vector`].
pub type Vector2 = [f64; 2];

/// A 2 × 2 matrix of `f64` values, stored row by row.
///
/// ```text
/// M = [ m₁₁  m₁₂ ]     stored as  [[m₁₁, m₁₂], [m₂₁, m₂₂]]
///     [ m₂₁  m₂₂ ]
/// ```
///
/// The four values are stored inline: 32 bytes, no heap. The type is
/// `Copy`.
///
/// # Example
///
/// ```
/// use rust_rope::math::matrix::Matrix2;
///
/// let m = Matrix2::new(1.0, 2.0,
///                      3.0, 4.0);
/// // [1 2] [5]   [1·5 + 2·6]   [17]
/// // [3 4] [6] = [3·5 + 4·6] = [39]
/// assert_eq!(m.mul_vector([5.0, 6.0]), [17.0, 39.0]);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Matrix2 {
    rows: [[f64; 2]; 2],
}

impl Matrix2 {
    /// The identity matrix `I = [[1, 0], [0, 1]]`. `I·v = v` for every `v`.
    pub const IDENTITY: Matrix2 = Matrix2::new(1.0, 0.0, 0.0, 1.0);

    /// Creates the matrix `[[m11, m12], [m21, m22]]`.
    ///
    /// The arguments are in **row-major** order: first the complete first
    /// row, then the complete second row.
    pub const fn new(m11: f64, m12: f64, m21: f64, m22: f64) -> Self {
        Self {
            rows: [[m11, m12], [m21, m22]],
        }
    }

    /// Creates a matrix from an array of two rows.
    pub const fn from_rows(rows: [[f64; 2]; 2]) -> Self {
        Self { rows }
    }

    /// Returns the two rows as an array (a copy of the data).
    pub const fn rows(self) -> [[f64; 2]; 2] {
        self.rows
    }

    /// Returns the element in row `row` and column `col` (both start at 0).
    ///
    /// # Panics
    ///
    /// Panics if `row > 1` or `col > 1`. An index out of range is a bug in
    /// the calling code, because a 2 × 2 matrix always has the indices 0
    /// and 1.
    pub const fn get(self, row: usize, col: usize) -> f64 {
        self.rows[row][col]
    }

    /// Returns the transpose `Mᵀ`: rows become columns.
    ///
    /// ```text
    /// [ a  b ]ᵀ   [ a  c ]
    /// [ c  d ]  = [ b  d ]
    /// ```
    ///
    /// For a rotation matrix, the transpose is the inverse rotation:
    /// `R(θ)ᵀ = R(−θ)`.
    pub const fn transpose(self) -> Self {
        let [[a, b], [c, d]] = self.rows;
        Self::new(a, c, b, d)
    }

    /// Returns the determinant `det(M) = a·d − b·c` of `M = [[a, b], [c, d]]`.
    ///
    /// Geometrically, `|det(M)|` is the factor by which `M` scales areas.
    /// A negative determinant means that `M` also reflects (it changes the
    /// orientation). A rotation matrix has `det(R(θ)) = 1`.
    pub fn determinant(self) -> f64 {
        let [[a, b], [c, d]] = self.rows;
        a * d - b * c
    }

    /// Calculates the matrix-vector product `M·v`.
    ///
    /// ```text
    /// [ a  b ] [ x ]   [ a·x + b·y ]
    /// [ c  d ] [ y ] = [ c·x + d·y ]
    /// ```
    ///
    /// Each component of the result is the dot product of one row of `M`
    /// with `v`.
    pub fn mul_vector(self, v: Vector2) -> Vector2 {
        let [[a, b], [c, d]] = self.rows;
        let [x, y] = v;
        [a * x + b * y, c * x + d * y]
    }

    /// Calculates the matrix-matrix product `M·N`.
    ///
    /// ```text
    /// [ a  b ] [ e  f ]   [ a·e + b·g   a·f + b·h ]
    /// [ c  d ] [ g  h ] = [ c·e + d·g   c·f + d·h ]
    /// ```
    ///
    /// The element in row `r` and column `c` of the result is the dot
    /// product of row `r` of `M` with column `c` of `N`.
    ///
    /// The order is important: in general, `M·N ≠ N·M`. The product `M·N`
    /// applied to a vector means "first apply `N`, then apply `M`".
    pub fn mul_matrix(self, other: Matrix2) -> Matrix2 {
        let [[a, b], [c, d]] = self.rows;
        let [[e, f], [g, h]] = other.rows;
        Matrix2::new(a * e + b * g, a * f + b * h, c * e + d * g, c * f + d * h)
    }

    /// Creates the rotation matrix for the angle `θ` (in radians).
    ///
    /// ```text
    /// R(θ) = [ cos(θ)  −sin(θ) ]
    ///        [ sin(θ)   cos(θ) ]
    /// ```
    ///
    /// `R(θ)·v` rotates the vector `v` around the origin by `θ`,
    /// counterclockwise for `θ > 0`. The rotation keeps the length of `v`.
    pub fn rotation(theta: f64) -> Matrix2 {
        let (sin, cos) = sin_cos(theta);
        Matrix2::rotation_from_sin_cos(sin, cos)
    }

    /// Creates the rotation matrix from the values `sin(θ)` and `cos(θ)`
    /// that the caller already has (Stage 9).
    ///
    /// ```text
    /// R = [ cos  −sin ]
    ///     [ sin   cos ]
    /// ```
    ///
    /// This function does no trigonometry. A caller that keeps a table of
    /// `sin` and `cos` values (a cache) uses it to skip the expensive
    /// `sin_cos` call. [`Matrix2::rotation`] also uses it, so the matrix
    /// layout exists in one place only.
    ///
    /// The function does not check that `sin² + cos² = 1`. Values that are
    /// not from one angle give a matrix that is not a rotation.
    #[inline]
    pub const fn rotation_from_sin_cos(sin: f64, cos: f64) -> Matrix2 {
        Matrix2::new(cos, -sin, sin, cos)
    }
}

/// Rotates the vector `v` around the origin by the angle `θ` (in radians).
///
/// This calculates `R(θ)·v`:
///
/// ```text
/// x' = x·cos(θ) − y·sin(θ)
/// y' = x·sin(θ) + y·cos(θ)
/// ```
///
/// # Example
///
/// ```
/// use rust_rope::math::matrix::rotate_vector;
/// use rust_rope::math::{approx_eq, DEFAULT_TOLERANCE};
/// use std::f64::consts::FRAC_PI_2;
///
/// // A quarter turn moves (1, 0) to (0, 1).
/// let [x, y] = rotate_vector([1.0, 0.0], FRAC_PI_2);
/// assert!(approx_eq(x, 0.0, DEFAULT_TOLERANCE));
/// assert!(approx_eq(y, 1.0, DEFAULT_TOLERANCE));
/// ```
pub fn rotate_vector(v: Vector2, theta: f64) -> Vector2 {
    Matrix2::rotation(theta).mul_vector(v)
}

/// Rotates the vector `v` with precomputed `sin(θ)` and `cos(θ)` values
/// (Stage 9).
///
/// The result is **bit-identical** to `rotate_vector(v, θ)` when
/// `(sin, cos) == sin_cos(θ)`: both functions build the same matrix and do
/// the same 4 multiplications and 2 additions in the same order.
///
/// # Example
///
/// ```
/// use rust_rope::math::matrix::{rotate_vector, rotate_vector_sin_cos};
///
/// let theta = 0.7_f64;
/// let (sin, cos) = theta.sin_cos();
/// assert_eq!(rotate_vector_sin_cos([3.0, -1.0], sin, cos), rotate_vector([3.0, -1.0], theta));
/// ```
#[inline]
pub fn rotate_vector_sin_cos(v: Vector2, sin: f64, cos: f64) -> Vector2 {
    Matrix2::rotation_from_sin_cos(sin, cos).mul_vector(v)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::{DEFAULT_TOLERANCE, approx_eq};

    // --- Construction and access -------------------------------------------

    #[test]
    fn new_moves_values_and_keeps_order() {
        let v = Vector::new(vec![1.0, -2.0, 3.5]);
        assert_eq!(v.dimension(), 3);
        assert_eq!(v.components(), &[1.0, -2.0, 3.5]);
        assert!(!v.is_empty());
    }

    #[test]
    fn from_slice_copies_values() {
        let data = [4.0, 5.0];
        let v = Vector::from_slice(&data);
        // `data` is still usable: `from_slice` only borrowed it.
        assert_eq!(data.len(), 2);
        assert_eq!(v.components(), &data);
    }

    #[test]
    fn empty_vector() {
        let v = Vector::new(Vec::new());
        assert_eq!(v.dimension(), 0);
        assert!(v.is_empty());
        assert_eq!(v.magnitude(), 0.0);
        assert_eq!(v.dot(&Vector::new(vec![])), Ok(0.0));
    }

    #[test]
    fn into_components_returns_the_same_values() {
        let v = Vector::new(vec![7.0, 8.0]);
        assert_eq!(v.into_components(), vec![7.0, 8.0]);
    }

    // --- Addition and subtraction ------------------------------------------

    #[test]
    fn add_normal_and_negative_values() {
        let a = Vector::new(vec![1.0, 2.0, 3.0]);
        let b = Vector::new(vec![4.0, -5.0, -3.0]);
        let sum = a.checked_add(&b).unwrap();
        assert_eq!(sum.components(), &[5.0, -3.0, 0.0]);
    }

    #[test]
    fn subtract_normal_and_negative_values() {
        let a = Vector::new(vec![1.0, 2.0, 3.0]);
        let b = Vector::new(vec![4.0, -5.0, 3.0]);
        let difference = a.checked_sub(&b).unwrap();
        assert_eq!(difference.components(), &[-3.0, 7.0, 0.0]);
    }

    #[test]
    fn adding_the_zero_vector_changes_nothing() {
        let a = Vector::new(vec![1.5, -2.5]);
        let zero = Vector::new(vec![0.0, 0.0]);
        assert_eq!(a.checked_add(&zero).unwrap(), a);
    }

    #[test]
    fn subtracting_a_vector_from_itself_gives_zero() {
        let a = Vector::new(vec![1.5, -2.5, 1e10]);
        let difference = a.checked_sub(&a).unwrap();
        assert_eq!(difference.components(), &[0.0, 0.0, 0.0]);
    }

    #[test]
    fn floating_point_addition_needs_a_tolerance() {
        let a = Vector::new(vec![0.1]);
        let b = Vector::new(vec![0.2]);
        let sum = a.checked_add(&b).unwrap().components()[0];
        assert_ne!(sum, 0.3); // 0.30000000000000004
        assert!(approx_eq(sum, 0.3, DEFAULT_TOLERANCE));
    }

    // --- Dot product --------------------------------------------------------

    #[test]
    fn dot_product_of_normal_values() {
        let a = Vector::new(vec![1.0, 2.0, 3.0]);
        let b = Vector::new(vec![4.0, 5.0, 6.0]);
        assert_eq!(a.dot(&b), Ok(32.0));
    }

    #[test]
    fn dot_product_with_negative_values() {
        let a = Vector::new(vec![1.0, -2.0]);
        let b = Vector::new(vec![-3.0, 4.0]);
        assert_eq!(a.dot(&b), Ok(-11.0)); // 1·(−3) + (−2)·4
    }

    #[test]
    fn dot_product_of_perpendicular_vectors_is_zero() {
        let x_axis = Vector::new(vec![1.0, 0.0]);
        let y_axis = Vector::new(vec![0.0, 1.0]);
        assert_eq!(x_axis.dot(&y_axis), Ok(0.0));
    }

    #[test]
    fn dot_product_is_symmetric() {
        let a = Vector::new(vec![0.3, -1.7, 2.2]);
        let b = Vector::new(vec![4.1, 0.5, -0.9]);
        let ab = a.dot(&b).unwrap();
        let ba = b.dot(&a).unwrap();
        assert!(approx_eq(ab, ba, DEFAULT_TOLERANCE));
    }

    #[test]
    fn dot_product_with_itself_is_magnitude_squared() {
        let v = Vector::new(vec![1.0, 2.0, 2.0]);
        let m = v.magnitude();
        assert!(approx_eq(v.dot(&v).unwrap(), m * m, DEFAULT_TOLERANCE));
    }

    // --- Magnitude ----------------------------------------------------------

    #[test]
    fn magnitude_of_three_four_is_five() {
        let v = Vector::new(vec![3.0, 4.0]);
        assert_eq!(v.magnitude(), 5.0);
    }

    #[test]
    fn magnitude_ignores_signs() {
        let v = Vector::new(vec![-3.0, -4.0]);
        assert_eq!(v.magnitude(), 5.0);
    }

    #[test]
    fn magnitude_of_zero_vector_is_zero() {
        assert_eq!(Vector::new(vec![0.0, 0.0, 0.0]).magnitude(), 0.0);
    }

    #[test]
    fn magnitude_of_irrational_result() {
        // ‖[1, 1]‖ = √2. √2 is irrational, so f64 can only approximate it.
        let v = Vector::new(vec![1.0, 1.0]);
        assert!(approx_eq(
            v.magnitude(),
            std::f64::consts::SQRT_2,
            DEFAULT_TOLERANCE
        ));
    }

    #[test]
    fn magnitude_overflows_for_very_large_components() {
        // 1e200² = 1e400 is larger than f64::MAX (about 1.8e308).
        // The naive formula overflows to infinity. This test documents the
        // limitation. See explain.md, Entry 2, section 8.
        let v = Vector::new(vec![1e200, 1e200]);
        assert!(v.magnitude().is_infinite());
    }

    // --- Dimension mismatch -------------------------------------------------

    #[test]
    fn mismatched_dimensions_return_an_error() {
        let a = Vector::new(vec![1.0, 2.0, 3.0]);
        let b = Vector::new(vec![1.0, 2.0]);
        let expected = Err(DimensionMismatch { left: 3, right: 2 });
        assert_eq!(a.checked_add(&b), expected);
        assert_eq!(a.checked_sub(&b), expected);
        assert_eq!(a.dot(&b), Err(DimensionMismatch { left: 3, right: 2 }));
        // The order of the operands is kept in the error.
        assert_eq!(b.dot(&a), Err(DimensionMismatch { left: 2, right: 3 }));
    }

    #[test]
    fn dimension_mismatch_has_a_readable_message() {
        let error = DimensionMismatch { left: 3, right: 2 };
        assert_eq!(
            error.to_string(),
            "dimension mismatch: the left vector has 3 components, the right vector has 2"
        );
    }

    // =======================================================================
    // Matrix2 and rotations
    // =======================================================================

    use std::f64::consts::{FRAC_PI_2, FRAC_PI_4, PI, TAU};

    fn assert_vector2_eq(actual: Vector2, expected: Vector2) {
        assert!(
            approx_eq(actual[0], expected[0], DEFAULT_TOLERANCE)
                && approx_eq(actual[1], expected[1], DEFAULT_TOLERANCE),
            "expected {expected:?}, got {actual:?}"
        );
    }

    fn assert_matrix2_eq(actual: Matrix2, expected: Matrix2) {
        for row in 0..2 {
            for col in 0..2 {
                assert!(
                    approx_eq(
                        actual.get(row, col),
                        expected.get(row, col),
                        DEFAULT_TOLERANCE
                    ),
                    "expected {expected:?}, got {actual:?}"
                );
            }
        }
    }

    /// The length of a 2-D vector, calculated with the existing `Vector`
    /// type (no duplicate norm function).
    fn length(v: Vector2) -> f64 {
        Vector::from_slice(&v).magnitude()
    }

    // --- Construction and access -------------------------------------------

    #[test]
    fn new_uses_row_major_order() {
        let m = Matrix2::new(1.0, 2.0, 3.0, 4.0);
        assert_eq!(m.get(0, 0), 1.0);
        assert_eq!(m.get(0, 1), 2.0);
        assert_eq!(m.get(1, 0), 3.0);
        assert_eq!(m.get(1, 1), 4.0);
        assert_eq!(m.rows(), [[1.0, 2.0], [3.0, 4.0]]);
        assert_eq!(Matrix2::from_rows([[1.0, 2.0], [3.0, 4.0]]), m);
    }

    #[test]
    #[should_panic]
    fn get_out_of_range_panics() {
        Matrix2::IDENTITY.get(2, 0);
    }

    #[test]
    fn transpose_swaps_off_diagonal_elements() {
        let m = Matrix2::new(1.0, 2.0, 3.0, 4.0);
        assert_eq!(m.transpose(), Matrix2::new(1.0, 3.0, 2.0, 4.0));
        assert_eq!(m.transpose().transpose(), m);
    }

    #[test]
    fn determinant_of_known_matrices() {
        assert_eq!(Matrix2::new(1.0, 2.0, 3.0, 4.0).determinant(), -2.0);
        assert_eq!(Matrix2::IDENTITY.determinant(), 1.0);
        // A matrix with two equal rows has determinant 0.
        assert_eq!(Matrix2::new(2.0, 3.0, 2.0, 3.0).determinant(), 0.0);
    }

    // --- Matrix-vector multiplication --------------------------------------

    #[test]
    fn matrix_vector_product_of_normal_values() {
        let m = Matrix2::new(1.0, 2.0, 3.0, 4.0);
        assert_eq!(m.mul_vector([5.0, 6.0]), [17.0, 39.0]);
    }

    #[test]
    fn matrix_vector_product_with_negative_values() {
        let m = Matrix2::new(-1.0, 2.0, 0.5, -3.0);
        // [−1·4 + 2·(−2), 0.5·4 + (−3)·(−2)] = [−8, 8]
        assert_eq!(m.mul_vector([4.0, -2.0]), [-8.0, 8.0]);
    }

    #[test]
    fn matrix_vector_product_with_zeros() {
        let m = Matrix2::new(1.0, 2.0, 3.0, 4.0);
        assert_eq!(m.mul_vector([0.0, 0.0]), [0.0, 0.0]);
        let zero = Matrix2::new(0.0, 0.0, 0.0, 0.0);
        assert_eq!(zero.mul_vector([5.0, -6.0]), [0.0, 0.0]);
    }

    #[test]
    fn identity_does_not_change_a_vector() {
        assert_eq!(Matrix2::IDENTITY.mul_vector([3.5, -7.0]), [3.5, -7.0]);
    }

    // --- Matrix-matrix multiplication --------------------------------------

    #[test]
    fn matrix_matrix_product_of_normal_values() {
        let m = Matrix2::new(1.0, 2.0, 3.0, 4.0);
        let n = Matrix2::new(5.0, 6.0, 7.0, 8.0);
        // [1·5 + 2·7, 1·6 + 2·8]   [19, 22]
        // [3·5 + 4·7, 3·6 + 4·8] = [43, 50]
        assert_eq!(m.mul_matrix(n), Matrix2::new(19.0, 22.0, 43.0, 50.0));
    }

    #[test]
    fn matrix_multiplication_is_not_commutative() {
        let m = Matrix2::new(1.0, 2.0, 3.0, 4.0);
        let n = Matrix2::new(5.0, 6.0, 7.0, 8.0);
        assert_eq!(n.mul_matrix(m), Matrix2::new(23.0, 34.0, 31.0, 46.0));
        assert_ne!(m.mul_matrix(n), n.mul_matrix(m));
    }

    #[test]
    fn identity_is_the_neutral_element() {
        let m = Matrix2::new(1.0, -2.0, 3.5, 0.0);
        assert_eq!(m.mul_matrix(Matrix2::IDENTITY), m);
        assert_eq!(Matrix2::IDENTITY.mul_matrix(m), m);
    }

    #[test]
    fn product_matrix_applies_right_factor_first() {
        // (M·N)·v = M·(N·v)
        let m = Matrix2::new(1.0, 2.0, 3.0, 4.0);
        let n = Matrix2::new(0.0, -1.0, 1.0, 0.0);
        let v = [2.0, -5.0];
        assert_eq!(m.mul_matrix(n).mul_vector(v), m.mul_vector(n.mul_vector(v)));
    }

    // --- Rotation at the four quarter turns --------------------------------

    #[test]
    fn rotation_by_zero_is_the_identity() {
        assert_eq!(Matrix2::rotation(0.0), Matrix2::IDENTITY);
        assert_eq!(rotate_vector([3.0, -4.0], 0.0), [3.0, -4.0]);
    }

    #[test]
    fn rotation_by_a_quarter_turn() {
        // R(π/2) = [[0, −1], [1, 0]]: (x, y) → (−y, x)
        assert_matrix2_eq(
            Matrix2::rotation(FRAC_PI_2),
            Matrix2::new(0.0, -1.0, 1.0, 0.0),
        );
        assert_vector2_eq(rotate_vector([1.0, 0.0], FRAC_PI_2), [0.0, 1.0]);
        assert_vector2_eq(rotate_vector([2.0, 3.0], FRAC_PI_2), [-3.0, 2.0]);
    }

    #[test]
    fn rotation_by_a_half_turn() {
        // R(π) = −I: (x, y) → (−x, −y)
        assert_matrix2_eq(Matrix2::rotation(PI), Matrix2::new(-1.0, 0.0, 0.0, -1.0));
        assert_vector2_eq(rotate_vector([1.0, 0.0], PI), [-1.0, 0.0]);
        assert_vector2_eq(rotate_vector([2.0, 3.0], PI), [-2.0, -3.0]);
    }

    #[test]
    fn rotation_by_three_quarter_turns() {
        // R(3π/2) = [[0, 1], [−1, 0]]: (x, y) → (y, −x)
        let theta = 3.0 * FRAC_PI_2;
        assert_matrix2_eq(Matrix2::rotation(theta), Matrix2::new(0.0, 1.0, -1.0, 0.0));
        assert_vector2_eq(rotate_vector([1.0, 0.0], theta), [0.0, -1.0]);
        assert_vector2_eq(rotate_vector([2.0, 3.0], theta), [3.0, -2.0]);
    }

    #[test]
    fn quarter_turn_result_is_not_exact() {
        // The exact result is (0, 1). The x component is about 6.1e-17,
        // because cos(π/2) in f64 is not exactly 0.
        let [x, y] = rotate_vector([1.0, 0.0], FRAC_PI_2);
        assert_ne!(x, 0.0);
        assert!(x.abs() < 1e-16);
        assert_eq!(y, 1.0);
    }

    #[test]
    fn rotation_at_45_degrees() {
        let s = std::f64::consts::FRAC_1_SQRT_2;
        assert_vector2_eq(rotate_vector([1.0, 0.0], FRAC_PI_4), [s, s]);
    }

    #[test]
    fn negative_angle_rotates_clockwise() {
        assert_vector2_eq(rotate_vector([1.0, 0.0], -FRAC_PI_2), [0.0, -1.0]);
    }

    #[test]
    fn rotating_the_zero_vector_gives_zero() {
        assert_eq!(rotate_vector([0.0, 0.0], 1.234), [0.0, 0.0]);
    }

    // --- Properties of rotation matrices -----------------------------------

    #[test]
    fn rotation_keeps_the_length() {
        for theta in [-2.0, -0.3, 0.0, 0.7, FRAC_PI_2, 2.5, PI, 100.0] {
            for v in [[1.0, 0.0], [3.0, 4.0], [-2.5, 0.1], [1e-3, -7.0]] {
                let rotated = rotate_vector(v, theta);
                assert!(approx_eq(length(rotated), length(v), DEFAULT_TOLERANCE));
            }
        }
    }

    #[test]
    fn rotation_keeps_dot_products() {
        // ⟨R·a, R·b⟩ = ⟨a, b⟩: a rotation keeps angles between vectors.
        let a = [1.0, 2.0];
        let b = [-3.0, 0.5];
        let dot = |u: Vector2, w: Vector2| Vector::from_slice(&u).dot(&Vector::from_slice(&w));
        for theta in [0.4, 1.0, 3.0, -2.2] {
            let rotated = dot(rotate_vector(a, theta), rotate_vector(b, theta)).unwrap();
            assert!(approx_eq(rotated, dot(a, b).unwrap(), DEFAULT_TOLERANCE));
        }
    }

    #[test]
    fn rotation_matrix_has_determinant_one() {
        for theta in [-1.0, 0.0, 0.5, 2.0, 4.0] {
            assert!(approx_eq(
                Matrix2::rotation(theta).determinant(),
                1.0,
                DEFAULT_TOLERANCE
            ));
        }
    }

    #[test]
    fn rotation_matrix_is_orthogonal() {
        // R(θ)ᵀ·R(θ) = I
        for theta in [-1.0, 0.3, 2.0, 5.0] {
            let r = Matrix2::rotation(theta);
            assert_matrix2_eq(r.transpose().mul_matrix(r), Matrix2::IDENTITY);
        }
    }

    #[test]
    fn transpose_is_the_inverse_rotation() {
        // R(θ)ᵀ = R(−θ)
        for theta in [-1.0, 0.3, 2.0] {
            assert_matrix2_eq(
                Matrix2::rotation(theta).transpose(),
                Matrix2::rotation(-theta),
            );
        }
    }

    #[test]
    fn rotations_compose_by_adding_angles() {
        // R(α)·R(β) = R(α + β)
        for (alpha, beta) in [(0.3, 0.4), (1.0, -2.5), (PI, FRAC_PI_2)] {
            let product = Matrix2::rotation(alpha).mul_matrix(Matrix2::rotation(beta));
            assert_matrix2_eq(product, Matrix2::rotation(alpha + beta));
        }
    }

    #[test]
    fn relative_rotation_identity() {
        // R(α)ᵀ·R(β) = R(β − α). RoPE depends on this identity (Stage 3).
        for (alpha, beta) in [(0.3, 0.4), (2.0, -1.0), (5.0, 7.5)] {
            let product = Matrix2::rotation(alpha)
                .transpose()
                .mul_matrix(Matrix2::rotation(beta));
            assert_matrix2_eq(product, Matrix2::rotation(beta - alpha));
        }
    }

    #[test]
    fn full_turn_returns_to_the_start() {
        assert_vector2_eq(rotate_vector([2.0, -1.0], TAU), [2.0, -1.0]);
    }
}
