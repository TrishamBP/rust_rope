//! Example: the Stage 2 mathematical foundations.
//!
//! Run with: `cargo run --example math_foundations`
//!
//! This example prints the numerical examples of `explain.md`, Entries 2–6.
//! It uses only the public API of the library.

use rust_rope::math::complex::Complex;
use rust_rope::math::matrix::{Matrix2, Vector, rotate_vector};
use rust_rope::math::trigonometry::{degrees_to_radians, unit_circle_point};
use std::f64::consts::{FRAC_PI_2, FRAC_PI_4, PI};

fn main() {
    println!("== Vectors (Entry 2) ==");
    let a = Vector::new(vec![1.0, 2.0, 3.0]);
    let b = Vector::new(vec![4.0, 5.0, 6.0]);
    println!("{:?}", a.checked_add(&b).unwrap().components());
    println!("{:?}", a.checked_sub(&b).unwrap().components());
    println!("{:?}", a.dot(&b));
    println!("{}", Vector::new(vec![3.0, 4.0]).magnitude());
    println!("{}", a.dot(&Vector::new(vec![1.0, 2.0])).unwrap_err());

    println!();
    println!("== Complex numbers (Entry 3) ==");
    let z = Complex::new(1.0, 2.0);
    let w = Complex::new(3.0, 4.0);
    println!("{:?}", z + w);
    println!("{:?}", z * w);
    println!("{:?}", Complex::I * Complex::I);
    println!("{:?}", w.conjugate());
    println!("{}", w.magnitude());
    println!("{}", Complex::new(-1.0, -1.0).phase());

    println!();
    println!("== Trigonometry (Entry 4) ==");
    for degrees in [0.0, 90.0, 180.0, 270.0] {
        let theta = degrees_to_radians(degrees);
        let [x, y] = unit_circle_point(theta);
        println!("{degrees:>5}° = {theta:.4} rad -> ({x:e}, {y:e})");
    }

    println!();
    println!("== Matrices and rotations (Entry 5) ==");
    let m = Matrix2::new(1.0, 2.0, 3.0, 4.0);
    println!("{:?}", m.mul_vector([5.0, 6.0]));
    println!(
        "{:?}",
        m.mul_matrix(Matrix2::new(5.0, 6.0, 7.0, 8.0)).rows()
    );
    for theta in [0.0, FRAC_PI_2, PI, 3.0 * FRAC_PI_2] {
        let [x, y] = rotate_vector([1.0, 0.0], theta);
        println!("θ = {theta:.4}: ({x:e}, {y:e})");
    }

    println!();
    println!("== Euler's formula (Entry 6) ==");
    let e_i_pi = Complex::from_polar(1.0, PI);
    println!("e^(iπ)          = {e_i_pi:?}");
    let z = Complex::new(1.0, 1.0);
    let rotated = z.rotate(FRAC_PI_4);
    println!("(1 + i)·e^(iπ/4) = {rotated:?}");
    println!(
        "magnitude {} -> {}, phase {} -> {}",
        z.magnitude(),
        rotated.magnitude(),
        z.phase(),
        rotated.phase()
    );
    let theta = 0.5;
    let by_complex = Complex::new(2.0, 3.0).rotate(theta);
    let by_matrix = rotate_vector([2.0, 3.0], theta);
    println!(
        "complex: ({:.15}, {:.15})",
        by_complex.re(),
        by_complex.im()
    );
    println!("matrix:  ({:.15}, {:.15})", by_matrix[0], by_matrix[1]);
}
