//! Example: the Stage 3 RoPE fundamentals.
//!
//! Run with: `cargo run --example rope_frequencies`
//!
//! This example prints the numerical examples of `explain.md`, Entries 7–9:
//! the inverse frequencies, the positional angles, and the rotation matrix
//! that each angle defines. It uses only the public API of the library.
//!
//! It does not rotate an embedding vector. That is planned for Stage 4.

use rust_rope::math::matrix::Matrix2;
use rust_rope::rope::config::RopeConfig;
use rust_rope::rope::frequencies::{AngleTable, inverse_frequencies};
use rust_rope::rope::position::{Position, PositionRange};
use std::f64::consts::TAU;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("== Inverse frequencies, d = 8, b = 10000 (Entry 7) ==");
    let config = RopeConfig::new(8, 10_000.0)?;
    let theta = inverse_frequencies(&config);
    for (j, &t) in theta.iter().enumerate() {
        let exponent = (2 * j) as f64 / config.dimension() as f64;
        println!(
            "j = {j}: 2j/d = {exponent:<4}  θ = {t:<22?}  wavelength 2π/θ = {:.2}",
            TAU / t
        );
    }

    println!();
    println!("== Ratio between neighbouring frequencies ==");
    for w in theta.windows(2) {
        println!("θ(j+1)/θ(j) = {:?}", w[1] / w[0]);
    }

    println!();
    println!("== Frequency range for other dimensions (b = 10000) ==");
    for d in [64, 128] {
        let t = inverse_frequencies(&RopeConfig::new(d, 10_000.0)?);
        println!(
            "d = {d:>3}: {} pairs, θ₀ = {:?}, θ₁ = {:.6e}, θ_last = {:.6e}",
            t.len(),
            t[0],
            t[1],
            t[t.len() - 1]
        );
    }

    println!();
    println!("== Angles φ(m, j) = m·θⱼ for m = 0..4 (Entry 9) ==");
    let table = AngleTable::new(PositionRange::from_zero(5)?, &theta)?;
    println!("       j = 0   j = 1   j = 2   j = 3   (rounded to 3 decimals)");
    for m in table.positions().iter() {
        let row = table.row(m).expect("the table contains its own positions");
        print!("m = {}  ", m.value());
        for phi in row {
            print!("{phi:<8.3}");
        }
        println!();
    }
    let phi_3_1 = table.get(Position::new(3)?, 1).unwrap_or(f64::NAN);
    println!("full f64 value of φ(3, 1) = 3 · 0.1: {phi_3_1:?}");

    println!();
    println!("== Rotation matrices R(m, j) at m = 1 ==");
    let m = Position::new(1)?;
    for j in 0..config.num_pairs() {
        let phi = table.get(m, j).expect("m = 1 and j < 4 are in the table");
        let [[a, b], [c, e]] = Matrix2::rotation(phi).rows();
        println!("j = {j}: φ = {phi:<6} R = [[{a:.6}, {b:.6}], [{c:.6}, {e:.6}]]");
    }

    println!();
    println!("== Position 0 gives the identity matrix ==");
    let [[a, b], [c, e]] = Matrix2::rotation(table.get(Position::ZERO, 0).unwrap_or(0.0)).rows();
    println!("R(0, 0) = [[{a}, {b}], [{c}, {e}]]");

    println!();
    println!("== Start offset: positions 1000..1002, pair j = 3 ==");
    let later = AngleTable::new(PositionRange::new(1000, 3)?, &theta)?;
    for m in later.positions().iter() {
        println!(
            "m = {}: φ = {:?}",
            m.value(),
            later.get(m, 3).unwrap_or(f64::NAN)
        );
    }
    Ok(())
}
