//! Exports RoPE test cases for the independent PyTorch reference (Stages 6–7).
//!
//! Run with: `cargo run --example export_reference_cases`
//!
//! The program writes `reference/data/rust_cases.txt`. Each case contains the
//! configuration, the deterministic input values, and the output of the Rust
//! implementation. `reference/compare_rope.py` reads the file, recomputes
//! every case with PyTorch, and compares.
//!
//! This is validation tooling. It uses only the public API of the library.
//!
//! ## File format (plain text, one item per line)
//!
//! ```text
//! case name=<id> api=<vector|tensor> base=<b> layout=<interleaved|split_half> rotary_dim=<d> shape=<B>,<S>,<H>,<D> start=<m>
//! input <B·S·H·D values in row-major order>
//! output <B·S·H·D values>
//! end
//! ```
//!
//! Values are written with `{:?}`. For `f64`, this prints the **shortest**
//! decimal text that converts back to exactly the same `f64` value. Thus the
//! exchange through text loses no precision.
//!
//! Stage 6 cases: interleaved, full rotation. Stage 7 adds split-half and
//! partial-rotation cases.

use rust_rope::math::tensor::{Tensor4, TensorShape};
use rust_rope::rope::config::{RopeConfig, RopeLayout};
use rust_rope::rope::embeddings::RotaryEmbedding;
use rust_rope::rope::position::Position;
use rust_rope::rope::scaling::RopeScaling;
use std::fmt::Write as _;
use std::fs;

type AnyError = Box<dyn std::error::Error>;

/// Deterministic input values with positive, negative and mixed signs.
fn input_value(k: usize, seed: f64) -> f64 {
    2.0 * (k as f64 * 0.731 + seed).sin() + 0.25 * (k as f64 * 1.37).cos()
}

fn write_values(out: &mut String, label: &str, values: &[f64]) {
    out.push_str(label);
    for v in values {
        // `write!` into a String cannot fail.
        let _ = write!(out, " {v:?}");
    }
    out.push('\n');
}

fn layout_name(layout: RopeLayout) -> &'static str {
    match layout {
        RopeLayout::Interleaved => "interleaved",
        RopeLayout::SplitHalf => "split_half",
    }
}

fn scaling_spec(scaling: RopeScaling) -> String {
    match scaling {
        RopeScaling::None => "none".to_string(),
        RopeScaling::PositionInterpolation {
            original_context,
            target_context,
        } => format!("pi:{original_context}:{target_context}"),
        RopeScaling::Linear { factor } => format!("linear:{factor:?}"),
        RopeScaling::NtkAware { factor } => format!("ntk:{factor:?}"),
        RopeScaling::Yarn {
            factor,
            original_context,
            alpha,
            beta,
        } => format!("yarn:{factor:?}:{original_context}:{alpha:?}:{beta:?}"),
    }
}

struct Case {
    name: String,
    api: &'static str,
    config: RopeConfig,
    scaling: RopeScaling,
    shape: TensorShape,
    start: usize,
    input: Vec<f64>,
    output: Vec<f64>,
}

impl Case {
    fn write(&self, out: &mut String) {
        let (s, c) = (self.shape, self.config);
        let _ = writeln!(
            out,
            "case name={} api={} base={:?} layout={} rotary_dim={} shape={},{},{},{} start={} scaling={}",
            self.name,
            self.api,
            c.base(),
            layout_name(c.layout()),
            c.rotary_dimension(),
            s.batch,
            s.sequence,
            s.heads,
            s.features,
            self.start,
            scaling_spec(self.scaling)
        );
        write_values(out, "input", &self.input);
        write_values(out, "output", &self.output);
        out.push_str("end\n");
    }
}

fn describe(config: RopeConfig) -> String {
    format!(
        "{}_d{}_r{}_b{}",
        layout_name(config.layout()),
        config.head_dimension(),
        config.rotary_dimension(),
        config.base()
    )
}

/// A single-vector case: `RotaryEmbedding::rotate(x, m)`.
fn vector_case(config: RopeConfig, m: usize) -> Result<Case, AnyError> {
    scaled_vector_case(config, RopeScaling::None, m)
}

/// A single-vector case with a scaling method (Stage 8).
fn scaled_vector_case(
    config: RopeConfig,
    scaling: RopeScaling,
    m: usize,
) -> Result<Case, AnyError> {
    let rope = RotaryEmbedding::with_scaling(config, scaling)?;
    let d = config.head_dimension();
    let input: Vec<f64> = (0..d)
        .map(|k| input_value(k, d as f64 + m as f64 * 0.01))
        .collect();
    let output = rope.rotate(&input, Position::new(m)?)?;
    Ok(Case {
        name: format!(
            "vector_{}_{}_m{m}",
            describe(config),
            scaling_spec(scaling).replace(':', "-")
        ),
        api: "vector",
        config,
        scaling,
        shape: TensorShape::new(1, 1, 1, d),
        start: m,
        input,
        output,
    })
}

/// A tensor case: `RotaryEmbedding::rotate_tensor_in_place(t, start)`.
/// The feature extent of the tensor is the head dimension of `config`.
fn tensor_case(
    config: RopeConfig,
    bsh: (usize, usize, usize),
    start: usize,
) -> Result<Case, AnyError> {
    scaled_tensor_case(config, RopeScaling::None, bsh, start)
}

/// A tensor case with a scaling method (Stage 8).
fn scaled_tensor_case(
    config: RopeConfig,
    scaling: RopeScaling,
    (batch, sequence, heads): (usize, usize, usize),
    start: usize,
) -> Result<Case, AnyError> {
    let rope = RotaryEmbedding::with_scaling(config, scaling)?;
    let shape = TensorShape::new(batch, sequence, heads, config.head_dimension());
    let seed = (batch * 7 + sequence * 5 + heads * 3) as f64;
    let mut k = 0;
    let tensor = Tensor4::from_fn(shape, |_, _, _, _| {
        k += 1;
        input_value(k - 1, seed)
    })?;
    let input = tensor.as_slice().to_vec();
    let mut rotated = tensor;
    rope.rotate_tensor_in_place(&mut rotated, Position::new(start)?)?;
    Ok(Case {
        name: format!(
            "tensor_{batch}x{sequence}x{heads}_{}_{}_start{start}",
            describe(config),
            scaling_spec(scaling).replace(':', "-")
        ),
        api: "tensor",
        config,
        scaling,
        shape,
        start,
        input,
        output: rotated.into_vec(),
    })
}

fn main() -> Result<(), AnyError> {
    let mut cases = Vec::new();

    // ---- Stage 6: interleaved, full rotation ------------------------------
    for d in [2, 4, 8, 64, 128] {
        for base in [10_000.0, 500_000.0] {
            for m in [0, 1, 7, 1000, 100_000] {
                cases.push(vector_case(RopeConfig::new(d, base)?, m)?);
            }
        }
    }
    cases.push(vector_case(RopeConfig::new(8, 2.0)?, 3)?);
    for (bsh, d) in [
        ((1, 1, 1), 8),
        ((2, 4, 2), 4),
        ((2, 5, 3), 8),
        ((1, 16, 4), 64),
        ((3, 7, 2), 128),
    ] {
        for (base, start) in [(10_000.0, 0), (10_000.0, 3), (500_000.0, 4096)] {
            cases.push(tensor_case(RopeConfig::new(d, base)?, bsh, start)?);
        }
    }

    // ---- Stage 7: split-half layout and partial rotation ------------------
    // (rotary dimension d, head dimension D)
    let dims = [
        (2, 2),
        (4, 4),
        (6, 6),
        (8, 8),
        (64, 64),
        (128, 128),
        (2, 8),
        (4, 6),
        (8, 12),
        (64, 128),
    ];
    for layout in [RopeLayout::Interleaved, RopeLayout::SplitHalf] {
        for (rotary, head) in dims {
            let config = RopeConfig::new(rotary, 10_000.0)?
                .with_layout(layout)
                .with_head_dimension(head)?;
            // Full interleaved cases already exist above; avoid duplicates.
            let duplicate = layout == RopeLayout::Interleaved && rotary == head;
            if !duplicate {
                for m in [0, 1, 5, 1000] {
                    cases.push(vector_case(config, m)?);
                }
            }
            for (bsh, start) in [((2, 4, 2), 0), ((2, 3, 3), 17)] {
                if !duplicate || head <= 8 {
                    cases.push(tensor_case(config, bsh, start)?);
                }
            }
        }
        let big = RopeConfig::new(64, 500_000.0)?
            .with_layout(layout)
            .with_head_dimension(128)?;
        cases.push(tensor_case(big, (2, 8, 4), 4096)?);
    }

    // ---- Stage 8: scaling methods -----------------------------------------
    let methods = [
        RopeScaling::PositionInterpolation {
            original_context: 2048,
            target_context: 8192,
        },
        RopeScaling::Linear { factor: 4.0 },
        RopeScaling::Linear { factor: 2.5 },
        RopeScaling::NtkAware { factor: 4.0 },
        RopeScaling::NtkAware { factor: 8.0 },
        RopeScaling::yarn(4.0, 2048),
        RopeScaling::yarn(16.0, 4096),
        RopeScaling::Linear { factor: 1.0 },
    ];
    for scaling in methods {
        for d in [8, 64, 128] {
            for m in [0, 1, 4096, 8191] {
                cases.push(scaled_vector_case(
                    RopeConfig::new(d, 10_000.0)?,
                    scaling,
                    m,
                )?);
            }
        }
        for layout in [RopeLayout::Interleaved, RopeLayout::SplitHalf] {
            let partial = RopeConfig::new(32, 500_000.0)?
                .with_layout(layout)
                .with_head_dimension(64)?;
            cases.push(scaled_tensor_case(partial, scaling, (2, 4, 3), 6000)?);
            let full = RopeConfig::new(16, 10_000.0)?.with_layout(layout);
            cases.push(scaled_tensor_case(full, scaling, (1, 5, 2), 0)?);
        }
    }

    let mut out = String::from("# rust_rope reference cases, format version 3\n");
    for case in &cases {
        case.write(&mut out);
    }
    fs::create_dir_all("reference/data")?;
    fs::write("reference/data/rust_cases.txt", out)?;
    println!(
        "wrote {} cases to reference/data/rust_cases.txt",
        cases.len()
    );
    Ok(())
}
