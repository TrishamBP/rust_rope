//! Allocation tests (Stage 9).
//!
//! These tests **count heap allocations**. They prove the documented
//! allocation behaviour of the RoPE functions:
//!
//! | Function | Heap allocations |
//! |---|---|
//! | `rotate_in_place`, `rotate_tensor_in_place` | 0 |
//! | `rotate_tensor_with_table`, `rotate_query_key_with_table` | 0 |
//! | `rotate` | 1 (the result `Vec`) |
//! | `rotate_query_key` | 2 (the two tensor copies) |
//! | `sin_cos_table` | 2 (the `sin` and `cos` arrays) |
//!
//! ## Why a separate file?
//!
//! The test installs its own **global allocator** with
//! `#[global_allocator]`. Only one global allocator can exist in a program.
//! Cargo compiles each file in `tests/` as a separate program. Thus this
//! file can replace the allocator without an effect on `rope_tests.rs` or
//! `numerical_tests.rs`.
//!
//! ## How the counter works
//!
//! `CountingAllocator` forwards every request to the normal system
//! allocator (`std::alloc::System`). Before it forwards an allocation, it
//! adds 1 to a **thread-local** counter. The test harness runs tests on
//! several threads at the same time. A thread-local counter counts only the
//! allocations of the current test thread, so parallel tests do not disturb
//! each other.
//!
//! ## `unsafe`
//!
//! `GlobalAlloc` is an `unsafe trait`: the compiler cannot check that an
//! allocator returns valid memory. The safety argument (explain.md,
//! Entry 17) is short: every method calls the same method of `System` with
//! the same arguments, and returns its result unchanged. Thus this allocator
//! is exactly as correct as `System`. The counter does not allocate (it is
//! a `Cell<usize>` with a `const` initializer), so the allocator never calls
//! itself.

use rust_rope::math::tensor::{Tensor4, TensorShape};
use rust_rope::rope::config::{RopeConfig, RopeLayout};
use rust_rope::rope::embeddings::RotaryEmbedding;
use rust_rope::rope::position::{Position, PositionRange};
use rust_rope::rope::scaling::RopeScaling;
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

thread_local! {
    /// The number of allocations of the current thread. `const { ... }`
    /// makes the initialization free: no lazy setup and no allocation on
    /// first use.
    static ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
}

/// A global allocator that counts allocations and forwards to `System`.
struct CountingAllocator;

fn count_one() {
    // `try_with` instead of `with`: during thread shutdown the thread-local
    // value can already be destroyed. Then we skip the count; we do not
    // panic inside the allocator.
    let _ = ALLOCATIONS.try_with(|n| n.set(n.get() + 1));
}

// SAFETY: every method forwards to `System` with unchanged arguments and
// returns the result of `System` unchanged. `System` satisfies the
// `GlobalAlloc` contract, so this type satisfies it too. `count_one` does not
// allocate and does not unwind.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        count_one();
        // SAFETY: the caller of `alloc` guarantees the `GlobalAlloc`
        // preconditions for `layout`; we pass them on unchanged.
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        count_one();
        // SAFETY: as for `alloc`.
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // A reallocation is a new allocation request too: count it.
        count_one();
        // SAFETY: `ptr` was allocated by this allocator, which means by
        // `System`, with `layout`; the caller guarantees this.
        unsafe { System.realloc(ptr, layout, new_size) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: `ptr` was allocated by `System` with `layout`.
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static GLOBAL: CountingAllocator = CountingAllocator;

/// Runs `f` and returns the number of heap allocations that it made on the
/// current thread.
fn allocations_in(f: impl FnOnce()) -> usize {
    let before = ALLOCATIONS.with(Cell::get);
    f();
    ALLOCATIONS.with(Cell::get) - before
}

fn test_tensor(shape: TensorShape) -> Tensor4 {
    Tensor4::from_fn(shape, |b, s, h, d| (b + s + h + d) as f64 * 0.25).unwrap()
}

fn embeddings() -> Vec<RotaryEmbedding> {
    let mut all = Vec::new();
    for layout in [RopeLayout::Interleaved, RopeLayout::SplitHalf] {
        let c = RopeConfig::new(32, 10_000.0)
            .unwrap()
            .with_layout(layout)
            .with_head_dimension(64)
            .unwrap();
        all.push(RotaryEmbedding::new(c));
        all.push(RotaryEmbedding::with_scaling(c, RopeScaling::yarn(8.0, 256)).unwrap());
    }
    all
}

#[test]
fn the_counter_counts() {
    // A self-test: if the counter did not work, every "0 allocations" test
    // below would pass for the wrong reason.
    assert_eq!(allocations_in(|| drop(Vec::<f64>::with_capacity(10))), 1);
    assert_eq!(allocations_in(|| drop(Vec::<f64>::new())), 0); // empty Vec: no allocation
}

#[test]
fn in_place_rotations_do_not_allocate() {
    let shape = TensorShape::new(2, 16, 4, 64);
    for rope in embeddings() {
        let mut v = vec![0.5; 64];
        let mut t = test_tensor(shape);
        let mut k = test_tensor(shape);
        let n = allocations_in(|| {
            rope.rotate_in_place(&mut v, Position::new(3).unwrap())
                .unwrap();
            rope.rotate_tensor_in_place(&mut t, Position::ZERO).unwrap();
            rope.rotate_query_key_in_place(&mut t, &mut k, Position::ZERO)
                .unwrap();
        });
        assert_eq!(n, 0, "{:?}", rope.config());
    }
}

#[test]
fn table_rotations_do_not_allocate() {
    let shape = TensorShape::new(2, 16, 4, 64);
    for rope in embeddings() {
        // The table allocates exactly two arrays: sin and cos.
        let mut table = None;
        let n = allocations_in(|| {
            table = Some(
                rope.sin_cos_table(PositionRange::from_zero(128).unwrap())
                    .unwrap(),
            );
        });
        assert_eq!(n, 2, "{:?}", rope.config());
        let table = table.unwrap();

        let mut q = test_tensor(shape);
        let mut k = test_tensor(shape);
        let n = allocations_in(|| {
            rope.rotate_tensor_with_table(&mut q, &table, Position::new(5).unwrap())
                .unwrap();
            rope.rotate_query_key_with_table(&mut q, &mut k, &table, Position::ZERO)
                .unwrap();
        });
        assert_eq!(n, 0, "{:?}", rope.config());
    }
}

#[test]
fn copying_rotations_allocate_exactly_their_outputs() {
    let rope = RotaryEmbedding::new(RopeConfig::new(64, 10_000.0).unwrap());
    let x = vec![1.0; 64];
    assert_eq!(
        allocations_in(|| drop(rope.rotate(&x, Position::new(9).unwrap()).unwrap())),
        1
    );
    let q = test_tensor(TensorShape::new(1, 8, 2, 64));
    assert_eq!(
        allocations_in(|| drop(rope.rotate_query_key(&q, &q, Position::ZERO).unwrap())),
        2
    );
}

#[test]
fn errors_do_not_allocate() {
    let rope = RotaryEmbedding::new(RopeConfig::new(64, 10_000.0).unwrap());
    let short = vec![1.0; 10];
    let n = allocations_in(|| assert!(rope.rotate(&short, Position::ZERO).is_err()));
    assert_eq!(n, 0);
}
