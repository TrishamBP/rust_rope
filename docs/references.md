# References

This list contains the primary sources for the project.
We add new sources when a stage needs them.

## RoPE and positional encodings

1. Su, J., Lu, Y., Pan, S., Murtadha, A., Wen, B., Liu, Y. (2021).
   *RoFormer: Enhanced Transformer with Rotary Position Embedding.*
   arXiv:2104.09864. — The original RoPE paper.
2. Vaswani, A., et al. (2017). *Attention Is All You Need.*
   arXiv:1706.03762. — The Transformer architecture and sinusoidal
   absolute position encodings.
3. Shaw, P., Uszkoreit, J., Vaswani, A. (2018). *Self-Attention with Relative
   Position Representations.* arXiv:1803.02155.
4. Raffel, C., et al. (2020). *Exploring the Limits of Transfer Learning with
   a Unified Text-to-Text Transformer (T5).* arXiv:1910.10683. — Relative
   position bias.
5. Biderman, S., Black, S., et al. (2021). *Rotary Embeddings: A Relative
   Revolution.* EleutherAI blog. — An accessible explanation of RoPE.
6. Touvron, H., et al. (2023). *LLaMA: Open and Efficient Foundation Language
   Models.* arXiv:2302.13971. — A widely used model that uses RoPE.

## Context extension (Stage 8; originally planned as Stage 5)

7. Chen, S., Wong, S., Chen, L., Tian, Y. (2023). *Extending Context Window of
   Large Language Models via Positional Interpolation.* arXiv:2306.15595.
8. bloc97 (2023). *NTK-Aware Scaled RoPE.* Online discussion post
   (r/LocalLLaMA). — The origin of NTK-aware scaling.
9. Peng, B., Quesnelle, J., Fan, H., Shippole, E. (2023). *YaRN: Efficient
   Context Window Extension of Large Language Models.* arXiv:2309.00071.
   — Section 3.1 states the NTK-aware formula `b' = b·s^(|D|/(|D|−2))`;
   sections 3.2 and 3.4 define the ramp `γ(r)` (with `α = 1`, `β = 32` for
   LLaMA) and the factor `√(1/t) = 0.1·ln(s) + 1`. This project implements
   these formulas (Stage 8).
18. Jacot, A., Gabriel, F., Hongler, C. (2018). *Neural Tangent Kernel:
    Convergence and Generalization in Neural Networks.* arXiv:1806.07572.
    — The theory behind the name "NTK-aware".
19. Tancik, M., et al. (2020). *Fourier Features Let Networks Learn High
    Frequency Functions in Low Dimensional Domains.* arXiv:2006.10739.
    — Why high-frequency input features matter.
20. Hugging Face `transformers`, `modeling_rope_utils.py`
    (`_compute_linear_scaling_rope_parameters`,
    `_compute_dynamic_ntk_parameters`, `_compute_yarn_parameters`). — A
    widely used implementation. Its YaRN ramp is discretized over dimension
    indices (`floor`/`ceil`); this project does not reproduce that detail.

## Rust

10. *The Rust Programming Language* ("The Book").
    https://doc.rust-lang.org/book/
11. *The Rust Reference.* https://doc.rust-lang.org/reference/
12. *The Cargo Book.* https://doc.rust-lang.org/cargo/
13. *The Rustonomicon* (unsafe Rust and memory layout).
    https://doc.rust-lang.org/nomicon/
14. *The Rust Performance Book.* https://nnethercote.github.io/perf-book/

## Validation tooling (Stage 6)

21. PyTorch (2.15.0, CPU build). https://pytorch.org — used only in
    `reference/` as an independent numerical reference.

## Numerical computing

15. Goldberg, D. (1991). *What Every Computer Scientist Should Know About
    Floating-Point Arithmetic.* ACM Computing Surveys 23(1).
16. IEEE 754-2019. *IEEE Standard for Floating-Point Arithmetic.*

## Writing standard

17. ASD-STE100. *Simplified Technical English.* AeroSpace, Security and
    Defence Industries Association of Europe (ASD).
