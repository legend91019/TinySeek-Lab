# Math-to-PyTorch Tutorial Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make TinySeek-Lab explain each major LM operation from formula to tensor shape, PyTorch API, and exact repository code.

**Architecture:** Add one bilingual reusable PyTorch reference, then deepen the existing Dense, MoE, V2, V3, training, and post-training chapters in place. Keep the stage files as the source of truth and preserve the experiment-driven narrative.

**Tech Stack:** Markdown, LaTeX math, Mermaid, PyTorch, repository contract tests.

## Global Constraints

- English and Chinese documentation must have matching technical coverage.
- Examples must match the current stage-model and trainer implementations.
- Educational MLA and GRPO limitations must remain explicit.
- Do not modify model behavior or experiment results.

---

### Task 1: Add the bilingual math-to-PyTorch reference

**Files:**
- Create: `docs/24_math_to_pytorch.md`
- Create: `docs/zh/24_math_to_pytorch.md`

- [ ] Explain the shared formula-to-code reading method and notation.
- [ ] Cover parameters, buffers, modules, broadcasting, reshape/layout, reductions, selection, and losses with repository examples.
- [ ] Add shape checks and beginner exercises.

### Task 2: Deepen the complete Dense LM walkthrough

**Files:**
- Modify: `docs/12_code_first_dense_lm.md`
- Modify: `docs/zh/12_code_first_dense_lm.md`

- [ ] Expand RMSNorm from formula through each PyTorch method and a numerical example.
- [ ] Expand RoPE, GQA attention, SwiGLU, residuals, embedding/weight tying, and shifted CE using the same template.
- [ ] Distinguish literal math from fused `scaled_dot_product_attention`.

### Task 3: Deepen architecture-evolution code walkthroughs

**Files:**
- Modify: `docs/21_from_dense_to_deepseek_moe.md`
- Modify: `docs/zh/21_from_dense_to_deepseek_moe.md`
- Modify: `docs/22_from_moe_to_deepseek_v2.md`
- Modify: `docs/zh/22_from_moe_to_deepseek_v2.md`
- Modify: `docs/23_from_v2_to_deepseek_v3.md`
- Modify: `docs/zh/23_from_v2_to_deepseek_v3.md`

- [ ] Map router softmax, top-k dispatch, weighted expert sums, and balance loss to code.
- [ ] Map latent KV compression, reconstruction, RoPE split, and cache accounting to educational MLA code.
- [ ] Map bias-based selection and MTP shifted targets to V3 code.

### Task 4: Deepen optimization and post-training walkthroughs

**Files:**
- Modify: `docs/16_training_loop_from_config_to_checkpoint.md`
- Modify: `docs/zh/16_training_loop_from_config_to_checkpoint.md`
- Modify: `docs/19_posttraining_code_walkthrough.md`
- Modify: `docs/zh/19_posttraining_code_walkthrough.md`

- [ ] Explain autograd, gradient accumulation, AMP scaling, unscaling, clipping, and optimizer ordering.
- [ ] Explain SFT masking, `log_softmax` plus `gather`, group-normalized advantages, and the GRPO mini objective.

### Task 5: Integrate navigation and enforce the contract

**Files:**
- Modify: `README.md`
- Modify: `README_zh.md`
- Modify: `docs/README.md`
- Modify: `docs/zh/README.md`
- Modify: `scripts/refresh_doc_nav.py`
- Modify: `tests/docs_contract_test.py`

- [ ] Insert the reference before the Dense chapter in both course indexes and navigation lists.
- [ ] Add contract assertions for bilingual coverage and required teaching markers.
- [ ] Refresh navigation and run docs, links, AST, and diff checks.

### Task 6: Fresh-reader review and delivery

**Files:**
- Review all files changed by Tasks 1-5.

- [ ] Ask an independent reviewer to identify unsupported formulas, API inaccuracies, and assumed background knowledge.
- [ ] Apply valid findings and rerun all documentation checks.
- [ ] Commit and push the verified documentation update to `main`.
