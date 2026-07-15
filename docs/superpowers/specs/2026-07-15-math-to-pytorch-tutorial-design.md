# Math-to-PyTorch Tutorial Design

## Goal

Turn the existing architecture walkthrough into a beginner-friendly path where
each important model operation can be traced from its mathematical definition
to tensor shapes, PyTorch APIs, and the exact TinySeek implementation.

## Teaching Pattern

Every deep dive uses the same order:

1. State the mathematical formula and define every symbol.
2. Give the input, intermediate, and output tensor shapes.
3. show a literal PyTorch translation before any optimized implementation.
4. Map each formula term to the repository code line by line.
5. Explain every non-obvious PyTorch API used by that code.
6. Show one small numerical or shape example.
7. List common implementation mistakes and one check-yourself exercise.

The Chinese and English chapters have the same technical coverage. The Chinese
version may use more explanatory prose, but neither version may omit formulas,
shape tables, or API explanations.

## Information Architecture

- Add a bilingual math-to-PyTorch reference chapter for reusable concepts and
  APIs such as broadcasting, `nn.Parameter`, `register_buffer`, `view`,
  `transpose`, `contiguous`, `softmax`, `topk`, and cross entropy.
- Expand the Dense chapter in place. RMSNorm, RoPE, GQA attention, SwiGLU,
  residual blocks, embedding/weight tying, and causal LM loss stay beside the
  first complete model that uses them.
- Expand the MoE, V2, and V3 evolution chapters in place. Their formulas remain
  beside the architecture change and measured decision gate.
- Expand training and post-training walkthroughs for autograd, AMP, gradient
  accumulation, clipping, label masking, log probabilities, advantages, and the
  educational GRPO objective.
- Put the reference chapter immediately before the Dense chapter in tutorial
  navigation so readers learn the notation and PyTorch vocabulary first.

## Accuracy Boundaries

- Explain the code that exists in this repository, not an imagined production
  DeepSeek implementation.
- Label educational MLA and GRPO simplifications explicitly.
- Distinguish mathematical equivalence from implementation details such as
  broadcasting, fused kernels, and tensor layout.
- Do not claim that a small numerical example proves training quality.

## Verification

- A documentation contract checks that both language versions exist and expose
  formula, shape, API, and code-mapping sections.
- Navigation and indexes include the new reference chapter.
- Existing Markdown-link and documentation-contract checks remain green.
- A fresh-reader review checks whether a beginner can answer what each tensor
  means and why every cited PyTorch operation is needed.
