# 01. DeepSeek Paper Map for LM Training

This map filters the local paper folder down to the LM training path.

## Core Reading Path

1. **DeepSeek LLM**
   - Why it matters: starts from scaling and training recipe.
   - TinySeek chapter: LR / batch-size sweep, warmup, schedule, data/model scale.
   - Local PDF: `../DeepSeek-papers/chronological-pdfs/02_...DeepSeek_LLM...pdf`

2. **DeepSeekMoE**
   - Why it matters: expert specialization, routed experts, load balance.
   - TinySeek chapter: dense FFN -> routed MoE FFN.
   - Local PDF: `../DeepSeek-papers/chronological-pdfs/03_...DeepSeekMoE...pdf`

3. **DeepSeek-V2**
   - Why it matters: combines DeepSeekMoE and MLA; pretrain + SFT + RL.
   - TinySeek chapter: educational MLA and KV-cache accounting.
   - Local PDF: `../DeepSeek-papers/chronological-pdfs/07_...DeepSeek-V2...pdf`

4. **DeepSeek-V3**
   - Why it matters: validates MoE + MLA, introduces auxiliary-loss-free
     balancing and multi-token prediction.
   - TinySeek chapter: advanced MoE load-balance ablations.
   - Local PDF: `../DeepSeek-papers/chronological-pdfs/15_...DeepSeek-V3...pdf`

5. **DeepSeek-R1**
   - Why it matters: R1-Zero tests RL directly from a pretrained base; R1 adds
     cold-start reasoning SFT, rejection sampling, SFT again, then RL.
   - TinySeek chapter: cold-start SFT vs direct rule RL.
   - Local PDF: `../DeepSeek-papers/chronological-pdfs/16_...DeepSeek-R1...pdf`

6. **DeepSeek-V3.2 / V4**
   - Why it matters: long-context sparse attention, larger post-training budget,
     continued MoE architecture evolution.
   - TinySeek chapter: optional sparse-attention and post-training scaling notes.

## Optional LM Papers

- DeepSeekMath and DeepSeek-Prover: reasoning and verification data.
- ESFT: expert-specialized fine-tuning.
- Native Sparse Attention: efficient long-context attention.
- Inference-Time Scaling for Generalist Reward Modeling: reward modeling and
  inference-time compute.
- Engram: conditional memory as another sparsity axis.

## Explicitly Skipped

- DreamCraft3D.
- DeepSeek-VL / VL2.
- Janus / JanusFlow / Janus-Pro.
- DeepSeek-OCR / OCR-2.

These are good papers, but not part of this LM-only tutorial.
