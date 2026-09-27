# DeepSeek-V4: efficient million-token context intelligence

English | [中文](README_zh.md) | [Previous: DeepSeek-V3.2-Exp](../06-deepseek-v3-2-sparse-attention/README.md) | [Paper index](../README.md)

Paper: [DeepSeek-V4](https://arxiv.org/abs/2606.19348), a preview technical report centered on one-million-token contexts.

## 1. Position

V4-Pro is reported at about 1.6T total and 49B activated parameters; V4-Flash at about 284B total and 13B activated. Both support 1M tokens and were pretrained on more than 32T tokens. The report changes attention, residual connections, and optimization together: CSA/HCA, mHC, and Muon.

## 2. Architecture and optimization

Compressed Sparse Attention (CSA) and Heavily Compressed Attention (HCA) use different levels of historical compression. MLA mainly compresses the KV representation; V4 also reduces how much history participates in attention while retaining local precision and remote summaries.

Manifold-Constrained Hyper-Connections (mHC) constrain residual mixing so signal propagation remains stable at very large scale. Muon is reported as an optimizer for faster convergence and stability. Architecture, optimizer, and kernel changes must not be conflated when interpreting gains.

## 3. Paper evidence

The abstract reports that at 1M context V4-Pro uses about 27% of V3.2 single-token inference FLOPs and about 10% of its KV cache. These are system-level paper results, not numbers reproducible with ordinary PyTorch attention. Read capability tables, efficiency tables, and convergence/stability evidence separately. See the [source ledger](../assets/deepseek-v4/SOURCES.md).

## 4. TinySeek boundary

TinySeek contains teaching MLA, MoE, and MTP, but no CSA/HCA, mHC, Muon, million-token data, or comparable memory system. It can illustrate the conceptual route from compressed representations to selected history, but cannot validate the reported 27% FLOPs or 10% KV cache.

## 5. Claim/evidence boundary

| Paper claim | Paper evidence | TinySeek evidence | Boundary |
| --- | --- | --- | --- |
| V4 supports million-token contexts | Model specification and long-context experiments | No million-token run | Paper-only |
| CSA/HCA reduce long-context cost | 1M-context efficiency comparison | Existing MLA cost record | Related problem, not same method |
| mHC and Muon improve stability/convergence | Training and optimizer experiments | No matching implementation | Paper-only |

## 6. Conclusion

V4 is a coordinated design for million-token context: attention avoids unnecessary history, mHC stabilizes residual flow, and Muon changes optimization. It advances V3.2's sparse-attention direction into a joint context, training-dynamics, and systems design.

## 7. Sources

- [arXiv](https://arxiv.org/abs/2606.19348)
- [PDF](https://arxiv.org/pdf/2606.19348)
- [Hugging Face collection](https://huggingface.co/collections/deepseek-ai/deepseek-v4)
- [Source ledger](../assets/deepseek-v4/SOURCES.md)

