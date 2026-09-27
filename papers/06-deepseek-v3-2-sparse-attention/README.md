# DeepSeek-V3.2-Exp: from MLA to sparse attention

English | [中文](README_zh.md) | [Previous: DeepSeek-V3](../04-deepseek-v3/README.md) | [Next: DeepSeek-V4](../07-deepseek-v4/README.md) | [Paper index](../README.md)

Paper: DeepSeek-V3.2-Exp, an experimental transition between V3.1-Terminus and V4. Its question is whether MLA's KV compression is enough when full attention still scores every historical token.

## 1. Position

DeepSeek Sparse Attention (DSA) uses a lightweight indexer to select a small set of relevant historical tokens, then computes exact attention only on the selected positions. The official material presents it as a long-context training/inference efficiency experiment aligned against V3.1-Terminus.

MLA compresses K/V state; DSA also sparsifies which history participates in computation. A useful route is V2 MLA, V3 MLA plus MoE, V3.2-Exp token selection, then V4 CSA/HCA.

## 2. Method

For query position i, the indexer scores historical positions j and keeps top-k candidates. The selector decides where to look; the main attention still computes final weights from the full representations. The intuition is to replace full O(L²) attention with selection cost plus O(Lk), where k is much smaller than context length L.

Training must align the indexer with useful attention patterns. At inference, indices may be cached or updated incrementally. The claim is not that every token can be discarded, but that learned selection can preserve the tokens useful for the current query.

## 3. Reading the experiments

Check three layers: selection quality, capability retention against V3.1-Terminus, and long-context FLOPs/KV-cache/throughput. Do not attribute every benchmark difference to DSA because the experimental release also involves training, kernels, and serving changes. See the official PDF and the [source ledger](../assets/deepseek-v3-2/SOURCES.md).

## 4. TinySeek boundary

The repository has teaching MLA, dense/GQA, and V3 routing implementations, but no DSA indexer, top-k sparse kernel, or V3.2-scale training. Existing architecture reports therefore cannot reproduce the paper efficiency. A future classroom sketch could compare full attention with a fixed top-k selector, but that would be an algorithm-shape demonstration, not validation.

## 5. Claim/evidence boundary

| Paper claim | Paper evidence | TinySeek evidence | Boundary |
| --- | --- | --- | --- |
| Sparse selection lowers long-context attention cost | DSA method and efficiency experiments | No DSA run | Paper-only; missing indexer, kernel, and scale |
| Capability aligns with V3.1-Terminus | Official benchmark comparison | V3 teaching evaluation only | Not directly comparable |
| MLA still leaves an attention-compute bottleneck | V3.2 motivation and long-context results | Existing MLA cost records | Directionally related, not proof of DSA |

## 6. Conclusion

V3.2-Exp bridges compressed state and selected computation: MLA stores less per token, while DSA lets each query attend to fewer historical tokens. Its efficiency depends on indexer training and specialized kernels, so ordinary small PyTorch attention cannot reproduce it.

## 7. Sources

- [Official repository](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp)
- [Technical report](https://raw.githubusercontent.com/deepseek-ai/DeepSeek-V3.2-Exp/main/DeepSeek_V3_2.pdf)
- [Source ledger](../assets/deepseek-v3-2/SOURCES.md)
+## Deep reading: DSA is about learnable selection

### 1. The bottleneck left after MLA

MLA reduces the cached representation, but if every query still scores every historical position, long-context attention remains close to quadratic. V3.2-Exp changes the question from “how much do we store?” to “how much do we compute?”

### 2. Indexer versus main attention

The indexer performs candidate recall; the main attention computes the precise aggregation. This is analogous to retrieval and ranking. If the indexer is too cheap it misses critical tokens; if it is too expensive the sparse gain disappears. The real tradeoff is recall, index cost, and exact-attention cost.

### 3. Why alignment against V3.1 matters

Capability alignment provides a basic control: if quality is roughly preserved while long-context efficiency improves, DSA is a plausible explanation. It is still not a randomized ablation because the experimental release can change training, kernels, and serving together.

### 4. Sparse-selection risks

Top-k selection can miss long-range dependencies, rare entities, or multi-hop evidence. Stable average results only establish adequacy on the paper's training and evaluation distribution. Selection recall and failure cases matter as much as average FLOPs.

### 5. TinySeek transfer

Without the DSA kernel, TinySeek can only demonstrate the algorithmic shape. A useful classroom implementation must record selection recall, PPL, long-range accuracy, and wall-clock cost, not just the number of skipped dot products.
+## Evidence map

| Paper location | Question | Authors' conclusion | Reading boundary |
| --- | --- | --- | --- |
| DSA method | Can an indexer recall useful history? | Sparse selection can replace full access | Selection recall and failures matter |
| V3.1 comparisons | Is capability preserved? | Public benchmarks are broadly aligned | Not a randomized ablation |
| Efficiency experiments | Do FLOPs and cache fall? | Long-context cost decreases | Requires specialized kernels and serving |
| Analysis/ablations | How sparse should attention be? | Quality and cost must be balanced | Smaller top-k is not automatically better |
