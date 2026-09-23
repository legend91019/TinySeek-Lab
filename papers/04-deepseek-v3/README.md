# DeepSeek-V3: make balancing and training efficiency system problems

English | [中文](README_zh.md) | [Previous: DeepSeek-V2](../03-deepseek-v2/README.md) | [Paper route](../README.md)

Paper: *DeepSeek-V3 Technical Report* (2024, [arXiv:2412.19437](https://arxiv.org/abs/2412.19437)). V3 keeps MLA and DeepSeekMoE and adds auxiliary-loss-free balancing, MTP, communication overlap, FP8 training and deployment engineering.

## 1. Position

The report describes a 671B-total/37B-activated model and a co-design of algorithms, framework and hardware. Its central algorithmic claims are selection bias for load balancing and Multi-Token Prediction; its systems claims include DualPipe, efficient all-to-all and stable FP8 training.

## 2. Transition

V3 changes discrete expert selection with a no-gradient bias rather than adding a load-balancing term to the main loss. MTP adds future-token prediction heads while keeping next-token loss. Systems sections make the sparse model trainable at scale.

## 3. Method

Underloaded experts receive higher selection bias and overloaded experts lower bias; the differentiable mixture weights still come from router affinity. MTP aligns earlier hidden states with future targets and can also support speculative decoding. DualPipe overlaps computation and communication; FP8 uses fine-grained quantization and high-precision accumulation.

## 4. Paper experiments

![Architecture](../assets/deepseek-v3/paper-fig-2-v3-architecture.png)

![MTP](../assets/deepseek-v3/paper-fig-3-mtp.png)

![MTP ablation](../assets/deepseek-v3/paper-table-4-mtp-ablation.png)

![Balance ablation](../assets/deepseek-v3/paper-table-5-balance-ablation.png)

![Expert load](../assets/deepseek-v3/paper-fig-9-expert-load.png)

The report also evaluates FP8/BF16 loss, long context, base/chat benchmarks, distillation and cost. Algorithmic ablations and system feasibility should be read as separate evidence layers.

## 5. TinySeek implementation

See [`stage3_deepseek_v3.py`](../../model/stages/stage3_deepseek_v3.py), [`model/tinyseek.py`](../../model/tinyseek.py), [`s06`](../../course/s06_v3_routing_mtp/README.md), and [`23_from_v2_to_deepseek_v3.md`](../../docs/23_from_v2_to_deepseek_v3.md). TinySeek has a readable bias update and MTP head, not FP8 kernels, DualPipe or distributed routing.

## 6. TinySeek supplemental evidence

Bias routing moves PPL `2.009 -> 2.024` and load CV `0.075 -> 0.081`, so it does not beat `aux=0.01` locally. MTP gives `2.207 +/- 0.017` versus `2.196 +/- 0.034` and increases VRAM `0.197 -> 0.246 GB`, but it was measured on a rejected MLA-style branch.

![TinySeek VRAM](../../experiments/architecture_lab_runs/figures/architecture_vram.svg)

This is **insufficient/incomparable evidence** for the paper-scale claim; the local result must not be used to assemble a cross-branch winner.

## 7. Claim/evidence table

| Paper claim | Paper evidence | TinySeek evidence | Relation | Boundary |
| --- | --- | --- | --- | --- |
| Bias balances experts without auxiliary loss | Table 5 and Figure 9 | Bias load CV slightly worse | Direction different: not directly transferable | Different scale, branch and tuning |
| MTP gives stable gains | Table 4 | Small mean difference on rejected branch | Insufficient/incomparable | Not measured on selected main branch |
| MLA+MoE scales with systems co-design | Architecture, cost, FP8/DualPipe sections | Readable single-device path | Paper-only | No systems reproduction |

## 8. Conclusion

V3 decomposes large-scale MoE bottlenecks into routing, supervision, communication, precision and deployment experiments. TinySeek teaches the interfaces and preserves the evidence boundary.

## 9. Sources and next paper

Paper: [arXiv:2412.19437](https://arxiv.org/abs/2412.19437). Continue to [DeepSeek-R1](../05-deepseek-r1/README.md).
