# DeepSeek-V3 sources

Paper: *DeepSeek-V3 Technical Report* (arXiv:2412.19437).

Source: <https://arxiv.org/pdf/2412.19437>

| Local asset | Paper source | Page | Reading note |
| --- | --- | ---: | --- |
| `paper-fig-2-v3-architecture.png` | Figure 2, basic architecture | 7 | V3 keeps MLA and DeepSeekMoE, then adds balancing and MTP details. |
| `paper-fig-3-mtp.png` | Figure 3, MTP implementation | 10 | MTP adds future-token prediction heads while retaining the main objective. |
| `paper-table-4-mtp-ablation.png` | Table 4, MTP ablation | 26 | The paper reports consistent gains from the MTP strategy. |
| `paper-table-5-balance-ablation.png` | Table 5, auxiliary-loss-free balancing ablation | 27 | The paper compares auxiliary-loss and bias-based load balancing. |
| `paper-fig-9-expert-load.png` | Figure 9, expert load | 28 | The load plot visualizes batch-wise balancing behavior. |

