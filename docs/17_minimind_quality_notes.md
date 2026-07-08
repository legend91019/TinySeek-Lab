# 17. What TinySeek-Lab Learns from MiniMind

MiniMind is strong because it feels like a complete learning product. A reader
quickly sees:

- what the project can train;
- roughly how much time, money, and hardware it needs;
- where the data comes from;
- how to train, evaluate, and run inference;
- what visual results already exist;
- what deployment paths are available later.

TinySeek-Lab has a different theme: it follows DeepSeek's LM research path
through dense baselines, recipes, MoE, MLA, SFT, and GRPO mini. But the repository
experience should still learn from MiniMind's result-first and loop-complete
style.

## Already Improved

| Area | TinySeek-Lab now |
| --- | --- |
| Result entrance | README now has a `Current Results` section near the top |
| Report hub | Added `experiments/README.md` and `experiments/README_zh.md` |
| Cost story | Logs GPU hours, rental cost, VRAM, tokens, and rough FLOPs |
| Figures | Generates PPL, VRAM, cost, sweep-loss, and VRAM-vs-PPL SVGs |
| Full loop | TinyStories -> dense/sweep/MoE/MLA/SFT/GRPO -> eval -> report |
| Code teaching | Added dense-LM-from-scratch and training-loop chapters |
| Navigation | Each chapter has previous/next/index links |

## Still Missing

| Area | Why it matters | Suggested work |
| --- | --- | --- |
| More visible quick runs | New readers want to run before reading papers | Add CPU smoke, small-GPU run, and 4090 run paths |
| Data recommendations | Readers should not get stuck on data processing | Provide ready-made HF dataset choices and commands |
| Stronger mini eval | PPL/addition/format score is too weak | Add arithmetic, copy, QA, format, and PPL slices |
| MoE visualization | MoE teaching needs routing/load-balance evidence | Generate routing histograms from expert-load JSON |
| Post-training result | GRPO mini currently teaches shape only | Strengthen cold-start SFT before GRPO |
| Checkpoint guidance | Readers need to know which checkpoints matter | Mark smoke checkpoints vs tutorial-start checkpoints |
| Front-page polish | GitHub users scan first | Add badges, result table, compact paths, and key figures |

## What Not to Copy Blindly

TinySeek-Lab is not trying to be the shortest path to a tiny chatbot. Its core is
to teach DeepSeek-style LM research moves. So it should keep these boundaries:

- do not shift focus to multimodal, agents, tool use, or deployment ecosystems;
- do not trade clear research variables for a prettier demo;
- do not present toy GRPO as serious RL performance;
- do not let complex data engineering hide the model and recipe lessons.

## Next Polish Priorities

1. Add three quick paths to README: CPU smoke, small-GPU teaching run, 4090 run.
2. Add a front-page result table: params, tokens, PPL, VRAM, cost.
3. Generate MoE expert-load histograms.
4. Add line-by-line SFT masking explanation.
5. Add line-by-line GRPO objective explanation.
6. Strengthen arithmetic/cold-start data and rerun post-training comparison.

<!-- tinyseek-nav -->

---

Previous: [v1 Training Runbook](14_v1_training_runbook.md) | [Tutorial Index](README.md)
