# Mini Eval

`mini_eval.py` 不是严肃 benchmark，而是教学用的小评测。它用来观察 checkpoint 是否真的发生了可见变化。

指标：

- JSONL 文本上的 perplexity。
- 加法题 exact match。
- 指令格式遵循分数。

示例：

```bash
python eval/mini_eval.py \
  --config configs/tiny_sft.json \
  --ckpt out/tiny_sft_last.pt \
  --data data/toy_pretrain.jsonl \
  --out out/mini_eval.json
```
