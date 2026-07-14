# Mini Eval

`mini_eval.py` is intentionally small. It is not a benchmark suite; it is a
teaching tool that checks whether a checkpoint changed in visible ways.

Metrics:

- Perplexity on a JSONL text file.
- Addition exact match for a few verifiable prompts.
- Exact-copy accuracy for short strings.
- Keyword QA accuracy for simple LM-training concepts.
- Format-following score for instruction-style prompts.

Example:

```bash
python eval/mini_eval.py \
  --config configs/tiny_sft.json \
  --ckpt out/tiny_sft_last.pt \
  --data data/toy_pretrain.jsonl \
  --out out/mini_eval.json
```
