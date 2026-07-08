# RTX 4090 v1 Run Manifest

- Execute mode: True
- Dataset path: `data/tinystories.jsonl`
- Hourly rate: 2.18 CNY/hour
- Sweep token target per run: 5000000

| Step | Name | Command |
| ---: | --- | --- |
| 1 | `train_v1_tiny_base` | `/root/miniconda3/bin/python trainer/train_pretrain.py --config configs/tiny_dense.json --data data/tinystories.jsonl --run_name v1_tiny_base --hourly_rate 2.18 --currency CNY --max_steps 200` |
| 2 | `eval_v1_tiny_base` | `/root/miniconda3/bin/python eval/mini_eval.py --config configs/tiny_dense.json --ckpt out/v1_tiny_base_last.pt --data data/tinystories.jsonl --out experiments/v1_4090_plan/eval_v1_tiny_base.json` |
| 3 | `train_v1_dense35` | `/root/miniconda3/bin/python trainer/train_pretrain.py --config configs/medium_dense_35m.json --data data/tinystories.jsonl --run_name v1_dense35 --hourly_rate 2.18 --currency CNY` |
| 4 | `eval_v1_dense35` | `/root/miniconda3/bin/python eval/mini_eval.py --config configs/medium_dense_35m.json --ckpt out/v1_dense35_last.pt --data data/tinystories.jsonl --out experiments/v1_4090_plan/eval_v1_dense35.json` |
| 5 | `train_v1_dense115` | `/root/miniconda3/bin/python trainer/train_pretrain.py --config configs/medium_dense_115m.json --data data/tinystories.jsonl --run_name v1_dense115 --hourly_rate 2.18 --currency CNY` |
| 6 | `eval_v1_dense115` | `/root/miniconda3/bin/python eval/mini_eval.py --config configs/medium_dense_115m.json --ckpt out/v1_dense115_last.pt --data data/tinystories.jsonl --out experiments/v1_4090_plan/eval_v1_dense115.json` |
| 7 | `train_v1_sweep_bs16_lr3e-4` | `/root/miniconda3/bin/python trainer/train_pretrain.py --config experiments/v1_4090_plan/configs/v1_sweep_bs16_lr3e-4.json --data data/tinystories.jsonl --run_name v1_sweep_bs16_lr3e-4 --hourly_rate 2.18 --currency CNY --max_steps 1221` |
| 8 | `eval_v1_sweep_bs16_lr3e-4` | `/root/miniconda3/bin/python eval/mini_eval.py --config experiments/v1_4090_plan/configs/v1_sweep_bs16_lr3e-4.json --ckpt out/v1_sweep_bs16_lr3e-4_last.pt --data data/tinystories.jsonl --out experiments/v1_4090_plan/eval_v1_sweep_bs16_lr3e-4.json` |
| 9 | `train_v1_sweep_bs16_lr6e-4` | `/root/miniconda3/bin/python trainer/train_pretrain.py --config experiments/v1_4090_plan/configs/v1_sweep_bs16_lr6e-4.json --data data/tinystories.jsonl --run_name v1_sweep_bs16_lr6e-4 --hourly_rate 2.18 --currency CNY --max_steps 1221` |
| 10 | `eval_v1_sweep_bs16_lr6e-4` | `/root/miniconda3/bin/python eval/mini_eval.py --config experiments/v1_4090_plan/configs/v1_sweep_bs16_lr6e-4.json --ckpt out/v1_sweep_bs16_lr6e-4_last.pt --data data/tinystories.jsonl --out experiments/v1_4090_plan/eval_v1_sweep_bs16_lr6e-4.json` |
| 11 | `train_v1_sweep_bs32_lr3e-4` | `/root/miniconda3/bin/python trainer/train_pretrain.py --config experiments/v1_4090_plan/configs/v1_sweep_bs32_lr3e-4.json --data data/tinystories.jsonl --run_name v1_sweep_bs32_lr3e-4 --hourly_rate 2.18 --currency CNY --max_steps 611` |
| 12 | `eval_v1_sweep_bs32_lr3e-4` | `/root/miniconda3/bin/python eval/mini_eval.py --config experiments/v1_4090_plan/configs/v1_sweep_bs32_lr3e-4.json --ckpt out/v1_sweep_bs32_lr3e-4_last.pt --data data/tinystories.jsonl --out experiments/v1_4090_plan/eval_v1_sweep_bs32_lr3e-4.json` |
| 13 | `train_v1_sweep_bs32_lr6e-4` | `/root/miniconda3/bin/python trainer/train_pretrain.py --config experiments/v1_4090_plan/configs/v1_sweep_bs32_lr6e-4.json --data data/tinystories.jsonl --run_name v1_sweep_bs32_lr6e-4 --hourly_rate 2.18 --currency CNY --max_steps 611` |
| 14 | `eval_v1_sweep_bs32_lr6e-4` | `/root/miniconda3/bin/python eval/mini_eval.py --config experiments/v1_4090_plan/configs/v1_sweep_bs32_lr6e-4.json --ckpt out/v1_sweep_bs32_lr6e-4_last.pt --data data/tinystories.jsonl --out experiments/v1_4090_plan/eval_v1_sweep_bs32_lr6e-4.json` |
| 15 | `train_v1_moe_activated35` | `/root/miniconda3/bin/python trainer/train_pretrain.py --config configs/medium_moe_activated_35m.json --data data/tinystories.jsonl --run_name v1_moe_activated35 --hourly_rate 2.18 --currency CNY` |
| 16 | `eval_v1_moe_activated35` | `/root/miniconda3/bin/python eval/mini_eval.py --config configs/medium_moe_activated_35m.json --ckpt out/v1_moe_activated35_last.pt --data data/tinystories.jsonl --out experiments/v1_4090_plan/eval_v1_moe_activated35.json` |
| 17 | `train_v1_tiny_mla` | `/root/miniconda3/bin/python trainer/train_pretrain.py --config configs/tiny_mla.json --data data/tinystories.jsonl --run_name v1_tiny_mla --hourly_rate 2.18 --currency CNY --max_steps 200` |
| 18 | `eval_v1_tiny_mla` | `/root/miniconda3/bin/python eval/mini_eval.py --config configs/tiny_mla.json --ckpt out/v1_tiny_mla_last.pt --data data/tinystories.jsonl --out experiments/v1_4090_plan/eval_v1_tiny_mla.json` |
| 19 | `prepare_sft_data` | `/root/miniconda3/bin/python scripts/prepare_toy_sft_data.py --out data/v1_toy_sft.jsonl` |
| 20 | `train_v1_tiny_sft` | `/root/miniconda3/bin/python trainer/train_sft.py --config configs/tiny_sft.json --data data/v1_toy_sft.jsonl --init_ckpt out/v1_tiny_base_last.pt --run_name v1_tiny_sft --hourly_rate 2.18 --currency CNY` |
| 21 | `eval_v1_tiny_sft` | `/root/miniconda3/bin/python eval/mini_eval.py --config configs/tiny_sft.json --ckpt out/v1_tiny_sft_last.pt --data data/tinystories.jsonl --out experiments/v1_4090_plan/eval_v1_tiny_sft.json` |
| 22 | `prepare_grpo_data` | `/root/miniconda3/bin/python scripts/prepare_toy_grpo_data.py --out data/v1_toy_grpo.jsonl` |
| 23 | `train_v1_tiny_grpo` | `/root/miniconda3/bin/python trainer/train_grpo.py --config configs/tiny_grpo.json --data data/v1_toy_grpo.jsonl --init_ckpt out/v1_tiny_sft_last.pt --run_name v1_tiny_grpo --hourly_rate 2.18 --currency CNY` |
| 24 | `eval_v1_tiny_grpo` | `/root/miniconda3/bin/python eval/mini_eval.py --config configs/tiny_grpo.json --ckpt out/v1_tiny_grpo_last.pt --data data/tinystories.jsonl --out experiments/v1_4090_plan/eval_v1_tiny_grpo.json` |
| 25 | `summarize_costs` | `/root/miniconda3/bin/python scripts/summarize_costs.py --input_dir out --markdown_out experiments/v1_4090_plan/cost_summary.md --csv_out experiments/v1_4090_plan/cost_summary.csv` |
