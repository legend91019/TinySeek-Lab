# 代码导读

这一章是读代码路线。建议先读完总体路线图，再读这一章，然后再跑更大的实验。

如果第 12 章讲“最初的模型怎么写出来”，第 16 章讲“训练程序怎么把模型变成
checkpoint 和报告”，那这一章就是两者之间的代码地图。

## 阅读顺序

1. `model/tinyseek_dense.py`：最小 dense decoder-only LM。
2. `model/tinyseek.py`：带 GQA、MoE、教学版 MLA 的实验模型。
3. `dataset/lm_dataset.py`：预训练、SFT、prompt 数据集。
4. `trainer/train_pretrain.py`：base model 预训练。
5. `trainer/train_sft.py`：SFT 和 cold-start 格式训练。
6. `trainer/train_grpo.py`：rule-based GRPO mini。
7. `eval/mini_eval.py`：loss、加法、格式遵循的小评测。
8. `scripts/generate_v1_report_assets.py`：把机器可读结果变成表格和 SVG 图。

## `model/tinyseek_dense.py`

这个文件是教学版模型。它的目标是让读者像从零写 GPT 一样理解 dense LM：

- config dataclass 保存模型宽度、深度、head 数、序列长度和 dropout。
- token embedding 把 token id 变成 hidden state。
- 每个 block 做 causal self-attention 和 MLP。
- 最后的 LM head 预测下一个 token。

最重要的是张量形状：

```text
input_ids [batch, seq]
-> embeddings [batch, seq, hidden]
-> blocks [batch, seq, hidden]
-> logits [batch, seq, vocab]
```

先读这个文件，再读 `model/tinyseek.py`。它先拿掉 DeepSeek-style 的复杂升级，只保留语言模型训练目标。

## `model/tinyseek.py`

这是主实验模型。它保留 decoder-only 骨架，但加入教程主线里的架构升级。

### `TinySeekConfig`

配置控制模型结构：

- `hidden_size`、`num_layers`、`num_heads`：dense Transformer 规模。
- `num_kv_heads`：小于 `num_heads` 时就是 GQA。
- `attention_impl`：选择普通 attention 或教学版 MLA。
- `use_moe`、`num_experts`、`top_k`：把 FFN 切换成 MoE。

### `RMSNorm`

RMSNorm 用 hidden state 的均方根幅度做归一化。它比 LayerNorm 更轻，是现代语言模型常见组件。

### RoPE 工具函数

`precompute_rope`、`rotate_half`、`apply_rope` 实现旋转位置编码。位置是在 attention 分数计算前注入到 query/key 向量里的。

### `CausalSelfAttention`

这个模块实现：

- Q projection。
- K/V projection。
- GQA 里的 KV 复制。
- 教学版 MLA：先压到 latent，再重建 K/V。
- PyTorch `scaled_dot_product_attention`，并设置 `is_causal=True`。

对应 DeepSeek 路线：

- GQA 降低 KV cache 成本。
- MLA 进一步把 KV 信息压到 latent path。

### `SwiGLU` 和 `DenseFFN`

dense FFN 使用 SwiGLU：

```text
SwiGLU(x) = W2(silu(W1 x) * W3 x)
```

这是 MLP 升级章节的核心：从旧式 ReLU/GELU MLP 升级到现代 LM block。

### `MoEFFN`

MoE 层对应 DeepSeekMoE 章节：

- router 给每个 token 分配专家概率。
- 每个 token 选择 `top_k` 个专家。
- 专家输出加权求和。
- shared expert 表示共享知识通道。
- auxiliary loss 用来缓解 routing collapse。

教程里要重点比较总参数量和激活参数量。MoE 可以总参数很多，但每个 token 只激活其中一小部分。

### `TinySeekForCausalLM`

完整模型会 tie embedding 和 LM head 权重，计算 next-token cross entropy，并提供：

- `generate`：简单自回归生成。
- `parameter_count`：总参数量。
- `activated_parameter_estimate`：dense 和 MoE 的计算量对比。

## `dataset/lm_dataset.py`

TinySeek 故意保持数据层简单。

### `JsonlTextDataset`

输入格式：

```json
{"text": "raw text for language modeling"}
```

它会用 byte tokenizer 编码文本，pad 到 `max_seq_len`，并把 pad label 设成 `-100`，让 cross entropy 忽略 pad。

### `JsonlInstructionDataset`

输入格式：

```json
{"prompt": "question", "response": "answer"}
```

内部会格式化成：

```text
### Instruction
...

### Response
...
```

prompt token 会被 mask 成 `-100`，只有 response token 参与 SFT loss。这就是本仓里 cold-start SFT 的最小实践：先教模型可读、规整的回答格式，再做 RL。

### `JsonlPromptDataset`

GRPO mini 使用这个数据集。它保存 prompt 和可验证 answer，比如简单算术题。

## `trainer/train_pretrain.py`

这是 base model 训练循环：

1. 读取 config 和 dataset。
2. 构建模型和 optimizer。
3. 做 next-token prediction。
4. 如果 `dtype` 是 `bfloat16` 或 `float16`，启用 AMP。
5. 支持 gradient accumulation。
6. 保存 checkpoint 和成本摘要。

成本摘要会记录 GPU 名称、峰值显存、训练时间、估算费用、token 数和粗略 FLOPs。这样每次实验都能写进报告。

trainer 还会在 validation 点写 `out/<run_name>_history.jsonl`。这些行故意保持
为简单 JSON，后续章节就可以不用外部 experiment tracker 也能画 loss 曲线。

## `trainer/train_sft.py`

SFT 复用同一套模型和优化器，只是换了数据集：

- 预训练：预测所有非 pad 文本 token。
- SFT：只预测 response token。

SFT 用来教：

- instruction 格式；
- 简洁回答；
- cold-start reasoning 风格；
- 特定领域的回答习惯。

## `trainer/train_grpo.py`

这是教学版 GRPO，不是工业级 RL。

对每个 prompt：

1. 采样一组 completion。
2. 用规则 reward 打分。
3. 在组内归一化 reward。
4. 提高高于组均值 completion 的 log probability。
5. 加一个 reference-model KL proxy 约束。

当前 reward 面向算术：最终整数完全正确给满分；输出数字或 answer-like 格式给少量 shaping 分。

## `eval/mini_eval.py`

Mini eval 给快速反馈：

- JSONL 文本上的 perplexity。
- 加法 exact match。
- 格式遵循分数。

它不是 benchmark，只是教程实验的 sanity check。

## 读完以后跑什么

完整端到端程序流见：
[训练主循环：从 Config 到 Checkpoint](16_training_loop_from_config_to_checkpoint.md)。

读完代码后，可以跑 v1 runbook：

```bash
python scripts/prepare_toy_data.py --out data/toy_pretrain.jsonl
python trainer/train_pretrain.py --config configs/tiny_dense.json --data data/toy_pretrain.jsonl --max_steps 20
python scripts/prepare_toy_sft_data.py --out data/toy_sft.jsonl
python trainer/train_sft.py --config configs/tiny_sft.json --data data/toy_sft.jsonl --init_ckpt out/tiny_dense_last.pt --max_steps 20
```

<!-- tinyseek-nav -->

---

上一篇: [训练主循环](16_training_loop_from_config_to_checkpoint.md) | [教程目录](README.md) | 下一篇: [阶段 0：Dense Baseline](02_stage0_dense_baseline.md)
