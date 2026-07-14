# DeepSeek Evolution Tutorial Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a beginner-friendly, bilingual code course that evolves one runnable small LM from DeepSeek LLM through DeepSeekMoE, DeepSeek-V2, DeepSeek-V3, and then connects it to the existing R1-style post-training path.

**Architecture:** Keep four self-contained teaching models in `model/stages/`, all exposing the same causal-LM interface, while extending `model/tinyseek.py` as the configurable experiment implementation. Add CPU smoke checks, fair experiment configs, pending-GPU reports, and paired Chinese/English chapters that explain each adjacent code change.

**Tech Stack:** Python 3.10+, PyTorch 2.1+, JSON configs, Markdown, Mermaid, repository SVG reports.

## Global Constraints

- Language-model-only scope; no multimodal, vision, video, OCR, embodied, or agent mainline.
- Stage 0-3 must be complete runnable models, not disconnected snippets.
- The unified model remains the formal training implementation.
- Paper claims, measured TinySeek results, and pending hypotheses must be labeled separately.
- No production claims for educational MLA, single-device MoE, or mini GRPO.
- Chinese and English core chapters must have matching coverage.
- GPU-only result cells remain `Pending GPU run` until measured.

---

### Task 1: Stage Package and Stage 0 Regression Tests

**Files:**
- Create: `model/stages/__init__.py`
- Create: `model/stages/stage0_deepseek_llm.py`
- Create: `tests/stage_models_test.py`

**Interfaces:**
- Produces: `Stage0Config`, `Stage0DeepSeekLM`.
- `Stage0DeepSeekLM.forward(input_ids, labels=None) -> dict[str, Tensor | None]` with `logits`, `loss`, and scalar `aux_loss`.

- [ ] **Step 1: Write the Stage 0 smoke test**

```python
def test_stage0() -> None:
    cfg = Stage0Config(vocab_size=64, max_seq_len=16, hidden_size=32, num_layers=2, num_heads=4, num_kv_heads=2)
    model = Stage0DeepSeekLM(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 12))
    out = model(x, x)
    assert out["logits"].shape == (2, 12, cfg.vocab_size)
    assert torch.isfinite(out["loss"])
```

- [ ] **Step 2: Run the test and confirm the missing import failure**

Run: `python tests/stage_models_test.py`

Expected: import failure for `model.stages` before implementation.

- [ ] **Step 3: Implement the complete Stage 0 model**

Copy the readable Dense baseline into a self-contained file and expose the stable result shape:

```python
return {
    "logits": logits,
    "loss": loss,
    "aux_loss": logits.new_zeros(()),
}
```

The file must contain config, RMSNorm, RoPE helpers, GQA attention, SwiGLU, block, tied LM head, shifted CE loss, parameter count, and KV-cache element estimate.

- [ ] **Step 4: Export Stage 0 and run static checks**

Run: `python -m py_compile model/stages/stage0_deepseek_llm.py tests/stage_models_test.py`

Expected: exit code 0.

- [ ] **Step 5: Commit**

```bash
git add model/stages tests/stage_models_test.py
git commit -m "feat: add runnable DeepSeek LLM teaching stage"
```

### Task 2: DeepSeekMoE Teaching Stage

**Files:**
- Create: `model/stages/stage1_deepseek_moe.py`
- Modify: `model/stages/__init__.py`
- Modify: `tests/stage_models_test.py`

**Interfaces:**
- Produces: `Stage1Config`, `Stage1DeepSeekMoE`, `FineGrainedMoE`.
- `expert_load_summary() -> dict` includes `total_counts`, `total_fractions`, `top_k`, and `num_experts`.

- [ ] **Step 1: Add MoE invariants to the smoke test**

```python
def test_stage1() -> None:
    cfg = Stage1Config(vocab_size=64, max_seq_len=16, hidden_size=32, num_layers=2, num_heads=4, num_kv_heads=2, num_routed_experts=4, top_k=2)
    model = Stage1DeepSeekMoE(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 12))
    out = model(x, x)
    stats = model.expert_load_summary()
    assert out["logits"].shape == (2, 12, cfg.vocab_size)
    assert out["aux_loss"].ndim == 0
    assert sum(stats["total_counts"]) == cfg.num_layers * x.numel() * cfg.top_k
    assert model.activated_parameter_estimate() < model.parameter_count()
```

- [ ] **Step 2: Implement fine-grained routed and shared experts**

Use a smaller per-expert intermediate width and sparse top-k dispatch:

```python
expert_hidden = int(config.hidden_size * config.expert_ffn_multiplier)
self.router = nn.Linear(config.hidden_size, config.num_routed_experts, bias=False)
self.routed = nn.ModuleList([SwiGLU(config.hidden_size, expert_hidden) for _ in range(config.num_routed_experts)])
self.shared = nn.ModuleList([SwiGLU(config.hidden_size, expert_hidden) for _ in range(config.num_shared_experts)])
```

Return weighted routed outputs, shared outputs, a scalar balance loss, and load counters.

- [ ] **Step 3: Keep the model complete**

Stage 1 must include the same embedding, attention, block stack, norm, head, and shifted loss as Stage 0. Only the FFN branch changes.

- [ ] **Step 4: Run stage tests**

Run: `python tests/stage_models_test.py`

Expected: `stage model tests ok`.

- [ ] **Step 5: Commit**

```bash
git add model/stages tests/stage_models_test.py
git commit -m "feat: teach DeepSeekMoE as a full model stage"
```

### Task 3: DeepSeek-V2 MLA Teaching Stage

**Files:**
- Create: `model/stages/stage2_deepseek_v2.py`
- Modify: `model/stages/__init__.py`
- Modify: `tests/stage_models_test.py`

**Interfaces:**
- Produces: `Stage2Config`, `Stage2DeepSeekV2`, `EducationalMLA`.
- `kv_cache_elements_per_token() -> int` reports the theoretical latent-plus-RoPE cache width.

- [ ] **Step 1: Add MLA shape and cache tests**

```python
def test_stage2() -> None:
    cfg = Stage2Config(vocab_size=64, max_seq_len=16, hidden_size=32, num_layers=2, num_heads=4, num_kv_heads=2, kv_lora_rank=12, qk_rope_head_dim=4)
    model = Stage2DeepSeekV2(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 12))
    out = model(x, x)
    assert out["logits"].shape == (2, 12, cfg.vocab_size)
    assert model.kv_cache_elements_per_token() == cfg.kv_lora_rank + cfg.qk_rope_head_dim
```

- [ ] **Step 2: Implement content compression and decoupled RoPE paths**

```python
compressed_kv = self.kv_down(x)
k_content = self.k_up(compressed_kv)
v = self.v_up(compressed_kv)
k_rope = apply_rope(self.k_rope_proj(x), cos, sin)
q_content, q_rope = split_query(self.q_proj(x))
q = torch.cat((q_content, apply_rope(q_rope, cos, sin)), dim=-1)
k = torch.cat((k_content, k_rope), dim=-1)
```

Document in code that training reconstructs K/V and that this repository does not yet implement a production cached-decoding kernel.

- [ ] **Step 3: Preserve DeepSeekMoE in each block**

The Stage 2 block combines `EducationalMLA` and `FineGrainedMoE`; it must not regress to a dense FFN.

- [ ] **Step 4: Run smoke and syntax checks**

Run: `python tests/stage_models_test.py`

Expected: `stage model tests ok`.

- [ ] **Step 5: Commit**

```bash
git add model/stages tests/stage_models_test.py
git commit -m "feat: add DeepSeek-V2 MLA teaching stage"
```

### Task 4: DeepSeek-V3 Routing and MTP

**Files:**
- Create: `model/stages/stage3_deepseek_v3.py`
- Modify: `model/stages/__init__.py`
- Modify: `model/tinyseek.py`
- Modify: `trainer/train_pretrain.py`
- Modify: `tests/stage_models_test.py`
- Modify: `tests/smoke_test.py`

**Interfaces:**
- Config fields: `router_balance_strategy`, `router_bias_update_rate`, `mtp_depth`, `mtp_loss_weight`.
- `update_router_bias() -> None` updates bias from the most recent routed-token counts.
- Forward result adds `mtp_loss` and `lm_loss`; total `loss` remains the optimized language-model objective.

- [ ] **Step 1: Add router direction and MTP tests**

```python
before = router.expert_bias.clone()
router.last_expert_counts.copy_(torch.tensor([20, 2, 2, 2]))
router.update_bias()
assert router.expert_bias[0] < before[0]
assert torch.all(router.expert_bias[1:] > before[1:])

out = model(x, x)
assert out["mtp_loss"].ndim == 0
assert torch.isfinite(out["loss"])
assert out["logits"].shape == (2, 12, cfg.vocab_size)
```

- [ ] **Step 2: Implement selection bias without changing affinity weights**

```python
affinity = torch.sigmoid(self.gate(flat))
selection_score = affinity + self.expert_bias
top_i = torch.topk(selection_score, k=self.top_k, dim=-1).indices
top_p = affinity.gather(1, top_i)
top_p = top_p / top_p.sum(dim=-1, keepdim=True)
```

Bias update decreases overloaded experts and increases underloaded experts using the configured update rate.

- [ ] **Step 3: Implement one or more sequential MTP modules**

Each MTP module combines normalized previous hidden state with the embedding of the next token, projects the concatenation, applies one Transformer block, and predicts another future token through the shared LM head. Mask `-100` labels and return the mean MTP loss.

- [ ] **Step 4: Extend the unified experiment model**

Add V3 config fields with backward-compatible defaults. Existing configs must produce the same Dense/MoE/MLA behavior. Add `mtp_loss`, `lm_loss`, `update_router_bias`, and balance strategy without changing old checkpoint key names unless required.

- [ ] **Step 5: Update the trainer objective and logging**

```python
loss = out["loss"] + out["aux_loss"]
...
scaler.step(optimizer)
scaler.update()
model.update_router_bias()
```

History rows include `lm_loss`, `mtp_loss`, `aux_loss`, and router stats.

- [ ] **Step 6: Run smoke tests**

Run: `python tests/smoke_test.py`

Run: `python tests/stage_models_test.py`

Expected: both scripts print their success messages.

- [ ] **Step 7: Commit**

```bash
git add model trainer/train_pretrain.py tests
git commit -m "feat: add DeepSeek-V3 routing and MTP lab"
```

### Task 5: Fair Architecture Experiment Lab

**Files:**
- Create: `configs/architecture_lab/dense_mha.json`
- Create: `configs/architecture_lab/dense_gqa.json`
- Create: `configs/architecture_lab/moe_aux.json`
- Create: `configs/architecture_lab/moe_bias.json`
- Create: `configs/architecture_lab/v3_mtp.json`
- Create: `scripts/inspect_stage_models.py`
- Create: `experiments/06_architecture_evolution_plan_zh.md`
- Create: `experiments/06_architecture_evolution_plan.md`

**Interfaces:**
- `inspect_stage_models.py` prints and optionally writes a JSON list with stage name, total params, activated params, logits shape, loss keys, and KV cache estimate.

- [ ] **Step 1: Add matched JSON configs**

All configs use the same vocabulary, hidden size, layer count, sequence length, batch size, optimizer settings, max steps, and seed. Only the target architecture fields differ.

- [ ] **Step 2: Implement the CPU inspection script**

```python
rows = [inspect_stage(name, model, config) for name, model, config in build_stages()]
print(json.dumps(rows, indent=2, ensure_ascii=False))
```

The script uses a fixed random input and performs a no-grad forward pass.

- [ ] **Step 3: Write paired experiment plans**

Tables contain hypothesis, controlled variables, command, required metrics, paper claim, TinySeek status, and failure interpretation. Result values use `Pending GPU run` where no measurement exists.

- [ ] **Step 4: Validate configs and script**

Run: `python -c "import json, pathlib; [json.loads(p.read_text()) for p in pathlib.Path('configs/architecture_lab').glob('*.json')]"`

Run: `python scripts/inspect_stage_models.py`

- [ ] **Step 5: Commit**

```bash
git add configs/architecture_lab scripts/inspect_stage_models.py experiments/06_architecture_evolution_plan*
git commit -m "experiments: add fair DeepSeek architecture lab"
```

### Task 6: Bilingual Code Evolution Course

**Files:**
- Create: `docs/zh/20_architecture_evolution_overview.md`
- Create: `docs/zh/21_from_dense_to_deepseek_moe.md`
- Create: `docs/zh/22_from_moe_to_deepseek_v2.md`
- Create: `docs/zh/23_from_v2_to_deepseek_v3.md`
- Create matching English files under `docs/`.
- Modify: `docs/zh/04_stage2_block_upgrades.md`
- Modify: `docs/zh/05_stage3_moe.md`
- Modify: `docs/zh/06_stage4_mla.md`
- Modify: `docs/zh/07_stage5_sft_cold_start.md`
- Modify: `docs/zh/08_stage6_grpo_mini.md`
- Modify matching English files.

**Interfaces:**
- Every architecture chapter links the stage file, unified implementation, matched config, tests, previous chapter, next chapter, and index.

- [ ] **Step 1: Write the evolution overview**

Include a timeline, problem/solution/trade-off table, model-vs-training distinction, and exact learning order.

- [ ] **Step 2: Write the Dense-to-MoE code chapter**

Explain the whole Stage 1 file, token routing with a concrete `[batch, seq, hidden]` example, fine-grained expert dimensions, shared experts, dispatch loop, balance loss, parameter accounting, and fair experiments.

- [ ] **Step 3: Write the MoE-to-V2 code chapter**

Explain MHA/GQA cache, latent compression, decoupled RoPE, training reconstruction, theoretical cache accounting, simplifications, and experiments.

- [ ] **Step 4: Write the V2-to-V3 code chapter**

Explain why auxiliary loss can conflict with LM learning, how selection bias works, why MTP adds training signal, label shifts, trainer integration, and ablations.

- [ ] **Step 5: Upgrade old short chapters into labs**

Remove stale status text, add prerequisites, controlled variables, commands, metrics, expected failure modes, links to detailed code chapters, and explicit result status.

- [ ] **Step 6: Produce matching English chapters**

English coverage must include every code and experiment concept. Chinese prose may be more expanded.

- [ ] **Step 7: Commit**

```bash
git add docs
git commit -m "docs: teach DeepSeek architecture evolution in code"
```

### Task 7: README, Navigation, and Result Presentation

**Files:**
- Modify: `README_zh.md`
- Modify: `README.md`
- Modify: `docs/zh/README.md`
- Modify: `docs/README.md`
- Modify: `scripts/refresh_doc_nav.py`
- Modify: `experiments/README_zh.md`
- Modify: `experiments/README.md`
- Modify: `docs/zh/04_current_progress.md`

- [ ] **Step 1: Reshape the README first viewport**

Show repository promise, LM-only scope, four-generation code route, measured 4090 cost, and direct start links before long setup text.

- [ ] **Step 2: Add visible course and evidence tables**

Map each generation to paper, complete stage file, key change, code chapter, and experiment. Embed at least one existing measured SVG result.

- [ ] **Step 3: Update tutorial indexes and navigation order**

Put architecture overview and Stage 0-3 code chapters before training experiments. Extend `EN_ORDER` and `ZH_ORDER` so every core chapter has previous/next/index links.

- [ ] **Step 4: Update experiment hub and progress report**

List the architecture lab as prepared but GPU-pending. Correct progress percentages and separate code completion from experimental evidence.

- [ ] **Step 5: Refresh navigation and inspect links**

Run: `python scripts/refresh_doc_nav.py`

Run a repository-local Markdown link checker script or equivalent path scan.

- [ ] **Step 6: Commit**

```bash
git add README.md README_zh.md docs experiments scripts/refresh_doc_nav.py
git commit -m "docs: polish TinySeek learning path and results"
```

### Task 8: Final Verification and Publication

**Files:**
- Modify only files needed to fix verification findings.

- [ ] **Step 1: Parse all Python files**

Run: `python -c "import ast, pathlib; files=list(pathlib.Path('.').glob('**/*.py')); [ast.parse(p.read_text(encoding='utf-8'), filename=str(p)) for p in files]; print(f'AST ok: {len(files)} files')"`

- [ ] **Step 2: Parse all JSON files**

Run: `python -c "import json, pathlib; files=list(pathlib.Path('configs').glob('**/*.json'))+list(pathlib.Path('experiments').glob('**/*.json')); [json.loads(p.read_text(encoding='utf-8')) for p in files]; print(f'JSON ok: {len(files)} files')"`

- [ ] **Step 3: Run available smoke tests**

Run: `python tests/stage_models_test.py`

Run: `python tests/smoke_test.py`

If local Python lacks PyTorch, record that limitation and run these on the next GPU host before filling result tables.

- [ ] **Step 4: Check docs and working tree**

Run: `git diff --check`

Run the Markdown link scan and bilingual pair scan. Expected: no broken local links and no missing core translation.

- [ ] **Step 5: Review claims**

Search `Pending GPU run`, `已完成`, `proved`, and `复现`; confirm no pending architecture experiment is described as measured.

- [ ] **Step 6: Commit and push**

```bash
git add .
git commit -m "feat: complete DeepSeek evolution teaching course"
git push origin main
```
