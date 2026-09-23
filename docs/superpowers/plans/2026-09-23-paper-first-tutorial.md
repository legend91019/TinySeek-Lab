# Paper-First Tutorial Restructure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reorganize TinySeek-Lab around five DeepSeek paper tutorials whose articles combine the paper's claims, paper figures/tables, existing TinySeek implementation, and existing small-scale evidence without running new experiments.

**Architecture:** Add a `papers/` documentation tree as the only recommended reading path. Each paper has one Chinese article and one English mirror following the same evidence-first template; paper figures live in `papers/assets/<paper>/`, while TinySeek figures and reports remain in `experiments/` and are linked in place. Existing `course/`, `docs/`, and `experiments/` content remains intact and receives migration notices.

**Tech Stack:** Markdown, existing SVG/JSON/Markdown reports, public arXiv PDFs, PowerShell, Git, existing Python smoke tests.

## Global Constraints

- Do not start GPU jobs, modify experiment configs, rerun evaluations, regenerate existing reports, or change model/trainer code.
- Treat the DeepSeek paper as the primary source; TinySeek results are supplemental evidence only.
- Label every result as paper evidence, TinySeek supplemental evidence, directionally consistent, inconsistent/not directly transferable, or uncovered/incomparable.
- Do not copy full PDFs into the repository; only add selected cited figure/table crops with source metadata.
- Preserve existing source files, experiment outputs, paths, and historical links.
- Chinese articles are the primary teaching text; English articles keep the same section order and conclusion boundaries.
- Use relative Markdown links that work from the repository root and from each paper directory.
- Keep all new files UTF-8 Markdown and use ASCII filenames.

---

### Task 1: Create the paper index and asset metadata contract

**Files:**
- Create: `papers/README_zh.md`
- Create: `papers/README.md`
- Create: `papers/assets/README.md`
- Create: `papers/01-deepseek-llm/README_zh.md`
- Create: `papers/01-deepseek-llm/README.md`
- Create: `papers/02-deepseek-moe/README_zh.md`
- Create: `papers/02-deepseek-moe/README.md`
- Create: `papers/03-deepseek-v2/README_zh.md`
- Create: `papers/03-deepseek-v2/README.md`
- Create: `papers/04-deepseek-v3/README_zh.md`
- Create: `papers/04-deepseek-v3/README.md`
- Create: `papers/05-deepseek-r1/README_zh.md`
- Create: `papers/05-deepseek-r1/README.md`
- Create: `papers/assets/deepseek-llm/SOURCES.md`
- Create: `papers/assets/deepseek-moe/SOURCES.md`
- Create: `papers/assets/deepseek-v2/SOURCES.md`
- Create: `papers/assets/deepseek-v3/SOURCES.md`
- Create: `papers/assets/deepseek-r1/SOURCES.md`

**Interfaces:**
- Produces the only recommended paper reading route and the metadata format consumed by all five articles.

- [ ] **Step 1: Write the Chinese paper index and article stubs**

  Create `papers/README_zh.md` with the five-paper sequence, one-sentence purpose for each paper, the evidence legend, the statement that no new experiments are run, and links to all five Chinese articles. Create each linked `README_zh.md` stub with its paper title and a line stating that the full article is completed in the corresponding article task.

- [ ] **Step 2: Write the English paper index and article stubs**

  Create `papers/README.md` with the same five-paper sequence and links to the English articles. Keep the same evidence legend and scope boundary as the Chinese index. Create each linked `README.md` stub with its paper title and the same completion note.

- [ ] **Step 3: Write the asset metadata contract**

  Create `papers/assets/README.md` specifying that every paper asset needs a source URL, paper title, figure/table number, page or section, local filename, and a short reading note. State that TinySeek charts remain linked from `experiments/`.

- [ ] **Step 4: Create source ledgers**

  Create one `SOURCES.md` in each paper asset directory with the exact public source URL for the paper, citation title, arXiv identifier, and columns for local asset filename, source figure/table, page/section, and article usage. Start each ledger with the paper's citation even before image assets are added.

- [ ] **Step 5: Verify index links and commit**

  Run a repository Markdown link checker or a PowerShell relative-path check against every index link. Expected: all newly referenced article stub paths exist. Commit with `docs: add paper-first reading index`.

### Task 2: Download and curate cited DeepSeek paper figures/tables

**Files:**
- Modify: `papers/assets/deepseek-llm/SOURCES.md`
- Modify: `papers/assets/deepseek-moe/SOURCES.md`
- Modify: `papers/assets/deepseek-v2/SOURCES.md`
- Modify: `papers/assets/deepseek-v3/SOURCES.md`
- Modify: `papers/assets/deepseek-r1/SOURCES.md`
- Create: selected `papers/assets/<paper>/paper-fig-*.png` and `paper-table-*.png` files identified from the public PDFs.

**Interfaces:**
- Produces cited local images used by the five articles. It does not alter numerical content or create new experimental results.

- [ ] **Step 1: Obtain the public PDFs**

  Use these public URLs to download temporary PDFs outside the repository: `https://arxiv.org/pdf/2401.02954`, `https://arxiv.org/pdf/2401.06066`, `https://arxiv.org/pdf/2405.04434`, `https://arxiv.org/pdf/2412.19437`, and `https://arxiv.org/pdf/2501.12948`. Record the exact version URL in each `SOURCES.md`; do not commit the PDF.

- [ ] **Step 2: Inspect paper contents**

  Extract each PDF's table of contents, figure captions, and experiment sections. Select only figures/tables needed to explain the paper's central architecture change, main ablation, and primary result. Do not select decorative or redundant plots.

- [ ] **Step 3: Render and crop selected assets**

  Render selected PDF pages at readable resolution, crop the figure/table without changing values, and save using `paper-fig-<number>-<short-name>.png` or `paper-table-<number>-<short-name>.png`.

- [ ] **Step 4: Complete source ledgers**

  For every image, fill the local filename, source figure/table number, page/section, exact URL, and one-sentence reading note. Mark each asset as a paper-original crop.

- [ ] **Step 5: Verify asset provenance and commit**

  Open each image for visual inspection, confirm labels are readable and unchanged, run `git diff --check`, and commit with `docs: add cited DeepSeek paper figures`.

### Task 3: Write the DeepSeek LLM article

**Files:**
- Modify: `papers/01-deepseek-llm/README_zh.md`
- Modify: `papers/01-deepseek-llm/README.md`
- Modify: `papers/assets/deepseek-llm/SOURCES.md`

**Interfaces:**
- Consumes: DeepSeek LLM PDF notes/assets, `course/s01_dense_baseline`, `course/s02_training_recipe`, `course/s03_gqa`, existing Dense/GQA reports and code.
- Produces: the first complete article in the common template, including links later articles use as the Dense/GQA baseline.

- [ ] **Step 1: Draft the Chinese article skeleton**

  Add the nine required sections: paper定位, generation change, method details, paper experiments, TinySeek implementation, TinySeek evidence, claim/evidence table, chapter conclusion, and citations/entry points.

- [ ] **Step 2: Explain the paper's core method**

  Cover the paper's decoder-only objective, normalization, positional encoding, activation, attention choice, tokenizer/data/training recipe, scaling or ablation methodology, and reported limitations. Mark any production-scale detail that TinySeek does not implement.

- [ ] **Step 3: Embed paper evidence**

  Embed the selected DeepSeek LLM figure/table crops with figure/table numbers, source links, and reading notes from `SOURCES.md`.

- [ ] **Step 4: Map existing TinySeek evidence**

  Link `stage0_deepseek_llm.py`, Dense/GQA configs, the existing architecture report figures, and the existing LR/batch report. State whether each local observation is directionally consistent, only local, or not directly comparable to the paper.

- [ ] **Step 5: Write the English mirror and verify**

  Translate the same claims and boundaries into `README.md`, preserving section order and all evidence labels. Run Markdown link checks and commit with `docs: add DeepSeek LLM paper tutorial`.

### Task 4: Write the DeepSeekMoE article

**Files:**
- Modify: `papers/02-deepseek-moe/README_zh.md`
- Modify: `papers/02-deepseek-moe/README.md`
- Modify: `papers/assets/deepseek-moe/SOURCES.md`

**Interfaces:**
- Consumes: DeepSeekMoE PDF notes/assets, `course/s04_deepseek_moe`, `docs/zh/21_from_dense_to_deepseek_moe.md`, stage1 code, and architecture report outputs.
- Produces: the MoE article linked after the DeepSeek LLM article.

- [ ] **Step 1: Explain the paper's motivation and architecture**

  Cover dense capacity/compute tension, expert segmentation, shared experts, routing, top-k dispatch, load balancing, expert specialization, and the paper's training/inference implications.

- [ ] **Step 2: Summarize the paper experiments**

  Present the paper's key ablations and conclusions with selected paper figures/tables. Distinguish total parameters, activated parameters, expert count, routing and quality metrics.

- [ ] **Step 3: Map TinySeek evidence**

  Embed or link existing MoE load and quality figures, report tables, configs, and `stage1_deepseek_moe.py`. Explain the shared-quality/throughput trade-off and aux-loss evidence without making a universal claim.

- [ ] **Step 4: Write the English mirror and verify**

  Preserve all evidence labels and caveats in the English article. Run link checks and commit with `docs: add DeepSeekMoE paper tutorial`.

### Task 5: Write the DeepSeek-V2 article

**Files:**
- Modify: `papers/03-deepseek-v2/README_zh.md`
- Modify: `papers/03-deepseek-v2/README.md`
- Modify: `papers/assets/deepseek-v2/SOURCES.md`

**Interfaces:**
- Consumes: DeepSeek-V2 PDF notes/assets, `course/s05_mla`, `docs/zh/22_from_moe_to_deepseek_v2.md`, stage2 code, and `v2_*` architecture runs.
- Produces: the V2 article linked after the MoE article.

- [ ] **Step 1: Explain DeepSeek-V2's combined design**

  Cover the relationship between DeepSeekMoE and MLA, latent KV compression, decoupled RoPE, cache accounting, training/inference motivation, and any additional paper components required to understand the reported system.

- [ ] **Step 2: Summarize the paper's experiments**

  Embed the selected paper MLA/compression/quality figures or tables and explain the paper's controls, metrics, and conclusions.

- [ ] **Step 3: Map TinySeek evidence and limitations**

  Link the control/low-rank/MLA reports, `stage2_deepseek_v2.py`, and existing architecture figures. Explicitly state that the teaching forward reconstructs K/V and lacks production cached-decode kernels; frame the local MLA regression as non-transferable disagreement rather than a refutation.

- [ ] **Step 4: Write the English mirror and verify**

  Keep the same method order, evidence table, and caveat language. Run link checks and commit with `docs: add DeepSeek-V2 paper tutorial`.

### Task 6: Write the DeepSeek-V3 article

**Files:**
- Modify: `papers/04-deepseek-v3/README_zh.md`
- Modify: `papers/04-deepseek-v3/README.md`
- Modify: `papers/assets/deepseek-v3/SOURCES.md`

**Interfaces:**
- Consumes: DeepSeek-V3 PDF notes/assets, `course/s06_v3_routing_mtp`, `docs/zh/23_from_moe_to_deepseek_v3.md`, stage3/unified model code, and architecture reports.
- Produces: the V3 article linked after V2.

- [ ] **Step 1: Explain the paper's full contribution set**

  Cover the paper's model architecture, auxiliary-loss-free balancing, expert selection bias, MTP, training stability/efficiency, parallelism or systems details, and the distinction between paper-scale engineering and TinySeek's readable implementation.

- [ ] **Step 2: Summarize the paper's experiments and report**

  Embed selected paper figures/tables for balancing, MTP or scaling evidence. Explain what each experiment isolates and what the paper concludes.

- [ ] **Step 3: Map TinySeek evidence**

  Link aux/bias and MTP reports, figures, configs, `stage3_deepseek_v3.py`, and `model/tinyseek.py`. State that bias did not beat aux in the local budget and MTP was measured on a rejected MLA-style branch, so the local result is evidence-limited.

- [ ] **Step 4: Write the English mirror and verify**

  Preserve the branch-dependency warning and all paper-vs-local labels. Run link checks and commit with `docs: add DeepSeek-V3 paper tutorial`.

### Task 7: Write the DeepSeek-R1 article

**Files:**
- Modify: `papers/05-deepseek-r1/README_zh.md`
- Modify: `papers/05-deepseek-r1/README.md`
- Modify: `papers/assets/deepseek-r1/SOURCES.md`

**Interfaces:**
- Consumes: DeepSeek-R1 PDF notes/assets, `course/s07_cold_start_sft`, `course/s08_grpo_and_evaluation`, post-training docs, trainer code, and GPU completion reports.
- Produces: the final article and the end of the paper-first route.

- [ ] **Step 1: Explain the R1 training story**

  Cover R1-Zero versus the full R1 pipeline, cold-start data, rejection sampling or supervised stages, reasoning-oriented RL, GRPO-style optimization, reward design, evaluation and the paper's reported findings. Clearly separate paper-scale pipeline components from the teaching implementation.

- [ ] **Step 2: Embed paper evidence**

  Use selected paper figures/tables for the training pipeline, reasoning performance and ablations, with complete source metadata.

- [ ] **Step 3: Map TinySeek evidence**

  Link SFT/GRPO code, existing post-training report and reasoning figure. Explain that the local five-question evaluation shows format movement but no answer generalization, and that this is a teaching boundary rather than a contradiction of R1.

- [ ] **Step 4: Write the English mirror and verify**

  Preserve the distinction between optimized reward and task correctness. Run link checks and commit with `docs: add DeepSeek-R1 paper tutorial`.

### Task 8: Migrate navigation and mark legacy entry points

**Files:**
- Modify: `README_zh.md`
- Modify: `README.md`
- Modify: `course/README_zh.md`
- Modify: `course/README.md`
- Modify: `docs/zh/README.md`
- Modify: `docs/README.md`
- Modify: `experiments/README_zh.md`
- Modify: `experiments/README.md`

**Interfaces:**
- Consumes: completed `papers/` index and articles.
- Produces: one obvious root-to-paper reading path while preserving historical and evidence links.

- [ ] **Step 1: Update root READMEs**

  Replace the current primary “start here” language with `papers/README_zh.md` / `papers/README.md`. Retain quick-start commands and add a short “reference archives” section linking the old course, docs and experiment center.

- [ ] **Step 2: Add migration notices**

  At the top of each old index, add a concise notice that `papers/` is now the recommended paper-first route, followed by links to the relevant paper article(s). Keep the existing historical index below the notice.

- [ ] **Step 3: Remove competing claims**

  Change wording that calls `course/` the “唯一推荐入口” or presents the experiment center as a parallel learning route. Describe those directories as unit-based historical lessons, reference notes and evidence archives.

- [ ] **Step 4: Verify all navigation**

  Run a Markdown link checker over root, `papers/`, old indexes and all new image links. Run the existing smoke test command from the repository documentation without starting a GPU job. Commit with `docs: route repository navigation through papers`.

### Task 9: Final documentation audit

**Files:**
- Modify: any article or index file with broken links, inconsistent labels, missing provenance or duplicated claims found during audit.

**Interfaces:**
- Consumes: all completed paper articles, source ledgers, migration notices and existing reports.
- Produces: a clean, internally consistent paper-first tutorial tree ready for review.

- [ ] **Step 1: Audit evidence labels**

  Search every new article for `论文主张 | 论文证据 | TinySeek 证据 | 关系 | 可迁移边界` and verify each quantitative TinySeek claim links to an existing report/config/figure.

- [ ] **Step 2: Audit source provenance**

  Verify every paper figure/table has a matching `SOURCES.md` row with URL, source number, page/section and reading note; open each local image once.

- [ ] **Step 3: Audit scope boundaries**

  Search for language that implies TinySeek reproduced DeepSeek scale, that a local failure refutes a paper, or that combines incompatible branches. Rewrite those sentences with the approved evidence vocabulary.

- [ ] **Step 4: Run final verification**

  Run `git diff --check`, the repository's existing Python smoke tests, and the Markdown link checker. Expected: no whitespace errors, smoke tests pass, and all links resolve. Record any checker limitation in the final handoff rather than changing experiment code.

- [ ] **Step 5: Commit the audit**

  Commit with `docs: audit paper-first tutorial evidence`.

## Self-Review Checklist

- [ ] Every spec requirement maps to at least one task: five articles, paper-first navigation, paper figures/tables, evidence hierarchy, no new experiments, legacy preservation, bilingual parity and verification.
- [ ] No task requires a new model, trainer, dataset, configuration or GPU run.
- [ ] All article tasks use the same nine-section template and cite existing code/report paths.
- [ ] The plan contains no placeholder language or unspecified validation step.
- [ ] Task order is dependency-safe: index/scaffold → sources/assets → articles → navigation → audit.
