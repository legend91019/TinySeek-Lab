# DeepSeek Phase 1 Tutorial Expansion Implementation Plan

**Goal:** Add three bilingual, paper-first DeepSeek tutorials and update repository navigation without changing model or experiment code.

**Architecture:** Preserve existing 01–05 directories. Add 06–08 with self-contained README files and source ledgers; update paper/root indexes to expose two learning lines.

**Tech Stack:** Markdown, existing PNG/SVG assets, PowerShell/Python for link checks.

## Global Constraints
- Paper claims and official figures are primary evidence.
- TinySeek evidence is supplementary and no new experiments are run.
- Scope is LLM/language-model papers only.
- Preserve existing links and old course/reference files.

### Task 1: Add V3.2-Exp bilingual tutorial and source ledger
- Create papers/06-deepseek-v3-2-sparse-attention/README_zh.md and README.md.
- Create papers/assets/deepseek-v3-2/SOURCES.md.
- Explain DSA, indexer/selection attention, long-context efficiency, V3.1-Terminus comparison, and TinySeek boundary.

### Task 2: Add V4 bilingual tutorial and source ledger
- Create papers/07-deepseek-v4/README_zh.md and README.md.
- Create papers/assets/deepseek-v4/SOURCES.md.
- Explain CSA/HCA, mHC, Muon, 1M context, reported efficiency figures, and scale limitations.

### Task 3: Add DeepSeekMath bilingual tutorial and source ledger
- Create papers/08-deepseek-math/README_zh.md and README.md.
- Create papers/assets/deepseek-math/SOURCES.md.
- Explain data mining, continued pretraining, supervised fine-tuning, GRPO, verifier rewards, and relation to R1.

### Task 4: Update indexes and navigation
- Modify papers/README_zh.md and papers/README.md with two routes and 06–08 links.
- Modify root README_zh.md and README.md to replace five-paper wording and update route diagrams.

### Task 5: Verify and commit
- Run Markdown link/path checks for all new and modified files.
- Run existing Python tests without changing experiment outputs.
- Review git diff --check, then commit the documentation changes.
