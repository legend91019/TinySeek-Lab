# 11. Notes on the MiniMind-Inspired Structure

TinySeek-Lab borrows the clean high-level layout from MiniMind:

- `model/` keeps model architecture code.
- `dataset/` keeps tokenizer and dataset wrappers.
- `trainer/` keeps one entry script per training stage.
- `scripts/` keeps helper utilities such as data preparation and generation.
- `docs/` keeps the tutorial chapters.

TinySeek-Lab differs in emphasis:

- It is a research tutorial first, model release second.
- It keeps multimodal/tool/agent paths out of v0.x.
- It makes DeepSeek paper anchors explicit in every chapter.
- It treats sweeps and ablations as first-class tutorial artifacts.

The structure should remain simple enough that a reader can open one file and
understand the corresponding stage.
