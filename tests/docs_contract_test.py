from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CHAPTERS = [
    "20_architecture_evolution_overview.md",
    "21_from_dense_to_deepseek_moe.md",
    "22_from_moe_to_deepseek_v2.md",
    "23_from_v2_to_deepseek_v3.md",
]
MATH_CHAPTER = "24_math_to_pytorch.md"
DEEP_DIVE_REQUIREMENTS = {
    "12_code_first_dense_lm.md": (
        "operatorname{RMS}",
        "torch.outer",
        "scaled_dot_product_attention",
        "F.silu",
        "F.cross_entropy",
    ),
    "21_from_dense_to_deepseek_moe.md": (
        "F.softmax(self.router",
        "torch.topk",
        "nonzero(as_tuple=True)",
        "F.one_hot",
        "L_{seq}",
    ),
    "22_from_moe_to_deepseek_v2.md": (
        "self.kv_down",
        "torch.split",
        "torch.cat",
        "kv_cache_elements_per_token",
        "C_{MLA/token}",
    ),
    "23_from_v2_to_deepseek_v3.md": (
        "torch.sigmoid",
        "gather(1",
        "torch.bincount",
        "torch.no_grad",
        "torch.stack",
    ),
    "16_training_loop_from_config_to_checkpoint.md": (
        "grad_accum_steps",
        "GradScaler",
        "unscale_",
        "clip_grad_norm_",
        "accum_count = 0",
    ),
    "19_posttraining_code_walkthrough.md": (
        "ignore_index=-100",
        "F.log_softmax",
        "gather(-1",
        "std(unbiased=False)",
        "adv.detach()",
    ),
}
EXACT_CODE_FRAGMENTS = {
    "21_from_dense_to_deepseek_moe.md": (
        ROOT / "model" / "stages" / "stage1_deepseek_moe.py",
        (
            "router_probs=F.softmax(self.router(flat),dim=-1)",
            "torch.topk(router_probs,k=self.config.top_k,dim=-1)",
        ),
    ),
    "22_from_moe_to_deepseek_v2.md": (
        ROOT / "model" / "stages" / "stage2_deepseek_v2.py",
        (
            "self.k_content_up(compressed_kv).view(",
            "repeat_kv(k_content,self.kv_repeats)",
            "self.k_rope_proj(x).view(",
        ),
    ),
    "23_from_v2_to_deepseek_v3.md": (
        ROOT / "model" / "stages" / "stage3_deepseek_v3.py",
        (
            "future_embed=self.embed(input_ids[:,depth+1:])",
            "mtp_logits=self.lm_head(self.norm(previous_hidden[:,:-1]))",
            "mtp_targets=labels[:,depth+2:]",
        ),
    ),
}


def test_bilingual_architecture_chapters() -> None:
    for name in CHAPTERS:
        en_path = ROOT / "docs" / name
        zh_path = ROOT / "docs" / "zh" / name
        assert en_path.exists(), f"Missing English chapter: {name}"
        assert zh_path.exists(), f"Missing Chinese chapter: {name}"
        for path in (en_path, zh_path):
            text = path.read_text(encoding="utf-8")
            assert "model/stages" in text
            assert "configs/architecture_lab" in text
            assert "<!-- tinyseek-nav -->" in text

    zh_overview = (ROOT / "docs" / "zh" / CHAPTERS[0]).read_text(encoding="utf-8")
    en_overview = (ROOT / "docs" / CHAPTERS[0]).read_text(encoding="utf-8")
    for phrase in ("可测量瓶颈", "研究假设", "决策门槛", "证据状态"):
        assert phrase in zh_overview
    for phrase in ("Measurable bottleneck", "Research hypothesis", "Decision gate", "Evidence status"):
        assert phrase in en_overview


def test_old_chapters_are_current() -> None:
    zh_sft = (ROOT / "docs" / "zh" / "07_stage5_sft_cold_start.md").read_text(encoding="utf-8")
    en_sft = (ROOT / "docs" / "07_stage5_sft_cold_start.md").read_text(encoding="utf-8")
    assert "还是占位" not in zh_sft
    assert "placeholder" not in en_sft.lower()
    assert "trainer/train_sft.py" in zh_sft
    assert "trainer/train_sft.py" in en_sft


def test_math_to_pytorch_walkthrough_is_bilingual() -> None:
    en_path = ROOT / "docs" / MATH_CHAPTER
    zh_path = ROOT / "docs" / "zh" / MATH_CHAPTER
    assert en_path.exists()
    assert zh_path.exists()

    en = en_path.read_text(encoding="utf-8")
    zh = zh_path.read_text(encoding="utf-8")
    for phrase in (
        "formula -> symbols -> tensor shapes",
        "nn.Parameter",
        "scaled_dot_product_attention",
        "F.cross_entropy",
        "<!-- tinyseek-nav -->",
    ):
        assert phrase in en
    for phrase in (
        "公式 -> 符号含义 -> 张量 shape",
        "nn.Parameter",
        "scaled_dot_product_attention",
        "F.cross_entropy",
        "<!-- tinyseek-nav -->",
    ):
        assert phrase in zh

    dense_en = (ROOT / "docs" / "12_code_first_dense_lm.md").read_text(encoding="utf-8")
    dense_zh = (ROOT / "docs" / "zh" / "12_code_first_dense_lm.md").read_text(encoding="utf-8")
    assert "operatorname{RMS}" in dense_en
    assert "operatorname{RMS}" in dense_zh
    assert "torch.rsqrt" in dense_en
    assert "torch.rsqrt" in dense_zh


def test_deep_dives_map_math_shapes_and_real_apis() -> None:
    for name, requirements in DEEP_DIVE_REQUIREMENTS.items():
        for root in (ROOT / "docs", ROOT / "docs" / "zh"):
            text = (root / name).read_text(encoding="utf-8")
            assert "$$" in text, f"Missing formula in {root / name}"
            for phrase in requirements:
                assert phrase in text, f"Missing {phrase!r} in {root / name}"


def test_bilingual_snippets_match_stage_source() -> None:
    compact = lambda text: "".join(text.split())
    for name, (source_path, fragments) in EXACT_CODE_FRAGMENTS.items():
        source = compact(source_path.read_text(encoding="utf-8"))
        en = compact((ROOT / "docs" / name).read_text(encoding="utf-8"))
        zh = compact((ROOT / "docs" / "zh" / name).read_text(encoding="utf-8"))
        for fragment in fragments:
            assert fragment in source, f"Invalid source contract: {fragment}"
            assert fragment in en, f"English snippet missing exact source fragment: {fragment}"
            assert fragment in zh, f"Chinese snippet missing exact source fragment: {fragment}"


def test_readme_and_indexes_expose_the_course() -> None:
    for path in (ROOT / "README.md", ROOT / "README_zh.md"):
        text = path.read_text(encoding="utf-8")
        assert "stage0_deepseek_llm.py" in text
        assert "stage3_deepseek_v3.py" in text
        assert "architecture_ppl.svg" in text
        assert "06_architecture_evolution_plan" in text
        assert "24_math_to_pytorch.md" in text
        assert "19_posttraining_code_walkthrough.md" in text
    zh_readme = (ROOT / "README_zh.md").read_text(encoding="utf-8")
    en_readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "本仓库是笔者学习 DeepSeek 论文时，为方便理解而完成的作品" in zh_readme
    assert "created while studying the DeepSeek papers" in en_readme
    for path in (ROOT / "docs" / "README.md", ROOT / "docs" / "zh" / "README.md"):
        text = path.read_text(encoding="utf-8")
        assert MATH_CHAPTER in text
        positions = [text.index(name) for name in CHAPTERS]
        assert positions == sorted(positions)
    nav_source = (ROOT / "scripts" / "refresh_doc_nav.py").read_text(encoding="utf-8")
    for name in CHAPTERS:
        assert nav_source.count(name) == 2
    assert nav_source.count(MATH_CHAPTER) == 2
    for path in (ROOT / "experiments" / "README.md", ROOT / "experiments" / "README_zh.md"):
        assert "06_architecture_evolution_plan" in path.read_text(encoding="utf-8")


if __name__ == "__main__":
    test_bilingual_architecture_chapters()
    test_old_chapters_are_current()
    test_math_to_pytorch_walkthrough_is_bilingual()
    test_deep_dives_map_math_shapes_and_real_apis()
    test_bilingual_snippets_match_stage_source()
    test_readme_and_indexes_expose_the_course()
    print("docs contract ok")
