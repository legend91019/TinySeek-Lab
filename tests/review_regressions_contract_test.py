from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_review_regression_contracts() -> None:
    model_source = (ROOT / "model" / "tinyseek.py").read_text(encoding="utf-8")
    trainer_source = (ROOT / "trainer" / "train_pretrain.py").read_text(encoding="utf-8")
    eval_source = (ROOT / "eval" / "mini_eval.py").read_text(encoding="utf-8")

    assert "mla_decoupled_rope" in model_source
    assert "qk_rope_head_dim" in model_source
    assert "mla_latent_size + self.rope_head_dim" in model_source
    assert 'register_buffer("expert_bias", torch.zeros(config.num_experts))' in model_source
    assert "def _load_from_state_dict" in model_source
    assert (ROOT / "model" / "routing.py").exists()
    assert '"val_lm_loss"' in trainer_source
    assert '"val_objective"' in trainer_source
    assert '"training_tokens_per_second"' in trainer_source
    ppl_section = eval_source.split("def eval_ppl", 1)[1].split("def generate_text", 1)[0]
    assert 'out["lm_loss"]' in ppl_section
    assert 'out["aux_loss"]' not in ppl_section


if __name__ == "__main__":
    test_review_regression_contracts()
    print("review regression contracts ok")
