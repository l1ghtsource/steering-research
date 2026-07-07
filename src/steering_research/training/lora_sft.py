from __future__ import annotations

from pathlib import Path
from typing import Any

from steering_research.data import BenchmarkStore
from steering_research.reports.dashboard import write_static_dashboard
from steering_research.reports.markdown import write_experiment_report
from steering_research.runtime import RunLogger, load_yaml, resolve_path


def _format_sft_text(prompt: str, answer: str) -> str:
    return f"{prompt}\nassistant: {answer}"


def _good_answer_rows(
    store: BenchmarkStore, behavior: str, origin: str, limit_pairs: int
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for pair in store.pairs(behavior=behavior, origin_bucket=origin, limit=limit_pairs):
        answer = pair.negative.assistant_text
        if answer.strip():
            rows.append({"text": _format_sft_text(pair.negative.prompt_text, answer)})
    return rows


def run_lora_sft(repo_root: Path, config_path: Path) -> Path:
    try:
        from datasets import Dataset
        from peft import LoraConfig, get_peft_model
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            DataCollatorForLanguageModeling,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        msg = "Install training dependencies with `uv sync --extra model --extra training`."
        raise RuntimeError(msg) from exc

    cfg = load_yaml(config_path)
    model_cfg = load_yaml(resolve_path(str(cfg["model"]), repo_root))
    data_cfg = load_yaml(resolve_path(str(cfg["dataset"]), repo_root))
    logger = RunLogger(
        resolve_path(str(cfg.get("run_output_dir", "runs")), repo_root),
        str(cfg["name"]),
        {"config": cfg, "model": model_cfg},
    )
    store = BenchmarkStore(
        root=resolve_path(str(data_cfg["root"]), repo_root),
        examples_path=str(data_cfg.get("examples", "processed/examples.jsonl")),
        contrasts_path=str(data_cfg.get("contrasts", "processed/contrasts.jsonl")),
        clean_splits_path=str(data_cfg.get("clean_splits", "processed/eval_splits_clean.json")),
    )
    rows = _good_answer_rows(
        store,
        behavior=str(cfg["behavior"]),
        origin=str(cfg["origin"]),
        limit_pairs=int(cfg["limit_pairs"]),
    )
    if not rows:
        msg = "No supervised rows were built from benchmark contrasts."
        raise ValueError(msg)
    logger.log_metric(
        {
            "stage": "dataset_built",
            "rows": len(rows),
            "behavior": cfg["behavior"],
            "origin": cfg["origin"],
        }
    )

    tokenizer: Any = AutoTokenizer.from_pretrained(
        str(model_cfg["model_id"]), trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model: Any = AutoModelForCausalLM.from_pretrained(
        str(model_cfg["model_id"]), trust_remote_code=True
    )
    if hasattr(model, "config"):
        model.config.use_cache = False
    peft_config = LoraConfig(
        r=int(cfg["lora_rank"]),
        lora_alpha=int(cfg["lora_alpha"]),
        lora_dropout=float(cfg["lora_dropout"]),
        target_modules=[str(x) for x in cfg["target_modules"]],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)

    dataset = Dataset.from_list(rows)

    def tokenize(batch: dict[str, list[str]]) -> dict[str, Any]:
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=int(cfg["max_length"]),
            padding=False,
        )

    tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])
    output_dir = resolve_path(str(cfg["output_dir"]), repo_root)
    max_steps = int(cfg.get("max_steps", -1))
    args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=float(cfg["num_train_epochs"]),
        max_steps=max_steps,
        learning_rate=float(cfg["learning_rate"]),
        per_device_train_batch_size=int(cfg["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(cfg["gradient_accumulation_steps"]),
        logging_steps=1,
        save_strategy="epoch",
        report_to=[],
    )
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    trainer = Trainer(model=model, args=args, train_dataset=tokenized, data_collator=collator)
    train_result = trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    train_metrics = {key: float(value) for key, value in train_result.metrics.items()}
    logger.log_metric({"stage": "train_finished", **train_metrics})
    summary = {
        "experiment": cfg["name"],
        "backend": str(model_cfg["name"]),
        "training_method": "lora_sft",
        "rows": len(rows),
        "adapter_output_dir": str(output_dir),
        "max_steps": max_steps,
        **train_metrics,
    }
    logger.write_json("summary.json", summary)
    write_experiment_report(
        logger.run_dir / "report.md",
        "E006 LoRA SFT",
        summary,
        [
            {"stage": "dataset_built", "rows": len(rows)},
            {"stage": "train_finished", **train_metrics},
        ],
    )
    write_static_dashboard(logger.run_dir.parent)
    return logger.run_dir
