---
icon: lucide/cpu
---

# Models and SAE

## Qwen3.5 local target

The local target is `Qwen/Qwen3.5-2B-Base`, configured in:

```text
configs/models/qwen35_2b.yaml
```

The H200 target configs are also present:

- `configs/models/qwen35_9b.yaml`
- `configs/models/qwen35_27b.yaml`

## Qwen backend

The backend:

1. loads tokenizer and model through `transformers`;
2. formats LatentBehaviorBench messages into plain role-prefixed text;
3. runs teacher-forcing forward passes with hidden states;
4. extracts requested activation views;
5. supports residual forward hooks for steering.

## Qwen-Scope backend

The Qwen-Scope SAE loader expects the official checkpoint format:

```text
layer{n}.sae.pt
  W_enc: (d_sae, d_model)
  W_dec: (d_model, d_sae)
  b_enc: (d_sae,)
  b_dec: (d_model,)
```

For 2B:

- SAE repo: `Qwen/SAE-Res-Qwen3.5-2B-Base-W32K-L0_50`
- hidden size: `2048`
- SAE width: `32768`
- TopK: `50`
- layers: `0..23`

## Activation views

| View | Use |
| --- | --- |
| `last_prompt_token` | Prompt-state monitor |
| `first_assistant_token` | Initial answer state |
| `assistant_answer_mean` | Answer-level behavior representation |
| `last_answer_token` | Late output state |

For paired same-prompt contrasts, answer-dependent views are often more
informative than `last_prompt_token`, because the prompt is identical across the
positive and negative sides.

