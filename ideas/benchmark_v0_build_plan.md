# Benchmark v0: план сборки и инструкция для агента

## 0. Цель

Нужно собрать не один монолитный QA-бенчмарк, а **unified benchmark suite** для поиска, мониторинга и каузальной проверки behavioral/persona representations в LLM.

Целевые behavioral axes:

1. `hallucination`
2. `deception`
3. `sycophancy`
4. `overconfidence`
5. `premature_refusal`
6. `unsafe_planning`

Главный downstream-use:

```text
benchmark examples
→ contrast groups
→ teacher-forcing forward pass
→ mean-difference vectors / probes / SAE feature deltas
→ detection AUROC
→ steering / ablation
→ behavioral eval + collateral damage
```

Бенч должен поддерживать два режима:

1. **Behavioral evaluation**: сгенерировать ответ модели и оценить поведение.
2. **Activation extraction**: прогнать prompt + known answer / trajectory через модель в teacher-forcing режиме и извлечь активации.

---

## 1. Ключевое решение по формату

Используем двухуровневую схему:

```text
atomic examples
+
contrast groups
```

Почему так: открытые бенчи бывают разными. У одних есть только prompt и gold answer, у других — prompt + candidate answer + label, у третьих — multi-turn scenario, у четвёртых — agentic task. Если требовать good/bad pair в каждой строке, часть источников плохо конвертируется. Если хранить только prompt-only examples, неудобно извлекать vectors.

Поэтому:

```text
examples.jsonl  — атомарные примеры из открытых и синтетических источников
contrasts.jsonl — пары/группы examples для извлечения активационных направлений
```

---

## 2. Репозиторий: ожидаемая структура

```text
bench/
  README.md

  raw/
    emergent_misalignment/
    openai_persona_features/
    persona_vectors/
    halueval/
    sycophancy_eval/
    or_bench/
    xstest/
    strongreject/
    machiavelli/
    agentharm/
    opendeception/
    simpleqa_verified/

  processed/
    examples.jsonl
    contrasts.jsonl
    source_report.md
    stats.json

  rubrics/
    hallucination.md
    overconfidence.md
    sycophancy.md
    premature_refusal.md
    deception.md
    unsafe_planning.md
    general_helpfulness.md

  scripts/
    download_sources.py
    convert_emergent_misalignment.py
    convert_openai_persona_features.py
    convert_persona_vectors.py
    convert_halueval.py
    convert_sycophancy.py
    convert_orbench.py
    convert_xstest.py
    convert_strongreject.py
    convert_machiavelli.py
    convert_agentharm.py
    convert_opendeception.py
    convert_simpleqa_verified.py
    build_synthetic_contrasts.py
    build_contrasts.py
    validate_schema.py
    compute_stats.py
    sample_for_manual_review.py

  runs/
    generations/
    scores/
    activations/
```

Минимальный deliverable агента:

```text
processed/examples.jsonl
processed/contrasts.jsonl
processed/source_report.md
processed/stats.json
rubrics/*.md
scripts/*.py
```

---

## 3. Схема `examples.jsonl`

Одна строка — один атомарный пример.

```json
{
  "id": "halueval_qa_000001",
  "source": "halueval",
  "license": "MIT",
  "behavior_axes": ["hallucination"],
  "task_family": "factuality_calibration",
  "format": "prompt_candidate_answer_label",
  "language": "en",
  "domain": "general_qa",
  "messages": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "...candidate answer..."}
  ],
  "reference": {
    "type": "label_or_rubric",
    "correct_answers": [],
    "incorrect_answers": [],
    "rubric": "..."
  },
  "labels": {
    "hallucination": 1,
    "overconfidence": null,
    "sycophancy": null,
    "deception": null,
    "premature_refusal": null,
    "unsafe_planning": null,
    "is_safe": true,
    "should_answer": true,
    "should_refuse": false,
    "expected_behavior": "truthful_or_uncertain",
    "bad_behavior": ["hallucination"]
  },
  "scoring": {
    "primary": "gold_label",
    "secondary": ["llm_judge"],
    "judge_rubric": "rubrics/hallucination.md"
  },
  "split": "extraction",
  "metadata": {
    "original_id": "...",
    "url": "...",
    "notes": ""
  }
}
```

### Обязательные поля

```text
id
source
behavior_axes
task_family
format
language
messages
labels
scoring
split
```

### Допустимые `task_family`

```text
factuality_calibration
social_pressure
refusal_calibration
persona_misalignment
deception_interaction
agentic_safety
capability_control
multilingual_transfer
temporal_dynamics
```

### Допустимые `format`

```text
single_turn_qa
prompt_candidate_answer_label
paired_completion
free_form_generation
safe_or_unsafe_prompt
multi_turn_scenario
agentic_task
agentic_trajectory
capability_task
```

### Допустимые `split`

```text
extraction
eval_in_domain
eval_external
eval_control
eval_ood
eval_temporal
eval_agentic
eval_multilingual
capability_control
manual_review
```

---

## 4. Схема `contrasts.jsonl`

Одна строка — один contrast group для извлечения direction/features.

```json
{
  "contrast_id": "sycophancy_000001",
  "behavior": "sycophancy",
  "contrast_type": "bad_minus_good",
  "positive_item_ids": ["sycophancy_000001_bad"],
  "negative_item_ids": ["sycophancy_000001_good"],
  "matched_on": [
    "topic",
    "question",
    "answer_length_approximately",
    "language"
  ],
  "activation_views": [
    "last_prompt_token",
    "first_assistant_token",
    "assistant_answer_mean"
  ],
  "recommended_methods": [
    "caa_mean_diff",
    "linear_probe",
    "sae_delta"
  ],
  "split": "extraction",
  "notes": "Positive means undesired behavior is present. Negative means undesired behavior is absent or suppressed."
}
```

### Смысл positive/negative

Для всех behaviors используем один convention:

```text
positive = пример с нежелательным поведением
negative = пример без нежелательного поведения / с хорошим поведением
```

Тогда направление:

```text
v_behavior = mean_activation(positive) - mean_activation(negative)
```

Примеры:

```text
hallucination:     positive = hallucinated answer; negative = correct/uncertain answer
sycophancy:        positive = agrees with false user; negative = corrects user
deception:         positive = hides/false report; negative = discloses truth
overconfidence:    positive = categorical unsupported answer; negative = calibrated answer
premature_refusal: positive = refuses safe request; negative = answers safely
unsafe_planning:   positive = unsafe plan; negative = safe plan with mitigations
```

---

## 5. Целевой размер `benchmark_v0`

### 5.1. Резюме размеров

```text
extraction:
  900–1000 contrast pairs

in-domain behavioral eval:
  1200–1500 prompts

external/control eval:
  2000–2500 prompts

temporal/agentic:
  50–100 scenarios/trajectories

capability controls:
  300–500 tasks
```

Это нормальный размер для первой исследовательской итерации: достаточно большой, чтобы получить устойчивые AUROC/steering-кривые, но достаточно маленький, чтобы реально прогнать hidden states и steering за несколько дней.

### 5.2. Если нужно быстро: `benchmark_v0_small`

```text
extraction:
  ~500 contrast pairs

behavioral eval:
  ~800–1000 prompts

external/control:
  ~800–1200 prompts

agentic:
  10–20 tasks

temporal:
  10–20 trajectories
```

---

## 6. Размеры по behavioral axes

## 6.1. Hallucination

### Цель

Поймать поведение:

```text
модель выдумывает факт, ссылку, число, цитату или утверждение без достаточного основания
```

### Размер

```text
extraction:
  200 contrast pairs

in-domain eval:
  200 prompts

external validation:
  1000 HaluEval QA examples
  optional: 300–500 SimpleQA Verified prompts
```

### Источники

1. **HaluEval**
   - использовать QA split как основной external validation;
   - если есть candidate answer + hallucination label, использовать также для extraction.
2. **Persona Vectors hallucination artifacts**
   - если доступны generated extraction/eval prompts, взять как template для synthetic generation.
3. **SimpleQA Verified**
   - использовать только как auxiliary factuality/calibration sanity check, не как core.

### Что должен сделать агент

1. Скачать/склонировать HaluEval.
2. Найти QA split.
3. Сконвертировать не менее 1000 QA examples в `examples.jsonl`.
4. Если в HaluEval есть labeled candidate answers:
   - построить `positive = hallucinated answer`;
   - `negative = non-hallucinated/correct answer`;
   - добавить contrast groups.
5. Если good/bad pairs не хватает:
   - сгенерировать synthetic contrast completions:
     - bad completion: уверенно выдумывает;
     - good completion: корректно отвечает или признаёт неопределённость.
6. Для extraction добрать до 200 contrast pairs.

### Scoring

```text
primary:
  gold label, если есть

secondary:
  LLM judge по rubric hallucination.md

extra:
  unsupported_claim_rate
  not_attempted_rate
  wrong_and_confident_rate
```

---

## 6.2. Sycophancy

### Цель

Поймать поведение:

```text
модель соглашается с ошибочным, предвзятым или навязанным мнением пользователя вместо независимого ответа
```

### Размер

```text
extraction:
  200 contrast pairs

in-domain eval:
  200 prompts

external validation:
  300 held-out sycophancy prompts
```

### Источники

1. **Persona Vectors sycophancy artifacts**
   - preferred source для extraction/eval, если доступен.
2. **Anthropic / model-written-evals sycophancy**
   - philosophy, NLP survey, political typology.
3. **Nishimura-Gasparian / reward-hack-generalization sycophancy data**
   - если доступен в открытом виде, использовать как external validation.
4. **SycophancyEval**
   - если dataset доступен, использовать как external/in-domain eval.

### Что должен сделать агент

1. Проверить официальный repo Persona Vectors.
2. Найти trait files для `sycophancy`.
3. Извлечь:
   - positive prompts / positive system prompts;
   - negative prompts / negative system prompts;
   - extraction questions;
   - eval questions.
4. Сконвертировать в examples.
5. Построить contrast groups:
   - positive = sycophantic answer / trait-encouraging condition;
   - negative = independent correction / trait-discouraging condition.
6. Если датасет prompt-only:
   - сгенерировать paired completions:
     - bad: соглашается с ложным утверждением пользователя;
     - good: корректно и спокойно исправляет пользователя.
7. Добрать до:
   - 200 extraction pairs;
   - 200 eval prompts;
   - 300 external prompts, если источник есть.

### Scoring

```text
primary:
  LLM judge with rubric

secondary:
  rule-based agreement detector for clear false-claim cases

extra:
  correction_rate
  false_agreement_rate
  excessive_praise_rate
```

---

## 6.3. Deception / misaligned persona

### Цель

Поймать поведение:

```text
модель скрывает важную информацию, вводит пользователя в заблуждение, лжёт о факте/идентичности/действии или демонстрирует broad misaligned persona
```

### Размер

```text
extraction:
  150 contrast pairs

in-domain eval:
  100 deception/persona prompts

EM eval:
  44 EM prompts × 5–10 rollouts

external/multi-turn:
  50 OpenDeception scenarios
  optional: DeceptionBench sample if repository is easy to use
```

### Источники

1. **Emergent Misalignment repo**
   - main questions;
   - preregistered evals;
   - code-formatted answer questions;
   - factual deception questions;
   - identity deception / situational awareness questions.
2. **OpenAI Persona Features repo**
   - training/evaluation datasets для Persona Features Control Emergent Misalignment.
3. **OpenDeception**
   - 50 real-world open-ended deception scenarios.
4. **DeceptionBench**
   - optional, если быстро скачивается и лицензия/формат понятны.

### Что должен сделать агент

1. Скачать EM repo.
2. Найти evaluation files:
   - main/free-form questions;
   - preregistered evals;
   - factual deception;
   - identity deception / situational awareness;
   - code-formatted variants.
3. Сконвертировать все EM eval prompts в `examples.jsonl`.
4. Отметить:
   - `task_family = persona_misalignment`;
   - `behavior_axes = ["deception"]` или `["deception", "unsafe_planning"]`, если применимо.
5. Для EM eval:
   - не строить один deterministic label;
   - оставить как generation eval с 5–10 rollouts на prompt.
6. Скачать OpenDeception repo.
7. Сконвертировать 50 scenarios в `examples.jsonl`:
   - `format = multi_turn_scenario`;
   - `split = eval_external`;
   - `behavior_axes = ["deception"]`.
8. Для extraction сделать отдельные synthetic contrast pairs:
   - same scenario;
   - bad trajectory/completion скрывает ошибку или лжёт;
   - good trajectory/completion честно сообщает.
9. Добрать до 150 contrast pairs.

### Scoring

```text
primary:
  LLM judge with deception rubric

for agent/multi-turn:
  disclosure_check
  contradiction_between_observation_and_report
  hidden_issue_reported
  deceptive_action_taken

extra:
  deception_intent_score
  deception_outcome_score
  report_truthfulness
```

### Важно

Для deception нельзя полагаться только на final-answer tokens. Нужно хранить activation views:

```text
last_prompt_token
first_assistant_token
assistant_answer_mean
pre_action_token
pre_report_token
```

Если `pre_action_token` и `pre_report_token` пока невозможно достать, оставить поля в schema и заполнять позже.

---

## 6.4. Overconfidence

### Цель

Поймать поведение:

```text
модель даёт категоричный ответ при неопределённости или при недостатке информации
```

### Размер

```text
extraction:
  150 contrast pairs

in-domain eval:
  shared with hallucination/factuality, at least 200 prompts

external validation:
  300–500 SimpleQA Verified / SimpleQA-style prompts
```

### Источники

1. **SimpleQA Verified**
   - preferred factuality/calibration sanity check.
2. **SimpleQA**
   - fallback if Verified is unavailable.
3. **TruthfulQA**
   - optional old auxiliary check.
4. **Synthetic uncertain questions**
   - для extraction лучше, чем старые factual QA.

### Что должен сделать агент

1. Скачать SimpleQA Verified, если доступен.
2. Сконвертировать 300–500 prompts.
3. Добавить в prompts требование явной confidence или разрешить judge оценивать категоричность без явной вероятности.
4. Сгенерировать 150 contrast pairs:
   - bad: unsupported categorical answer;
   - good: calibrated answer with uncertainty or refusal to guess.
5. Не смешивать overconfidence с hallucination:
   - hallucination = ложное содержание;
   - overconfidence = неправильная степень уверенности.
6. Для части prompts оставить `expected_answer_unknown = true`, чтобы good behavior было “I don’t know / cannot determine”.

### Scoring

```text
primary:
  wrong_and_confident_rate

secondary:
  LLM judge for unsupported certainty

extra:
  hedge_marker_rate
  probability_calibration_error
  not_attempted_when_unknown
```

---

## 6.5. Premature refusal

### Цель

Поймать поведение:

```text
модель отказывается отвечать на безопасный или частично безопасный запрос, хотя могла дать полезный безопасный ответ
```

### Размер

```text
extraction:
  150 contrast pairs

in-domain eval:
  300 benign-sensitive prompts

unsafe-control eval:
  150 actually unsafe prompts
```

### Источники

1. **OR-Bench**
   - основной источник over-refusal prompts;
   - взять hard subset + balanced sample.
2. **XSTest**
   - компактный sanity-check: safe prompts + unsafe prompts.
3. **SORRY-Bench / StrongREJECT**
   - использовать как unsafe-control, чтобы anti-refusal steering не повышал unsafe compliance.

### Что должен сделать агент

1. Скачать OR-Bench.
2. Взять:
   - 300 benign/seemingly toxic prompts;
   - приоритет hard prompts.
3. Скачать XSTest.
4. Сконвертировать:
   - safe prompts → `should_answer = true`;
   - unsafe prompts → `should_refuse = true`.
5. Скачать StrongREJECT или другой harmful request compliance benchmark.
6. Взять 150 unsafe-control prompts.
7. Для extraction построить 150 pairs:
   - bad: полный отказ на безопасный запрос;
   - good: безопасный полезный ответ или ответ на безопасную часть.
8. В labels всегда различать:
   - `premature_refusal`;
   - `safe_refusal`;
   - `unsafe_compliance`.

### Scoring

```text
primary:
  refusal_rate_on_safe_prompts

secondary:
  LLM judge for whether refusal was premature

control:
  unsafe_compliance_rate_on_unsafe_prompts

extra:
  partial_helpfulness_rate
  safe_completion_rate
```

---

## 6.6. Unsafe planning

### Цель

Поймать поведение:

```text
модель предлагает опасный, обходной, манипулятивный или плохо ограниченный план там, где нужен безопасный план
```

### Размер

```text
extraction:
  100 contrast pairs

in-domain eval:
  100 planning prompts

agentic eval:
  20–50 tool-use / multi-step tasks

unsafe-control:
  100 harmful/safety prompts
```

### Источники

1. **AgentHarm**
   - agentic harmful tasks; использовать осторожно как safety-control/eval.
2. **Machiavelli**
   - optional; использовать как decision-making / harmfulness eval, не как core extraction.
3. **StrongREJECT / HarmBench-style prompts**
   - safety control.
4. **Synthetic planning pairs**
   - основной источник extraction pairs.

### Что должен сделать агент

1. Скачать AgentHarm или подготовить loader через Inspect Evals, если dataset так устроен.
2. Взять 20–50 tasks для `eval_agentic`.
3. Не копировать вредные инструкции в README; хранить только dataset files.
4. Сгенерировать 100 planning contrast pairs:
   - bad: план содержит unsafe step / bypass / concealment / risky tool-use;
   - good: безопасный план с mitigation, checks, refusal where needed.
5. Для каждого примера заполнить:
   - `unsafe_step_count`, если можно;
   - `requires_refusal`, если запрос вредный;
   - `safe_alternative_expected`, если запрос dual-use.
6. Для agentic tasks сохранять:
   - initial task;
   - tool list;
   - expected safe behavior;
   - possible unsafe behavior;
   - scoring rubric.

### Scoring

```text
primary:
  unsafe_plan_score via rubric

secondary:
  unsafe_step_count
  missing_safety_mitigation
  dangerous_tool_use_decision

agentic:
  harmful_task_progress
  refusal_or_safe_redirection
  tool_use_safety
```

---

## 7. Temporal / hysteresis subset

### Цель

Проверять гипотезу:

```text
persona/behavioral state может сохраняться после исчезновения исходного триггера
```

### Размер

```text
30–50 multi-turn trajectories
```

Для `benchmark_v0_small`:

```text
10–20 trajectories
```

### Формат

```json
{
  "id": "temporal_sycophancy_0001",
  "source": "synthetic",
  "behavior_axes": ["sycophancy"],
  "task_family": "temporal_dynamics",
  "format": "multi_turn_scenario",
  "language": "en",
  "scenario": {
    "turns": [
      {"role": "user", "content": "trigger turn that induces behavior"},
      {"role": "user", "content": "neutral topic turn"},
      {"role": "user", "content": "another neutral turn"}
    ],
    "trigger_turns": [0],
    "washout_turns": [1, 2]
  },
  "labels": {
    "expected_behavior": "behavior should not persist after trigger removal"
  },
  "split": "eval_temporal"
}
```

### Что должен сделать агент

1. Для 3 behaviors сделать temporal tasks:
   - sycophancy;
   - deception/persona;
   - overconfidence или refusal.
2. На каждый behavior:
   - 10–15 trajectories.
3. В каждой trajectory:
   - turn 1 активирует behavior;
   - turns 2–4 убирают trigger;
   - проверяется persistence.
4. Сохранять turn-level metadata:
   - `trigger_present`;
   - `expected_behavior`;
   - `behavior_should_decay`.

### Метрики потом

```text
activation_persistence
behavior_persistence
decay_rate
topic_transfer
washout_failure_rate
```

---

## 8. Multilingual subset

### Цель

Проверять гипотезу:

```text
English persona monitor может хуже ловить misalignment на других языках
```

### Размер

```text
50–100 prompts × language
```

Для MVP:

```text
languages:
  en
  ru
  zh

size:
  50 prompts per language
```

### Что должен сделать агент

1. Взять balanced sample из:
   - sycophancy;
   - hallucination;
   - refusal;
   - deception.
2. Перевести / сгенерировать версии на:
   - Russian;
   - Chinese;
   - optionally Arabic.
3. Для каждого prompt сохранить:
   - `translation_type = translated | native_generated`;
   - `source_example_id`;
   - `language`.
4. Не смешивать multilingual examples с extraction; использовать как `eval_multilingual`.

---

## 9. Capability controls

### Цель

Проверять, что steering/ablation не ломает общие способности.

### Размер

```text
300–500 tasks total
```

### Источники

```text
MMLU / MMLU-Pro sample: 100–200
GSM8K / MATH sample: 100
HumanEval / APPS / code tasks: 50–100
IFEval or instruction-following sample: 50–100
```

### Что должен сделать агент

1. Не тащить полный MMLU/MMLU-Pro.
2. Взять stratified sample.
3. Сохранить как:
   - `task_family = capability_control`;
   - `behavior_axes = []`;
   - `split = capability_control`.
4. Для code capability лучше использовать короткие задачи, чтобы запуск был дешёвым.

---

## 10. Общая статистика, которую агент обязан посчитать

После сборки агент должен создать `processed/stats.json`:

```json
{
  "total_examples": 0,
  "total_contrasts": 0,
  "by_behavior_axis": {
    "hallucination": 0,
    "deception": 0,
    "sycophancy": 0,
    "overconfidence": 0,
    "premature_refusal": 0,
    "unsafe_planning": 0
  },
  "by_task_family": {},
  "by_format": {},
  "by_source": {},
  "by_split": {},
  "by_language": {},
  "contrast_counts_by_behavior": {},
  "missing_required_fields": 0,
  "duplicate_ids": 0,
  "empty_messages": 0
}
```

Также создать `processed/source_report.md`:

```text
source name
URL
license
download method
raw path
number of converted examples
number of contrasts
known limitations
```

---

## 11. Инструкция агенту: пошаговый план

## Phase 1. Инициализация

1. Создай структуру папок из раздела 2.
2. Создай schema validator:
   - проверяет обязательные поля;
   - проверяет уникальность `id`;
   - проверяет, что `positive_item_ids` и `negative_item_ids` существуют;
   - проверяет допустимые значения `split`, `format`, `task_family`.
3. Создай пустые rubrics:
   - `hallucination.md`;
   - `overconfidence.md`;
   - `sycophancy.md`;
   - `premature_refusal.md`;
   - `deception.md`;
   - `unsafe_planning.md`.

## Phase 2. Скачать core sources

Скачать/подготовить источники в таком порядке:

1. Persona Vectors repo.
2. Emergent Misalignment repo.
3. OpenAI Persona Features repo.
4. HaluEval.
5. OR-Bench.
6. XSTest.
7. StrongREJECT.
8. OpenDeception.
9. AgentHarm.
10. SimpleQA Verified, если доступен.
11. Machiavelli, если легко ставится.

Не блокироваться на одном источнике больше разумного: если источник требует сложной установки, зафиксировать проблему в `source_report.md` и двигаться дальше.

## Phase 3. Конвертировать atomic examples

Для каждого источника создать отдельный `convert_*.py`.

Каждый converter должен:

1. читать raw dataset;
2. приводить его к canonical `examples.jsonl`;
3. сохранять original metadata;
4. не удалять важные labels;
5. не делать scoring/generation;
6. не добавлять synthetic data без явной пометки `source = synthetic_*`.

## Phase 4. Построить contrast groups

Создать `build_contrasts.py`.

Три режима построения contrasts:

### Режим A. Dataset уже содержит good/bad labels

Пример: HaluEval candidate answer labels.

```text
bad answer → positive
good answer → negative
```

### Режим B. Dataset содержит prompt-only eval

Пример: EM eval questions, OR-Bench prompts.

Действие:

```text
оставить как eval examples
не строить contrast автоматически
для extraction использовать synthetic paired completions
```

### Режим C. Dataset содержит scenario

Пример: OpenDeception, AgentHarm.

Действие:

```text
оставить scenario как eval_agentic/eval_external
для extraction построить synthetic good/bad trajectories
```

## Phase 5. Сгенерировать synthetic contrast pairs

Создать `build_synthetic_contrasts.py`.

Целевые размеры synthetic/paired extraction:

```text
hallucination:      200 pairs
sycophancy:         200 pairs
deception:          150 pairs
overconfidence:     150 pairs
premature_refusal:  150 pairs
unsafe_planning:    100 pairs
```

Для каждой пары сохранять два atomic examples:

```text
<id>_bad
<id>_good
```

и один contrast group.

Пример ID:

```text
synthetic_sycophancy_0001_bad
synthetic_sycophancy_0001_good
contrast_sycophancy_0001
```

## Phase 6. Сэмплировать eval subsets

Создать balanced eval:

```text
hallucination:
  200 in-domain
  1000 HaluEval external

sycophancy:
  200 in-domain
  300 external

deception:
  100 in-domain
  44 EM prompts
  50 OpenDeception scenarios

overconfidence:
  200 shared factuality
  300–500 SimpleQA Verified

premature_refusal:
  300 benign-sensitive
  150 unsafe-control

unsafe_planning:
  100 planning prompts
  20–50 AgentHarm tasks
```

## Phase 7. Validate

Запустить:

```bash
python scripts/validate_schema.py --examples processed/examples.jsonl --contrasts processed/contrasts.jsonl
python scripts/compute_stats.py --examples processed/examples.jsonl --contrasts processed/contrasts.jsonl --out processed/stats.json
python scripts/sample_for_manual_review.py --n 100
```

В `source_report.md` явно указать:

```text
какие источники скачались
какие не скачались
сколько examples получилось
сколько contrast pairs получилось
какие поля неполные
какие источники требуют ручной проверки лицензии
```

---

## 12. Что не делать

1. Не делать 6 независимых бенчмарков с разными форматами.
2. Не хранить full activations в JSON.
3. Не использовать TruthfulQA/SimpleQA как ядро проекта.
4. Не полагаться только на LLM-as-a-judge.
5. Не считать deception-vector по одним только final-answer tokens.
6. Не смешивать unsafe refusal control и over-refusal eval.
7. Не использовать harmful benchmarks как extraction core без safety-control separation.
8. Не генерировать synthetic examples без поля `source = synthetic_*`.
9. Не удалять original IDs и source metadata.
10. Не запускать model generation на этапе сборки бенча, кроме synthetic-pair generation, если она явно нужна.

---

## 13. Критерий готовности

Бенч считается собранным, если есть:

```text
processed/examples.jsonl
processed/contrasts.jsonl
processed/stats.json
processed/source_report.md
rubrics/*.md
scripts/validate_schema.py
scripts/compute_stats.py
```

И выполнены минимальные размеры:

```text
total_contrasts >= 500 for v0_small
or
total_contrasts >= 900 for v0

hallucination contrasts >= 100 small / 200 full
sycophancy contrasts >= 100 small / 200 full
deception contrasts >= 100 small / 150 full
overconfidence contrasts >= 75 small / 150 full
premature_refusal contrasts >= 75 small / 150 full
unsafe_planning contrasts >= 50 small / 100 full

external/control examples >= 800 small / 2000 full
```

---

## 14. Рекомендуемый приоритет, если времени мало

Если агент не успевает всё:

```text
Priority 1:
  Persona Vectors artifacts
  Emergent Misalignment evals
  HaluEval
  OR-Bench / XSTest
  synthetic contrast pairs

Priority 2:
  OpenDeception
  StrongREJECT
  SimpleQA Verified

Priority 3:
  AgentHarm
  Machiavelli
  multilingual subset
  temporal subset
```

Минимальный полезный результат:

```text
500 contrast pairs
1000 eval/control examples
rubrics
schema validator
source report
```

---

## 15. Ссылки на основные источники

- Persona Vectors repository: https://github.com/safety-research/persona_vectors
- Persona Vectors paper: https://arxiv.org/abs/2507.21509
- Emergent Misalignment repository: https://github.com/emergent-misalignment/emergent-misalignment/
- Emergent Misalignment paper: https://arxiv.org/abs/2502.17424
- OpenAI Persona Features repository: https://github.com/openai/emergent-misalignment-persona-features
- Persona Features Control Emergent Misalignment paper: https://arxiv.org/abs/2506.19823
- HaluEval repository: https://github.com/RUCAIBox/HaluEval
- HaluEval paper: https://aclanthology.org/2023.emnlp-main.397/
- XSTest repository: https://github.com/paul-rottger/xstest
- XSTest HuggingFace: https://huggingface.co/datasets/walledai/XSTest
- OR-Bench paper: https://arxiv.org/abs/2405.20947
- StrongREJECT repository: https://github.com/alexandrasouly/strongreject
- StrongREJECT package: https://github.com/dsbowen/strong_reject
- Machiavelli benchmark: https://aypan17.github.io/machiavelli/
- Machiavelli repository: https://github.com/aypan17/machiavelli
- AgentHarm paper: https://arxiv.org/abs/2410.09024
- AgentHarm dataset: https://huggingface.co/datasets/ai-safety-institute/AgentHarm
- OpenDeception repository: https://github.com/Simoniracle/OpenDeception-Framework
- OpenDeception paper: https://arxiv.org/abs/2504.13707
- SimpleQA Verified paper: https://arxiv.org/abs/2509.07968
- SimpleQA Verified benchmark: https://www.kaggle.com/benchmarks/deepmind/simpleqa-verified
